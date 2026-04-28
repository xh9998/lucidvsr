import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import accelerate
import torch
import yaml
from torch.distributed.elastic.multiprocessing.errors import record

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, launch_training_task
from diffsynth.diffusion.loss import FlowMatchSFTLoss
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_dit_stage2_v6 import clear_stage2_caches, enable_stage2_causal_attention, set_stage2_grid
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_ShapeChecker,
)
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v5 import FlashVSRTarStreamingDatasetV5
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5


class FlashVSRStage2Pipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        prompt_tensor_path=None,
        lq_proj_checkpoint=None,
        lq_proj_layer_num=None,
        zero_init_lq_proj_in=True,
        stage2_attention_mode: str = "dense_time_causal",
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=None,
        )
        pipe.__class__ = FlashVSRStage2Pipeline
        pipe.prompt_tensor_path = prompt_tensor_path
        pipe.fixed_prompt_tensor = None
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            v5.FlashVSRUnit_FixedPrompt(),
            v5.WanVideoUnit_InputVideoEmbedderV5(),
            v5.FlashVSRUnit_LQVideoEmbedder(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage2_model_fn
        pipe.compilable_models = ["dit"]
        pipe.debug_tensor_dump_dir = None
        pipe.lq_proj_scale = 1.0
        pipe.stage2_attention_mode = stage2_attention_mode
        enable_stage2_causal_attention(pipe.dit, mode=stage2_attention_mode)

        effective_lq_proj_layers = 1 if lq_proj_layer_num is None else int(lq_proj_layer_num)
        pipe.lq_proj_in = v5.FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=effective_lq_proj_layers,
            zero_init_output=zero_init_lq_proj_in and lq_proj_checkpoint is None,
        ).to(device=device, dtype=torch_dtype)
        if lq_proj_checkpoint is not None:
            state_dict = torch.load(lq_proj_checkpoint, map_location="cpu")
            pipe.lq_proj_in.load_state_dict(state_dict, strict=True)
        return pipe


def flashvsr_stage2_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    batch_size = latents.shape[0]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(f"Context batch mismatch: context={tuple(context.shape)} latents={tuple(latents.shape)}")
    context = dit.text_embedding(context)

    x, (f, h, w) = dit.patchify(latents)
    x = x.flatten(2).transpose(1, 2)
    attention_mode = getattr(dit, "flashvsr_stage2_attention_mode", "dense_time_causal")
    if attention_mode == "block_streaming_causal":
        raise NotImplementedError(
            "block_streaming_causal must chunk the latent sequence as the official FlashVSR inference path does "
            "(first f=6, later f=2, with cache and overlap/buffer handling). "
            "The dense_time_causal mode is the current v6.0 smoke baseline."
        )
    set_stage2_grid(dit, (int(f), int(h), int(w)))
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if lq_latents is not None:
        expected_tokens = x.shape[1]
        for layer_latents in lq_latents:
            if layer_latents.shape[1] != expected_tokens:
                raise ValueError(
                    f"Stage2 video-only requires lq token count to match x: "
                    f"x={expected_tokens}, lq={layer_latents.shape[1]}, grid={(f, h, w)}"
                )

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + lq_latents[block_id] * lq_proj_scale
        if dit.training:
            def block_forward(hidden_states, block=block, context=context, t_mod=t_mod, freqs=freqs):
                return block(hidden_states, context, t_mod, freqs)

            x = gradient_checkpoint_forward(
                block_forward,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
            )
        else:
            x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


class FlashVSRStage2TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        prompt_tensor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=384,
        lora_checkpoint=None,
        lq_proj_checkpoint=None,
        resume_stage1_checkpoint=None,
        lq_proj_layer_num=None,
        lq_proj_scale: float = 1.0,
        zero_init_lq_proj_in=True,
        freeze_lq_proj_in: bool = False,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        stage2_attention_mode: str = "dense_time_causal",
        fp8_models=None,
        offload_models=None,
        device="cpu",
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            raise ValueError("Stage2 v6 requires gradient checkpointing for practical memory use.")
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        self.pipe = FlashVSRStage2Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            prompt_tensor_path=prompt_tensor_path,
            lq_proj_checkpoint=lq_proj_checkpoint,
            lq_proj_layer_num=lq_proj_layer_num,
            zero_init_lq_proj_in=zero_init_lq_proj_in,
            stage2_attention_mode=stage2_attention_mode,
        )
        self.pipe.lq_proj_scale = float(lq_proj_scale)
        self.pipe = self.split_pipeline_units("sft", self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            task="sft",
        )
        if resume_stage1_checkpoint is not None:
            if lora_base_model is None:
                raise ValueError("resume_stage1_checkpoint requires lora_base_model.")
            self._load_stage1_resume_checkpoint(resume_stage1_checkpoint, lora_base_model)
        if freeze_lq_proj_in:
            for param in self.pipe.lq_proj_in.parameters():
                param.requires_grad = False
            self.pipe.lq_proj_in.eval()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        export_names = self.trainable_param_names()
        export_names.update(name for name, _ in self.named_parameters() if name.startswith("pipe.lq_proj_in."))
        state_dict = {name: param for name, param in state_dict.items() if name in export_names}
        if remove_prefix is not None:
            state_dict = {
                (name[len(remove_prefix):] if name.startswith(remove_prefix) else name): param
                for name, param in state_dict.items()
            }
        return state_dict

    def _load_stage1_resume_checkpoint(self, checkpoint_path: str, lora_base_model: str) -> None:
        state_dict = v5.load_state_dict(checkpoint_path, device="cpu")
        lq_proj_state, lora_state, _ = v5.flashvsr_stage1_split_exported_state(state_dict)
        if lq_proj_state:
            self.pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
            print(f"Stage2 v6 loaded lq_proj_in from {checkpoint_path}, keys={len(lq_proj_state)}", flush=True)
        if lora_state:
            lora_model = getattr(self.pipe, lora_base_model)
            mapped_lora_state = self.mapping_lora_state_dict(lora_state)
            result = lora_model.load_state_dict(mapped_lora_state, strict=False)
            print(
                f"Stage2 v6 loaded LoRA from {checkpoint_path}, "
                f"keys={len(mapped_lora_state)}, missing={len(result[0])}, unexpected={len(result[1])}",
                flush=True,
            )

    def get_pipeline_inputs(self, data):
        if not torch.is_tensor(data["video"]) or not torch.is_tensor(data["lq_video"]):
            raise ValueError("Stage2 v6 expects tensor-collated video/lq_video.")
        video = data["video"]
        return (
            {
                "input_video": video,
                "lq_video": data["lq_video"],
                "height": int(video.shape[-2]),
                "width": int(video.shape[-1]),
                "num_frames": int(video.shape[1]),
                "cfg_scale": 1.0,
                "tiled": False,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "framewise_decoding": False,
                "vace_reference_image": None,
                "seed": 0,
                "lq_proj_scale": self.pipe.lq_proj_scale,
            },
            {},
            {},
        )

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        merged_inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            merged_inputs = self.pipe.unit_runner(unit, self.pipe, *merged_inputs)
        return FlowMatchSFTLoss(self.pipe, **merged_inputs[0], **merged_inputs[1])


def _stage2_parser():
    parser = v5.flashvsr_parser()
    parser.add_argument(
        "--stage2_attention_mode",
        type=str,
        default="dense_time_causal",
        choices=("dense_time_causal", "block_streaming_causal"),
        help="Stage2 causal self-attention backend. dense_time_causal is correctness fallback; block_streaming_causal follows FlashVSR's f=6/f=2 cache contract and requires the chunked model_fn path.",
    )
    return parser


def parse_stage2_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = _stage2_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**v5._flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.image_tar_url is not None:
        args.picked17k_image_tar_url = args.image_tar_url
    else:
        args.image_tar_url = args.picked17k_image_tar_url
    return args


def _combined_video_url(args) -> Optional[str]:
    roots = [root for root in (args.yubari_video_tar_url, args.takano_video_tar_url, args.internal_url) if root]
    return ",".join(roots) if roots else None


@record
def main(argv=None):
    args = parse_stage2_args(argv)
    accelerator = accelerate.Accelerator()
    v5.configure_deepspeed_runtime(accelerator, args)
    if accelerator.is_main_process:
        v5.dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)

    video_url = _combined_video_url(args)
    if not video_url:
        raise ValueError("Stage2 v6 is video-only and requires yubari_video_tar_url, takano_video_tar_url, or internal_url.")
    dataset = FlashVSRStreamingDataset(
        internal_url=video_url,
        metadata_url=args.metadata_url,
        metadata_source=args.metadata_source,
        max_parquet_records=args.max_parquet_records,
        min_overall_score=args.min_overall_score,
        require_qwen35_parse_success=args.require_qwen35_parse_success,
        image_internal_url=None,
        image_dataset_prob=0.0,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=args.enable_degradation,
        degradation_config_path=args.degradation_config_path,
        degradation_seed=args.degradation_seed,
        hq_prefix_frames=args.hq_prefix_frames,
        control_dropout_prob=args.control_dropout_prob,
        shuffle_buffer=args.shuffle_buffer,
        global_seed=args.global_seed,
        output_tensors=True,
    )

    model = FlashVSRStage2TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lq_proj_checkpoint=args.lq_proj_checkpoint,
        resume_stage1_checkpoint=args.resume_stage1_checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        lq_proj_scale=args.lq_proj_scale,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        freeze_lq_proj_in=args.freeze_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        stage2_attention_mode=args.stage2_attention_mode,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    if accelerator.is_local_main_process:
        trainable_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
        print(f"Stage2 v6 attention mode: {args.stage2_attention_mode}", flush=True)
        print(f"Trainable parameter tensors: {len(trainable_params)}", flush=True)
        print(f"Trainable parameter count: {sum(numel for _, numel in trainable_params)}", flush=True)

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt="pipe.",
        save_steps=args.save_steps,
        extra_save_steps=args.extra_save_steps,
        log_loss_steps=args.log_loss_steps,
        validation_callback=None,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
