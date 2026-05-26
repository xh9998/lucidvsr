import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import accelerate
import torch
import torch.nn.functional as F
import yaml
from torch.distributed.elastic.multiprocessing.errors import record

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.diffusion import ModelLogger, launch_training_task
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5
from wanvideo.model_training.flashvsr import train_flashvsr_stage2_v6_4_lora as v6


def _flatten_video_for_lpips(video: torch.Tensor) -> torch.Tensor:
    """Convert BCHW video tensor from B,C,T,H,W to BT,C,H,W for frame LPIPS."""
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor B,C,T,H,W, got {tuple(video.shape)}")
    return video.permute(0, 2, 1, 3, 4).reshape(-1, video.shape[1], video.shape[3], video.shape[4])


def _weighted_video_mse(pred: torch.Tensor, target: torch.Tensor, *, first_frame_weight: float) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Weighted MSE shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    per_frame = (pred.float() - target.float()).pow(2).mean(dim=(1, 3, 4))
    if per_frame.shape[1] > 0 and float(first_frame_weight) != 1.0:
        weights = torch.ones(per_frame.shape[1], device=per_frame.device, dtype=per_frame.dtype)
        weights[0] = float(first_frame_weight)
        per_frame = per_frame * weights.unsqueeze(0)
        return per_frame.mean()
    return per_frame.mean()


def _weighted_frame_loss(frame_loss: torch.Tensor, *, batch_size: int, num_frames: int, first_frame_weight: float) -> torch.Tensor:
    if frame_loss.numel() != batch_size * num_frames:
        raise ValueError(f"Frame loss size mismatch: got={frame_loss.numel()} expected={batch_size * num_frames}")
    frame_loss = frame_loss.reshape(batch_size, num_frames)
    if num_frames > 0 and float(first_frame_weight) != 1.0:
        weights = torch.ones(num_frames, device=frame_loss.device, dtype=frame_loss.dtype)
        weights[0] = float(first_frame_weight)
        return (frame_loss * weights.unsqueeze(0)).mean()
    return frame_loss.mean()


class _LazyLPIPS(torch.nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as error:
            raise RuntimeError(
                "Stage3 v7-A requires the `lpips` package when stage3_lpips_weight > 0. "
                "Install lpips or set --stage3_lpips_weight 0 for a dry smoke."
            ) from error
        self.loss = lpips.LPIPS(net=net)
        self.loss.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred.float(), target.float()).reshape(-1)


def _get_stage3_lpips(pipe, net: str):
    # Do not assign this module through nn.Module.__setattr__ on `pipe`.
    # Stage3 creates LPIPS after DeepSpeed initialization; registering it as a
    # new frozen child module makes DeepSpeed checkpointing fail because the
    # parameter-name mapping was built before LPIPS existed.
    if hasattr(pipe, "_modules") and "_stage3_lpips" in pipe._modules:
        del pipe._modules["_stage3_lpips"]
    cache = pipe.__dict__.get("_stage3_lpips_cache")
    cache_net = pipe.__dict__.get("_stage3_lpips_cache_net")
    if cache is None or cache_net != net:
        cache = _LazyLPIPS(net=net).to(device=pipe.device)
        cache.eval()
        object.__setattr__(pipe, "_stage3_lpips_cache", cache)
        object.__setattr__(pipe, "_stage3_lpips_cache_net", net)
    return cache


def Stage3AOneStepReconLoss(pipe, *, stage3_recon_num_latents: int, stage3_flow_weight: float,
                            stage3_mse_weight: float, stage3_lpips_weight: float,
                            stage3_lpips_net: str, stage3_first_frame_pixel_weight: float,
                            stage3_first_frame_lpips_weight: float, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    target_latents = inputs["input_latents"]
    noise = torch.randn_like(target_latents)
    inputs["latents"] = pipe.scheduler.add_noise(target_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(target_latents, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    flow_loss = F.mse_loss(noise_pred.float(), training_target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)

    # One-step prediction to sigma=0. This is the v7-A student output z_pred.
    z_pred = pipe.scheduler.step(noise_pred, timestep, inputs["latents"], to_final=True)

    latent_t = int(z_pred.shape[2])
    recon_num = min(max(int(stage3_recon_num_latents), 1), latent_t)
    # Wan decoder is too memory-heavy to decode a full 89-frame clip during
    # training. Decode a temporal prefix only; flow matching still supervises
    # the full latent sequence, while pixel/LPIPS regularizes a real decoded
    # one-step output path.
    z_decode = z_pred[:, :, :recon_num].contiguous()

    pipe.load_models_to_device(["vae"])
    x_pred = pipe.vae.decode(
        z_decode,
        device=pipe.device,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    )
    target_frames = int(x_pred.shape[2])
    x_gt = pipe.preprocess_video(inputs["input_video"]).to(device=pipe.device, dtype=x_pred.dtype)
    x_gt = x_gt[:, :, :target_frames].contiguous()
    if x_gt.shape != x_pred.shape:
        raise ValueError(f"Stage3 v7-A decoded/GT mismatch: pred={tuple(x_pred.shape)} gt={tuple(x_gt.shape)}")

    mse_loss = _weighted_video_mse(
        x_pred,
        x_gt,
        first_frame_weight=stage3_first_frame_pixel_weight,
    )
    if stage3_lpips_weight > 0:
        lpips_module = _get_stage3_lpips(pipe, stage3_lpips_net)
        lpips_loss = lpips_module(_flatten_video_for_lpips(x_pred), _flatten_video_for_lpips(x_gt))
        lpips_loss = _weighted_frame_loss(
            lpips_loss,
            batch_size=int(x_pred.shape[0]),
            num_frames=target_frames,
            first_frame_weight=stage3_first_frame_lpips_weight,
        )
    else:
        lpips_loss = torch.zeros((), device=pipe.device, dtype=torch.float32)

    total = (
        float(stage3_flow_weight) * flow_loss
        + float(stage3_mse_weight) * mse_loss
        + float(stage3_lpips_weight) * lpips_loss
    )
    if not hasattr(pipe, "_stage3_last_losses"):
        pipe._stage3_last_losses = {}
    pipe._stage3_last_losses = {
        "loss": float(total.detach().cpu()),
        "loss_flow": float(flow_loss.detach().cpu()),
        "loss_mse": float(mse_loss.detach().cpu()),
        "loss_lpips": float(lpips_loss.detach().cpu()),
        "recon_latents": int(recon_num),
        "decoded_frames": int(target_frames),
        "first_frame_pixel_weight": float(stage3_first_frame_pixel_weight),
        "first_frame_lpips_weight": float(stage3_first_frame_lpips_weight),
    }
    if os.environ.get("FLASHVSR_STAGE3_DEBUG_LOSS") == "1" and os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            "[stage3_v7_a_loss] "
            f"loss={pipe._stage3_last_losses['loss']:.6f} "
            f"flow={pipe._stage3_last_losses['loss_flow']:.6f} "
            f"mse={pipe._stage3_last_losses['loss_mse']:.6f} "
            f"lpips={pipe._stage3_last_losses['loss_lpips']:.6f} "
            f"recon_latents={recon_num} decoded_frames={target_frames} "
            f"first_frame_pixel_weight={stage3_first_frame_pixel_weight} "
            f"first_frame_lpips_weight={stage3_first_frame_lpips_weight}",
            flush=True,
        )
    return total


class FlashVSRStage3ATrainingModule(v6.FlashVSRStage2TrainingModule):
    def __init__(
        self,
        *args,
        stage3_recon_num_latents: int = 2,
        stage3_flow_weight: float = 1.0,
        stage3_mse_weight: float = 1.0,
        stage3_lpips_weight: float = 2.0,
        stage3_lpips_net: str = "vgg",
        stage3_first_frame_pixel_weight: float = 4.0,
        stage3_first_frame_lpips_weight: float = 4.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stage3_recon_num_latents = int(stage3_recon_num_latents)
        self.stage3_flow_weight = float(stage3_flow_weight)
        self.stage3_mse_weight = float(stage3_mse_weight)
        self.stage3_lpips_weight = float(stage3_lpips_weight)
        self.stage3_lpips_net = str(stage3_lpips_net)
        self.stage3_first_frame_pixel_weight = float(stage3_first_frame_pixel_weight)
        self.stage3_first_frame_lpips_weight = float(stage3_first_frame_lpips_weight)

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        merged_inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            merged_inputs = self.pipe.unit_runner(unit, self.pipe, *merged_inputs)
        return Stage3AOneStepReconLoss(
            self.pipe,
            **merged_inputs[0],
            **merged_inputs[1],
            stage3_recon_num_latents=self.stage3_recon_num_latents,
            stage3_flow_weight=self.stage3_flow_weight,
            stage3_mse_weight=self.stage3_mse_weight,
            stage3_lpips_weight=self.stage3_lpips_weight,
            stage3_lpips_net=self.stage3_lpips_net,
            stage3_first_frame_pixel_weight=self.stage3_first_frame_pixel_weight,
            stage3_first_frame_lpips_weight=self.stage3_first_frame_lpips_weight,
        )


class FlashVSRStage3AValidationCallback(v6.FlashVSRStage2ValidationCallback):
    def __call__(self, accelerator, model, checkpoint_path: str, step: int):
        if not self.validation_samples:
            return
        validation_dir = os.path.join(self.output_path, "validation", f"step-{step}")
        os.makedirs(validation_dir, exist_ok=True)

        pipe = model.pipe
        scheduler_state = {
            "timesteps": pipe.scheduler.timesteps.clone() if getattr(pipe.scheduler, "timesteps", None) is not None else None,
            "training": getattr(pipe.scheduler, "training", None),
        }
        training_mode = model.training
        model.eval()
        try:
            for sample_index, sample in enumerate(self.validation_samples):
                sample_dir = os.path.join(validation_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                hr_tensor = sample["video"]
                lq_tensor = sample["lq_video"]
                hr_frames = v5._tensor_video_to_pil_frames(hr_tensor)
                lq_frames = v5._tensor_video_to_pil_frames(lq_tensor)
                v5.save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                v5.save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

                pipe.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
                inputs_shared = {
                    "input_video": None,
                    "lq_video": lq_tensor.unsqueeze(0),
                    "seed": self.seed_base + sample_index,
                    "rand_device": "cpu",
                    "height": int(hr_tensor.shape[-2]),
                    "width": int(hr_tensor.shape[-1]),
                    "num_frames": int(hr_tensor.shape[0]),
                    "cfg_scale": 1.0,
                    "cfg_merge": False,
                    "tiled": True,
                    "tile_size": (30, 52),
                    "tile_stride": (15, 26),
                    "framewise_decoding": False,
                    "lq_proj_scale": pipe.lq_proj_scale,
                }
                for unit in pipe.units:
                    inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, {}, {})
                pipe.load_models_to_device(pipe.in_iteration_models)
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                timestep = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                noise_pred = pipe.model_fn(**models, **inputs_shared, timestep=timestep)
                latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[0], inputs_shared["latents"], to_final=True)
                pipe.load_models_to_device(["vae"])
                video = pipe.vae.decode(
                    latents,
                    device=pipe.device,
                    tiled=True,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
                sr_frames = pipe.vae_output_to_video(video)
                v5.save_video(sr_frames, os.path.join(sample_dir, "sr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "checkpoint_path": checkpoint_path,
                            "step": int(step),
                            "sample_index": int(sample_index),
                            "validation_mode": "stage3_v7_a_one_step_recon",
                            "input_num_frames": int(hr_tensor.shape[0]),
                            "output_num_frames": len(sr_frames),
                            "sample_seed": v5._serialize_sample_seed(sample.get("sample_seed")),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
        finally:
            model.train(training_mode)
            if scheduler_state["training"]:
                pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
            else:
                if scheduler_state["timesteps"] is not None:
                    pipe.scheduler.timesteps = scheduler_state["timesteps"]
                if scheduler_state["training"] is not None:
                    pipe.scheduler.training = scheduler_state["training"]


def _stage3_parser():
    parser = v6._stage2_parser()
    parser.add_argument(
        "--resume_stage2_checkpoint",
        type=str,
        default=None,
        help="Alias for --resume_stage1_checkpoint in Stage3; should point to a Stage2 sparse-causal checkpoint.",
    )
    parser.add_argument("--stage3_recon_num_latents", type=int, default=2)
    parser.add_argument("--stage3_flow_weight", type=float, default=1.0)
    parser.add_argument("--stage3_mse_weight", type=float, default=1.0)
    parser.add_argument("--stage3_lpips_weight", type=float, default=2.0)
    parser.add_argument("--stage3_lpips_net", type=str, default="vgg", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--stage3_first_frame_pixel_weight", type=float, default=4.0)
    parser.add_argument("--stage3_first_frame_lpips_weight", type=float, default=4.0)
    return parser


def parse_stage3_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = _stage3_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**v5._flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.resume_stage2_checkpoint is not None:
        if args.resume_stage1_checkpoint is not None and args.resume_stage1_checkpoint != args.resume_stage2_checkpoint:
            parser.error("--resume_stage1_checkpoint and --resume_stage2_checkpoint point to different files.")
        args.resume_stage1_checkpoint = args.resume_stage2_checkpoint
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.image_tar_url is not None:
        args.picked17k_image_tar_url = args.image_tar_url
    else:
        args.image_tar_url = args.picked17k_image_tar_url
    if args.num_frames % 8 != 1:
        parser.error("Stage3 v7-A currently follows Stage2 89->85 streaming shape, requiring num_frames % 8 == 1.")
    return args


@record
def main(argv=None):
    def _excepthook(exc_type, exc_value, exc_traceback):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[fatal rank={rank} local_rank={local_rank}] {exc_type.__name__}: {exc_value}", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook
    args = parse_stage3_args(argv)
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    }
    data_loader_config_cls = getattr(accelerate, "DataLoaderConfiguration", None)
    if data_loader_config_cls is not None:
        accelerator_kwargs["dataloader_config"] = data_loader_config_cls(
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    v5.configure_deepspeed_runtime(accelerator, args)
    if accelerator.is_main_process:
        v5.dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)

    dataset = v6.FlashVSRStage2VideoOnlyDataset(
        yubari_video_tar_url=args.yubari_video_tar_url,
        takano_video_tar_url=args.takano_video_tar_url,
        internal_url=args.internal_url,
        yubari_video_prob=args.yubari_video_prob,
        takano_video_prob=args.takano_video_prob,
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
    model = FlashVSRStage3ATrainingModule(
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
        stage2_topk_ratio=args.stage2_topk_ratio,
        stage2_local_num=args.stage2_local_num,
        stage3_recon_num_latents=args.stage3_recon_num_latents,
        stage3_flow_weight=args.stage3_flow_weight,
        stage3_mse_weight=args.stage3_mse_weight,
        stage3_lpips_weight=args.stage3_lpips_weight,
        stage3_lpips_net=args.stage3_lpips_net,
        stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
        stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    if accelerator.is_local_main_process:
        trainable_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
        print("Stage3 v7-A: one-step student + Wan decoder reconstruction smoke path", flush=True)
        print(f"stage3_recon_num_latents={args.stage3_recon_num_latents}", flush=True)
        print(f"stage3_loss_weights flow={args.stage3_flow_weight} mse={args.stage3_mse_weight} lpips={args.stage3_lpips_weight}", flush=True)
        print(
            "stage3_first_frame_weights "
            f"pixel={args.stage3_first_frame_pixel_weight} lpips={args.stage3_first_frame_lpips_weight}",
            flush=True,
        )
        print(f"Stage2 attention mode inherited for student: {args.stage2_attention_mode}", flush=True)
        print(f"Trainable parameter tensors: {len(trainable_params)}", flush=True)
        print(f"Trainable parameter count: {sum(numel for _, numel in trainable_params)}", flush=True)

    validation_callback = None
    if args.validation_num_samples > 0 and accelerator.is_main_process:
        print("Preparing fixed Stage3 v7-A validation samples...", flush=True)
        validation_samples = v5.collect_fixed_validation_samples(dataset, args.validation_num_samples)
        print(f"Prepared {len(validation_samples)} fixed Stage3 v7-A validation samples.", flush=True)
        validation_callback = FlashVSRStage3AValidationCallback(
            output_path=args.output_path,
            validation_samples=validation_samples,
            num_inference_steps=args.validation_num_inference_steps,
            fps=args.validation_fps,
            seed_base=(args.global_seed if args.global_seed is not None else 20260513),
            use_wandb=args.use_wandb,
        )
    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=v5.flashvsr_stage1_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
