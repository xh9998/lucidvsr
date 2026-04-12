import argparse
import json
import os
import random
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import accelerate
import torch
import yaml
from torch.distributed.elastic.multiprocessing.errors import record

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import launch_training_task
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v2 import (
    FlashVSRStage1TrainingModule,
    FlashVSRValidationCallback,
    WanFixedPromptFlashVSRStage1Pipeline,
    WanTextPromptLQPipeline,
    _tensor_video_to_pil_frames,
    _flatten_flashvsr_config,
    configure_deepspeed_runtime,
    dump_resolved_args,
    flashvsr_parser,
    flashvsr_stage1_export,
)
from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


class FixedSampleOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, sample_dir: str, global_seed: int = 20260407, shuffle: bool = True):
        super().__init__()
        self.sample_dir = Path(sample_dir)
        self.global_seed = global_seed
        self.shuffle = shuffle
        self.load_from_cache = False
        self.custom_collate_fn = self._collate_fn
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        sample_dirs = sorted(p for p in self.sample_dir.glob("sample_*") if p.is_dir())
        if not sample_dirs:
            raise FileNotFoundError(f"No sample_* directories found under {self.sample_dir}")
        samples: List[Dict[str, Any]] = []
        for index, sample_path in enumerate(sample_dirs):
            hr_path = sample_path / "hr.pt"
            lq_path = sample_path / "lq.pt"
            meta_path = sample_path / "meta.json"
            if not hr_path.exists() or not lq_path.exists():
                raise FileNotFoundError(f"Missing hr.pt/lq.pt under {sample_path}")
            sample: Dict[str, Any] = {
                "video": torch.load(hr_path, map_location="cpu").float(),
                "lq_video": torch.load(lq_path, map_location="cpu").float(),
                "sample_seed": torch.tensor(index, dtype=torch.long),
                "sample_path": str(sample_path),
            }
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as file:
                    sample["meta"] = json.load(file)
            samples.append(sample)
        return samples

    def _clone_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cached: Dict[str, Any] = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                cached[key] = value.clone()
            else:
                cached[key] = deepcopy(value)
        return cached

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        keys = batch[0].keys()
        for key in keys:
            values = [item[key] for item in batch]
            if torch.is_tensor(values[0]):
                merged[key] = torch.stack(values, dim=0)
            else:
                merged[key] = values
        return merged

    def fixed_validation_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        return [self._clone_sample(sample) for sample in self.samples[:num_samples]]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        rng = random.Random(self.global_seed + worker_id)
        indices = list(range(len(self.samples)))
        while True:
            if self.shuffle:
                rng.shuffle(indices)
            for offset, sample_index in enumerate(indices):
                if offset % num_workers != worker_id:
                    continue
                yield self._clone_sample(self.samples[sample_index])


class DebugFlashVSRValidationCallback(FlashVSRValidationCallback):
    def _build_v2_validation_pipe(self, device, torch_dtype):
        return WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            prompt_tensor_path=self.validation_prompt_tensor_path,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
        )

    def _build_wan_text_baseline_pipe(self, device, torch_dtype):
        return WanTextPromptLQPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            tokenizer_config=self.validation_tokenizer_config,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
        )

    def __call__(self, accelerator, model, checkpoint_path: str, step: int):
        if not self.validation_samples:
            return
        validation_dir = os.path.join(self.output_path, "validation", f"step-{step}")
        os.makedirs(validation_dir, exist_ok=True)

        inference_model = model
        pipe = inference_model.pipe
        scheduler_state = {
            "timesteps": pipe.scheduler.timesteps.clone() if hasattr(pipe.scheduler, "timesteps") and pipe.scheduler.timesteps is not None else None,
            "training": getattr(pipe.scheduler, "training", None),
        }
        training_mode = inference_model.training
        inference_model.eval()
        try:
            for sample_index, sample in enumerate(self.validation_samples):
                sample_dir = os.path.join(validation_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                hr_tensor = sample["video"]
                lq_tensor = sample["lq_video"]
                hr_frames = _tensor_video_to_pil_frames(hr_tensor)
                lq_frames = _tensor_video_to_pil_frames(lq_tensor)
                save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

                exported_state = flashvsr_stage1_export(model.state_dict())
                lq_proj_state = {}
                lora_state = {}
                for key, value in exported_state.items():
                    if key.startswith("lq_proj_in."):
                        lq_proj_state[key[len("lq_proj_in."):]] = value.detach().cpu()
                    elif "lora_" in key:
                        lora_state[key] = value.detach().cpu()

                if self.validation_use_wan_text_baseline:
                    if not self.validation_prompt:
                        raise ValueError("validation_prompt must be set when validation_use_wan_text_baseline is enabled.")
                    baseline_pipe = self._build_wan_text_baseline_pipe(device=pipe.device, torch_dtype=pipe.torch_dtype)
                    baseline_pipe.lq_proj_scale = pipe.lq_proj_scale
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
                    if lora_state:
                        baseline_pipe.load_lora(
                            baseline_pipe.dit,
                            state_dict=lora_state,
                            verbose=0,
                        )
                    sr_frames = baseline_pipe.infer_from_lq_text(
                        prompt=self.validation_prompt,
                        negative_prompt=self.validation_negative_prompt,
                        lq_video=lq_tensor.unsqueeze(0),
                        height=int(hr_tensor.shape[-2]),
                        width=int(hr_tensor.shape[-1]),
                        num_frames=int(hr_tensor.shape[0]),
                        seed=self.seed_base + sample_index,
                        rand_device="cpu",
                        cfg_scale=self.validation_cfg_scale,
                        num_inference_steps=self.num_inference_steps,
                        tiled=True,
                        output_type="quantized",
                    )
                else:
                    baseline_pipe = self._build_v2_validation_pipe(device=pipe.device, torch_dtype=pipe.torch_dtype)
                    baseline_pipe.lq_proj_scale = pipe.lq_proj_scale
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
                    if lora_state:
                        baseline_pipe.load_lora(
                            baseline_pipe.dit,
                            state_dict=lora_state,
                            verbose=0,
                        )
                    sr_frames = baseline_pipe.infer_from_lq(
                        lq_video=lq_tensor.unsqueeze(0),
                        height=int(hr_tensor.shape[-2]),
                        width=int(hr_tensor.shape[-1]),
                        num_frames=int(hr_tensor.shape[0]),
                        seed=self.seed_base + sample_index,
                        rand_device="cpu",
                        num_inference_steps=self.num_inference_steps,
                        tiled=True,
                        output_type="quantized",
                    )
                save_video(sr_frames, os.path.join(sample_dir, "sr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "checkpoint_path": checkpoint_path,
                            "step": step,
                            "sample_index": sample_index,
                            "validation_mode": "wan_text_baseline" if self.validation_use_wan_text_baseline else "v2_wan_fixed_prompt_projection",
                            "validation_cfg_scale": self.validation_cfg_scale,
                            "sample_seed": int(sample.get("sample_seed", torch.tensor(-1)).item() if torch.is_tensor(sample.get("sample_seed")) else sample.get("sample_seed", -1)),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
        finally:
            inference_model.train(training_mode)
            if scheduler_state["training"]:
                pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
            else:
                if scheduler_state["timesteps"] is not None:
                    pipe.scheduler.timesteps = scheduler_state["timesteps"]
                if scheduler_state["training"] is not None:
                    pipe.scheduler.training = scheduler_state["training"]


def flashvsr_debug_parser():
    parser = flashvsr_parser()
    for action in parser._actions:
        if action.dest == "dataset_mode":
            action.choices = tuple(list(action.choices) + ["debug_fixed"])
            action.default = "debug_fixed"
    parser.add_argument("--debug_sample_dir", type=str, default=None, help="Directory containing exported sample_xxx/hr.pt and lq.pt files.")
    parser.add_argument("--debug_shuffle", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True)
    return parser


def parse_flashvsr_debug_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = flashvsr_debug_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**_flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.dataset_mode == "debug_fixed" and not args.debug_sample_dir:
        parser.error("--debug_sample_dir is required when dataset_mode=debug_fixed.")
    return args


def build_debug_dataset(args):
    if args.dataset_mode != "debug_fixed":
        raise ValueError(f"train_flashvsr_stage1_v2_debug.py only supports dataset_mode=debug_fixed, got {args.dataset_mode}")
    return FixedSampleOverfitDataset(
        sample_dir=args.debug_sample_dir,
        global_seed=args.global_seed if args.global_seed is not None else 20260407,
        shuffle=args.debug_shuffle,
    )


@record
def main():
    def _flashvsr_excepthook(exc_type, exc_value, exc_traceback):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(
            f"[fatal rank={rank} local_rank={local_rank}] "
            f"{getattr(exc_type, '__name__', str(exc_type))}: {exc_value}",
            flush=True,
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _flashvsr_excepthook
    args = parse_flashvsr_debug_args()
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
    configure_deepspeed_runtime(accelerator, args)

    if accelerator.is_main_process:
        dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[stage rank={rank} local_rank={local_rank}] about to build fixed-sample dataset", flush=True)
    dataset = build_debug_dataset(args)
    print(
        f"[stage rank={rank} local_rank={local_rank}] fixed-sample dataset ready "
        f"sample_dir={args.debug_sample_dir} num_samples={len(dataset.samples)}",
        flush=True,
    )

    model = FlashVSRStage1TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lq_proj_checkpoint=args.lq_proj_checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        lq_proj_scale=args.lq_proj_scale,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        debug_tensor_dump_dir=args.debug_tensor_dump_dir,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )

    validation_callback = None
    if args.validation_num_samples > 0 and accelerator.is_main_process:
        validation_samples = dataset.fixed_validation_samples(args.validation_num_samples)
        validation_prompt = None
        if args.validation_prompt_file:
            with open(args.validation_prompt_file, "r", encoding="utf-8") as file:
                validation_prompt = file.read().strip()
        model_paths = json.loads(args.model_paths) if args.model_paths is not None else []
        if not model_paths:
            raise ValueError("V2 debug validation requires model_paths to locate the base Wan model.")
        base_model_dir = str(Path(model_paths[0]).resolve().parent)
        if args.validation_use_wan_text_baseline:
            validation_model_configs = [
                ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(Path(base_model_dir) / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
            ]
            validation_tokenizer_config = ModelConfig(path=str(Path(base_model_dir) / "google/umt5-xxl"))
        else:
            validation_model_configs = [
                ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
            ]
            validation_tokenizer_config = None
        validation_callback = DebugFlashVSRValidationCallback(
            output_path=args.output_path,
            validation_samples=validation_samples,
            num_inference_steps=args.validation_num_inference_steps,
            fps=args.validation_fps,
            seed_base=(args.global_seed if args.global_seed is not None else 20260407),
            use_wandb=args.use_wandb,
            validation_prompt=validation_prompt,
            validation_negative_prompt=args.validation_negative_prompt,
            validation_cfg_scale=args.validation_cfg_scale,
            validation_use_wan_text_baseline=args.validation_use_wan_text_baseline,
            validation_model_configs=validation_model_configs,
            validation_tokenizer_config=validation_tokenizer_config,
            validation_prompt_tensor_path=args.prompt_tensor_path,
            validation_lq_proj_layer_num=args.lq_proj_layer_num,
        )
        print(f"[stage rank={rank} local_rank={local_rank}] fixed validation samples ready", flush=True)

    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=flashvsr_stage1_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
