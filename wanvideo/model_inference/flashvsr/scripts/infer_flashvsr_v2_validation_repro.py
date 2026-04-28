import argparse
import json
import os
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import VideoData, save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v2 import (
    WanFixedPromptFlashVSRStage1Pipeline,
    _tensor_video_to_pil_frames,
    collect_fixed_validation_samples,
    flashvsr_stage1_export,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外部完整复现 train_flashvsr_stage1_v2.py 的 validation 路径。")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--save_tensor_pt", action="store_true", default=False)
    parser.add_argument("--also_run_mp4_roundtrip", action="store_true", default=False)
    parser.add_argument("--disable_mp4_roundtrip", action="store_true", default=False)
    parser.add_argument("--lq_proj_scale_override", type=float, default=None)
    parser.add_argument("--num_inference_steps_override", type=int, default=None)
    parser.add_argument("--fps_override", type=int, default=None)
    return parser.parse_args()


def parse_sample_indices(raw_value: str, default_count: int) -> List[int]:
    if not raw_value.strip():
        return list(range(default_count))
    indices: List[int] = []
    for part in raw_value.replace(" ", "").split(","):
        if not part:
            continue
        indices.append(int(part))
    return indices


def load_resolved_args(experiment_dir: str) -> Dict[str, Any]:
    candidate_paths = [
        os.path.join(experiment_dir, "output", "resolved_args.yaml"),
        os.path.join(experiment_dir, "resolved_args.yaml"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                return yaml.safe_load(file)
    raise FileNotFoundError(f"resolved_args.yaml not found under experiment_dir={experiment_dir}")


def build_streaming_dataset(args_dict: Dict[str, Any]) -> FlashVSRStreamingDataset:
    return FlashVSRStreamingDataset(
        internal_url=args_dict.get("internal_url"),
        metadata_url=args_dict.get("metadata_url"),
        metadata_source=args_dict.get("metadata_source"),
        max_parquet_records=args_dict.get("max_parquet_records"),
        min_overall_score=args_dict.get("min_overall_score", -1.0),
        require_qwen35_parse_success=args_dict.get("require_qwen35_parse_success", False),
        image_internal_url=args_dict.get("image_internal_url", ""),
        image_dataset_prob=args_dict.get("image_dataset_prob", 0.0),
        height=args_dict["height"],
        width=args_dict["width"],
        num_frames=args_dict["num_frames"],
        stride=args_dict.get("stride", 1),
        max_source_frames=args_dict.get("max_source_frames"),
        enable_degradation=args_dict.get("enable_degradation", True),
        degradation_config_path=args_dict.get("degradation_config_path"),
        degradation_seed=args_dict.get("degradation_seed"),
        hq_prefix_frames=args_dict.get("hq_prefix_frames", 0),
        control_dropout_prob=args_dict.get("control_dropout_prob", 0.0),
        shuffle_buffer=args_dict.get("shuffle_buffer", 128),
        global_seed=args_dict.get("global_seed"),
        output_tensors=True,
    )


def parse_model_paths(raw_model_paths: Any) -> List[str]:
    if isinstance(raw_model_paths, list):
        return raw_model_paths
    if isinstance(raw_model_paths, str):
        return json.loads(raw_model_paths)
    raise TypeError(f"Unsupported model_paths type: {type(raw_model_paths)}")


def build_validation_model_configs(model_paths: List[str]) -> List[ModelConfig]:
    if not model_paths:
        raise ValueError("resolved_args.model_paths is empty")
    base_model_dir = str(Path(model_paths[0]).resolve().parent)
    return [
        ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
    ]


def normalize_exported_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("pipe.dit.") or key.startswith("pipe.lq_proj_in.") for key in state_dict):
        return flashvsr_stage1_export(state_dict)
    return state_dict


def split_flashvsr_ckpt(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    exported_state = normalize_exported_state(state_dict)
    lq_proj_state: Dict[str, torch.Tensor] = {}
    lora_state: Dict[str, torch.Tensor] = {}
    other_state: Dict[str, torch.Tensor] = {}
    for key, value in exported_state.items():
        if key.startswith("lq_proj_in."):
            lq_proj_state[key[len("lq_proj_in."):]] = value
        elif "lora_" in key:
            lora_state[key] = value
        else:
            other_state[key] = value
    return lq_proj_state, lora_state, other_state


def infer_lq_proj_layer_num(lq_proj_state: Dict[str, torch.Tensor]) -> int | None:
    indices: List[int] = []
    prefix = "linear_layers."
    for key in lq_proj_state:
        if key.startswith(prefix):
            layer_id = key[len(prefix):].split(".", 1)[0]
            if layer_id.isdigit():
                indices.append(int(layer_id))
    if not indices:
        return None
    return max(indices) + 1


def pil_frames_to_unit_tensor(frames: List[Any]) -> torch.Tensor:
    tensors = []
    for frame in frames:
        array = np.array(frame, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def tensor_stats_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    diff = a - b
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff ** 2).mean()).item()),
    }


def export_sample_inputs(sample_dir: str, sample: Dict[str, Any], fps: int, save_tensor_pt: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    hr_tensor = sample["video"].detach().cpu().clone()
    lq_tensor = sample["lq_video"].detach().cpu().clone()
    hr_frames = _tensor_video_to_pil_frames(hr_tensor)
    lq_frames = _tensor_video_to_pil_frames(lq_tensor)
    save_video(hr_frames, os.path.join(sample_dir, "hr_from_tensor.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    save_video(lq_frames, os.path.join(sample_dir, "lq_from_tensor.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    if save_tensor_pt:
        torch.save(hr_tensor, os.path.join(sample_dir, "hr_tensor.pt"))
        torch.save(lq_tensor, os.path.join(sample_dir, "lq_tensor.pt"))
    return hr_tensor, lq_tensor


def run_sr_from_tensor(
    pipe: WanFixedPromptFlashVSRStage1Pipeline,
    sample_dir: str,
    hr_tensor: torch.Tensor,
    lq_tensor: torch.Tensor,
    seed: int,
    fps: int,
    num_inference_steps: int,
) -> None:
    sr_frames = pipe.infer_from_lq(
        lq_video=lq_tensor.unsqueeze(0),
        height=int(hr_tensor.shape[-2]),
        width=int(hr_tensor.shape[-1]),
        num_frames=int(hr_tensor.shape[0]),
        seed=seed,
        rand_device="cpu",
        num_inference_steps=num_inference_steps,
        tiled=True,
        output_type="quantized",
    )
    save_video(sr_frames, os.path.join(sample_dir, "sr_from_tensor.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def run_sr_from_mp4_roundtrip(
    pipe: WanFixedPromptFlashVSRStage1Pipeline,
    sample_dir: str,
    hr_tensor: torch.Tensor,
    lq_tensor: torch.Tensor,
    seed: int,
    fps: int,
    num_inference_steps: int,
) -> Dict[str, float]:
    roundtrip_frames = VideoData(
        os.path.join(sample_dir, "lq_from_tensor.mp4"),
        height=int(hr_tensor.shape[-2]),
        width=int(hr_tensor.shape[-1]),
    ).raw_data()
    roundtrip_frames = roundtrip_frames[: int(hr_tensor.shape[0])]
    save_video(roundtrip_frames, os.path.join(sample_dir, "lq_from_mp4_roundtrip.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    roundtrip_tensor = pil_frames_to_unit_tensor(roundtrip_frames)
    torch.save(roundtrip_tensor, os.path.join(sample_dir, "lq_tensor_from_mp4_roundtrip.pt"))
    sr_frames = pipe.infer_from_lq(
        lq_video=roundtrip_tensor.unsqueeze(0),
        height=int(hr_tensor.shape[-2]),
        width=int(hr_tensor.shape[-1]),
        num_frames=int(hr_tensor.shape[0]),
        seed=seed,
        rand_device="cpu",
        num_inference_steps=num_inference_steps,
        tiled=True,
        output_type="quantized",
    )
    save_video(sr_frames, os.path.join(sample_dir, "sr_from_mp4_roundtrip.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return tensor_stats_diff(lq_tensor, roundtrip_tensor)


def main() -> None:
    args = parse_args()
    resolved_args = load_resolved_args(args.experiment_dir)

    dataset = build_streaming_dataset(resolved_args)
    total_samples = args.num_samples if args.num_samples is not None else int(resolved_args.get("validation_num_samples", 3))
    validation_samples = collect_fixed_validation_samples(dataset, total_samples)
    sample_indices = parse_sample_indices(args.sample_indices, len(validation_samples))

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)
    lq_proj_layer_num = resolved_args.get("lq_proj_layer_num") or infer_lq_proj_layer_num(lq_proj_state)
    model_paths = parse_model_paths(resolved_args["model_paths"])

    pipe = WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=build_validation_model_configs(model_paths),
        prompt_tensor_path=resolved_args["prompt_tensor_path"],
        lq_proj_layer_num=lq_proj_layer_num,
    )
    pipe.lq_proj_scale = float(args.lq_proj_scale_override if args.lq_proj_scale_override is not None else resolved_args.get("lq_proj_scale", 1.0))
    missing, unexpected = pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if missing:
        print(f"warning: missing lq_proj keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected lq_proj keys: {len(unexpected)}")
    pipe.clear_lora(verbose=0)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)

    fps = int(args.fps_override if args.fps_override is not None else resolved_args.get("validation_fps", 8))
    num_inference_steps = int(args.num_inference_steps_override if args.num_inference_steps_override is not None else resolved_args.get("validation_num_inference_steps", 10))
    seed_base = int(resolved_args.get("global_seed", 20260407))
    step_value = args.step

    os.makedirs(args.output_dir, exist_ok=True)
    summary: Dict[str, Any] = {
        "experiment_dir": args.experiment_dir,
        "checkpoint_path": args.checkpoint_path,
        "step": step_value,
        "lq_proj_scale": pipe.lq_proj_scale,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "lq_proj_layer_num": lq_proj_layer_num,
        "sample_indices": sample_indices,
        "other_ckpt_key_count": len(other_state),
        "samples": [],
    }
    print(
        json.dumps(
            {
                "checkpoint_path": args.checkpoint_path,
                "lq_proj_scale": pipe.lq_proj_scale,
                "num_inference_steps": num_inference_steps,
                "fps": fps,
                "sample_indices": sample_indices,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for sample_index in sample_indices:
        sample = deepcopy(validation_samples[sample_index])
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_index:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        hr_tensor, lq_tensor = export_sample_inputs(sample_dir, sample, fps=fps, save_tensor_pt=args.save_tensor_pt)
        sample_seed_raw = sample.get("sample_seed", -1)
        sample_seed = int(sample_seed_raw.item()) if torch.is_tensor(sample_seed_raw) else int(sample_seed_raw)
        infer_seed = seed_base + sample_index

        run_sr_from_tensor(
            pipe=pipe,
            sample_dir=sample_dir,
            hr_tensor=hr_tensor,
            lq_tensor=lq_tensor,
            seed=infer_seed,
            fps=fps,
            num_inference_steps=num_inference_steps,
        )

        roundtrip_stats = None
        if args.also_run_mp4_roundtrip and not args.disable_mp4_roundtrip:
            roundtrip_stats = run_sr_from_mp4_roundtrip(
                pipe=pipe,
                sample_dir=sample_dir,
                hr_tensor=hr_tensor,
                lq_tensor=lq_tensor,
                seed=infer_seed,
                fps=fps,
                num_inference_steps=num_inference_steps,
            )

        sample_meta = {
            "sample_index": sample_index,
            "sample_id": sample.get("sample_id"),
            "sample_seed": sample_seed,
            "infer_seed": infer_seed,
            "video_shape": list(hr_tensor.shape),
            "lq_shape": list(lq_tensor.shape),
            "lq_proj_scale": pipe.lq_proj_scale,
            "num_inference_steps": num_inference_steps,
            "fps": fps,
            "roundtrip_lq_stats": roundtrip_stats,
        }
        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
            json.dump(sample_meta, file, ensure_ascii=False, indent=2)
        summary["samples"].append(sample_meta)

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"saved_dir={args.output_dir}")


if __name__ == "__main__":
    main()
