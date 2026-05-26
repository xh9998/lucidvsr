#!/usr/bin/env python3
"""Decode Stage3 DMD tensor dumps to videos for visual inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "abs_mean": float(value.abs().mean()),
    }


def _build_vae_pipe(base_model_dir: str, device: str, torch_dtype: torch.dtype):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[ModelConfig(path=os.path.join(base_model_dir, "Wan2.1_VAE.pth"))],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    if pipe.vae is None:
        raise RuntimeError("Wan VAE was not loaded.")
    return pipe


def _decode_and_save(pipe, latent: torch.Tensor, path: Path, fps: int, tiled: bool) -> dict[str, Any]:
    latent = latent.to(dtype=pipe.torch_dtype)
    with torch.inference_mode():
        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(
            latent,
            device=pipe.device,
            tiled=tiled,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        frames = pipe.vae_output_to_video(video)
    save_video(frames, str(path), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return {"num_frames": len(frames), "path": str(path)}


def _copy_fixed_inputs(fixed_root: str | None, out_dir: Path, fps: int) -> dict[str, Any]:
    if not fixed_root:
        return {}
    sample_path = Path(fixed_root) / "sample_00.pt"
    if not sample_path.exists():
        return {"fixed_input_error": f"not found: {sample_path}"}
    payload = torch.load(sample_path, map_location="cpu")
    result: dict[str, Any] = {"fixed_sample": str(sample_path)}
    for key, filename in (("video", "gt_sample_00.mp4"), ("lq_video", "lq_sample_00.mp4")):
        tensor = payload.get(key)
        if torch.is_tensor(tensor):
            frames = v5._tensor_video_to_pil_frames(tensor.detach().cpu().float().clamp(0, 1))
            save_video(frames, str(out_dir / filename), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
            result[key] = {"path": str(out_dir / filename), **_tensor_stats(tensor)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to dmd_tensors.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_dir", default="/mnt/models/Wan2.1-T2V-1.3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--fixed_lqgt_root", default="")
    parser.add_argument("--tiled", action="store_true", default=False)
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.dump, map_location="cpu")
    pipe = _build_vae_pipe(args.base_model_dir, args.device, dtype)

    meta: dict[str, Any] = {
        "dump": args.dump,
        "tiled": bool(args.tiled),
        "decoded": {},
        "tensor_stats": {},
        "fixed_inputs": _copy_fixed_inputs(args.fixed_lqgt_root, out_dir, args.fps),
    }
    names = [
        ("clean_latents", "student_z_pred.mp4"),
        ("real_x0", "g_real_x0.mp4"),
        ("fake_x0", "g_fake_x0.mp4"),
        ("noisy_latents", "shared_noisy_latents_decode.mp4"),
    ]
    for key, filename in names:
        tensor = payload.get(key)
        if not torch.is_tensor(tensor):
            continue
        meta["tensor_stats"][key] = _tensor_stats(tensor)
        meta["decoded"][key] = _decode_and_save(pipe, tensor, out_dir / filename, args.fps, bool(args.tiled))

    with (out_dir / "decode_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    print(f"[decode_dmd_tensor_dump] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
