#!/usr/bin/env python3
"""CPU/GPU degradation ablation using the same path as inference testset export.

The normal x4 evaluation set is generated from GT-sized clips, then Aliyun
degradation removes the final bicubic restore so LQ stays at 1/4 resolution.
This script follows that rule before comparing CPU and GPU degradation.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wanvideo.data.flashvsr.tests.export_inference_testset6_aliyun_x4_lq import (  # noqa: E402
    AliyunVideoCompressionX4LQ,
)


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
DEFAULT_CONFIG = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml"
)


def _list_videos(input_dir: Path, limit: int) -> List[Path]:
    generated_markers = ("lq_", "absdiff", "repeat")
    videos = sorted(
        path
        for path in input_dir.rglob("*")
        if path.suffix.lower() in VIDEO_EXTS
        and not path.name.startswith("._")
        and not path.stem.startswith(generated_markers)
    )
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir}")
    return videos[:limit]


def _read_video(path: Path, max_frames: int) -> torch.Tensor:
    frames: List[torch.Tensor] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_rgb().to_ndarray()
            frames.append(torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0))
            if len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError(f"Could not decode any frame from {path}")
    while len(frames) < max_frames:
        frames.append(frames[-1].clone())
    return torch.stack(frames, dim=0).contiguous()


def _resize_cover_crop(video_tchw: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize preserving aspect ratio, then center-crop to target size."""
    _, _, src_h, src_w = video_tchw.shape
    scale = max(height / src_h, width / src_w)
    new_h = max(height, int(round(src_h * scale)))
    new_w = max(width, int(round(src_w * scale)))
    resized = F.interpolate(video_tchw, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0, 1)
    top = max(0, (new_h - height) // 2)
    left = max(0, (new_w - width) // 2)
    return resized[:, :, top : top + height, left : left + width].contiguous()


def _write_video(video_tchw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_tchw = video_tchw.detach().cpu().float().clamp(0, 1)
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height = int(video_tchw.shape[-2])
        stream.width = int(video_tchw.shape[-1])
        stream.pix_fmt = "yuv420p"
        for frame_tensor in video_tchw:
            arr = (frame_tensor.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _move_params(params: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in params.items():
        moved[key] = value.to(device=device) if torch.is_tensor(value) else value
    return moved


def _summarize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.numel() <= 8:
                summary[key] = tensor.reshape(-1).tolist()
            else:
                summary[key] = {
                    "shape": list(tensor.shape),
                    "mean": float(tensor.float().mean()),
                    "std": float(tensor.float().std()),
                }
        else:
            summary[key] = value
    return summary


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a - b).abs()
    return float(diff.mean()), float(diff.max())


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CPU/GPU degradation ablation.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = _list_videos(input_dir, args.num_videos)
    cpu_device = torch.device("cpu")
    gpu_device = torch.device(args.gpu_device)
    cpu_model = AliyunVideoCompressionX4LQ(config_path=args.config, device="cpu")
    gpu_model = AliyunVideoCompressionX4LQ(config_path=args.config, device=str(gpu_device))
    records = []

    for index, video_path in enumerate(videos):
        seed = int(args.seed) + index
        name = video_path.stem
        raw = _read_video(video_path, max_frames=args.num_frames)
        gt = _resize_cover_crop(raw, height=args.height, width=args.width)

        _set_all_seeds(seed)
        params_cpu = cpu_model._sample_degradation_params(cpu_device)
        with torch.no_grad():
            _set_all_seeds(seed + 10_000)
            cpu_out = cpu_model._apply_degradation(gt.to(cpu_device), _move_params(params_cpu, cpu_device)).detach().cpu()
            _set_all_seeds(seed + 10_000)
            gpu_out = gpu_model._apply_degradation(gt.to(gpu_device), _move_params(params_cpu, gpu_device)).detach().cpu()
            _set_all_seeds(seed + 10_000)
            cpu_repeat = cpu_model._apply_degradation(gt.to(cpu_device), _move_params(params_cpu, cpu_device)).detach().cpu()
            _set_all_seeds(seed + 10_000)
            gpu_repeat = gpu_model._apply_degradation(gt.to(gpu_device), _move_params(params_cpu, gpu_device)).detach().cpu()

        sample_dir = output_dir / f"{index:02d}_{name}"
        _write_video(gt, sample_dir / "gt.mp4", fps=args.fps)
        _write_video(cpu_out, sample_dir / "lq_cpu.mp4", fps=args.fps)
        _write_video(gpu_out, sample_dir / "lq_gpu.mp4", fps=args.fps)
        _write_video(((cpu_out - gpu_out).abs() * args.diff_scale).clamp(0, 1), sample_dir / f"absdiff_x{args.diff_scale:g}.mp4", fps=args.fps)
        _write_video(cpu_repeat, sample_dir / "lq_cpu_repeat.mp4", fps=args.fps)
        _write_video(gpu_repeat, sample_dir / "lq_gpu_repeat.mp4", fps=args.fps)
        with open(sample_dir / "params.json", "w", encoding="utf-8") as file:
            json.dump(_summarize_params(params_cpu), file, ensure_ascii=False, indent=2)

        cpu_gpu_mean, cpu_gpu_max = _diff_stats(cpu_out, gpu_out)
        cpu_repeat_mean, cpu_repeat_max = _diff_stats(cpu_out, cpu_repeat)
        gpu_repeat_mean, gpu_repeat_max = _diff_stats(gpu_out, gpu_repeat)
        record = {
            "index": index,
            "input": str(video_path),
            "seed": seed,
            "gt_shape_tchw": list(gt.shape),
            "lq_shape_tchw": list(cpu_out.shape),
            "cpu_gpu_mean_abs_diff": cpu_gpu_mean,
            "cpu_gpu_max_abs_diff": cpu_gpu_max,
            "cpu_repeat_mean_abs_diff": cpu_repeat_mean,
            "cpu_repeat_max_abs_diff": cpu_repeat_max,
            "gpu_repeat_mean_abs_diff": gpu_repeat_mean,
            "gpu_repeat_max_abs_diff": gpu_repeat_max,
            "cpu_video": str(sample_dir / "lq_cpu.mp4"),
            "gpu_video": str(sample_dir / "lq_gpu.mp4"),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--gpu_device", default="cuda:0")
    parser.add_argument("--diff_scale", type=float, default=8.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
