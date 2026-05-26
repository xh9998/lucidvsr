#!/usr/bin/env python3
"""Compare Aliyun degradation on CPU vs GPU with fixed sampled params.

This script is intentionally isolated from training code. It samples degradation
parameters once on CPU for each input video, then applies the same parameter set
on CPU and GPU. It can also repeat the same apply path on each device to
separate device implementation differences from hidden randomness.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import av
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wanvideo.data.flashvsr.degradation.aliyun_video_degradation import (  # noqa: E402
    AliyunVideoCompressionDegradationModel,
)


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def _list_videos(input_dir: Path, limit: int) -> List[Path]:
    videos = sorted(
        path
        for path in input_dir.rglob("*")
        if path.suffix.lower() in VIDEO_EXTS and not path.name.startswith("._")
    )
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir}")
    return videos[:limit]


def _read_video(path: Path, max_frames: int) -> torch.Tensor:
    frames: List[torch.Tensor] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = frame.to_rgb().to_ndarray()
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
            frames.append(tensor)
            if len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError(f"Could not decode any frame from {path}")
    return torch.stack(frames, dim=0).contiguous()


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


def _save_png(frame_chw: torch.Tensor, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (frame_chw.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _move_params(params: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _summarize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in params.items():
        if torch.is_tensor(value):
            if value.numel() <= 8:
                summary[key] = value.detach().cpu().reshape(-1).tolist()
            else:
                summary[key] = {
                    "shape": list(value.shape),
                    "mean": float(value.detach().cpu().float().mean()),
                    "std": float(value.detach().cpu().float().std()),
                }
        else:
            summary[key] = value
    return summary


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = _list_videos(input_dir, args.num_videos)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CPU/GPU degradation ablation.")
    gpu_device = torch.device(args.gpu_device)
    cpu_device = torch.device("cpu")

    cpu_model = AliyunVideoCompressionDegradationModel(config_path=args.config, device="cpu")
    gpu_model = AliyunVideoCompressionDegradationModel(config_path=args.config, device=str(gpu_device))
    records = []
    for index, video_path in enumerate(videos):
        seed = int(args.seed) + index
        name = video_path.stem
        video = _read_video(video_path, max_frames=args.num_frames)
        video_bchw = video.to(dtype=torch.float32)

        _set_all_seeds(seed)
        params_cpu = cpu_model._sample_degradation_params(cpu_device)

        with torch.no_grad():
            cpu_out = cpu_model._apply_degradation(video_bchw.to(cpu_device), _move_params(params_cpu, cpu_device)).detach().cpu()
            gpu_out = gpu_model._apply_degradation(video_bchw.to(gpu_device), _move_params(params_cpu, gpu_device)).detach().cpu()
            cpu_repeat_out = None
            gpu_repeat_out = None
            if args.repeat_check:
                cpu_repeat_out = cpu_model._apply_degradation(
                    video_bchw.to(cpu_device), _move_params(params_cpu, cpu_device)
                ).detach().cpu()
                gpu_repeat_out = gpu_model._apply_degradation(
                    video_bchw.to(gpu_device), _move_params(params_cpu, gpu_device)
                ).detach().cpu()

        diff = (cpu_out - gpu_out).abs()
        cpu_repeat_diff = (cpu_out - cpu_repeat_out).abs() if cpu_repeat_out is not None else None
        gpu_repeat_diff = (gpu_out - gpu_repeat_out).abs() if gpu_repeat_out is not None else None
        sample_dir = output_dir / f"{index:02d}_{name}"
        _write_video(video_bchw, sample_dir / "gt.mp4", fps=args.fps)
        _write_video(cpu_out, sample_dir / "lq_cpu.mp4", fps=args.fps)
        _write_video(gpu_out, sample_dir / "lq_gpu.mp4", fps=args.fps)
        _write_video(diff.clamp(0, 1) * args.diff_scale, sample_dir / f"absdiff_x{args.diff_scale:g}.mp4", fps=args.fps)
        if args.repeat_check and cpu_repeat_out is not None and gpu_repeat_out is not None:
            _write_video(cpu_repeat_out, sample_dir / "lq_cpu_repeat.mp4", fps=args.fps)
            _write_video(gpu_repeat_out, sample_dir / "lq_gpu_repeat.mp4", fps=args.fps)
            _write_video(
                cpu_repeat_diff.clamp(0, 1) * args.diff_scale,
                sample_dir / f"cpu_repeat_absdiff_x{args.diff_scale:g}.mp4",
                fps=args.fps,
            )
            _write_video(
                gpu_repeat_diff.clamp(0, 1) * args.diff_scale,
                sample_dir / f"gpu_repeat_absdiff_x{args.diff_scale:g}.mp4",
                fps=args.fps,
            )
        _save_png(video_bchw[0], sample_dir / "frames" / "gt_000.png")
        _save_png(cpu_out[0], sample_dir / "frames" / "lq_cpu_000.png")
        _save_png(gpu_out[0], sample_dir / "frames" / "lq_gpu_000.png")
        _save_png((diff[0] * args.diff_scale).clamp(0, 1), sample_dir / "frames" / f"absdiff_x{args.diff_scale:g}_000.png")
        if args.repeat_check and cpu_repeat_diff is not None and gpu_repeat_diff is not None:
            _save_png(
                (cpu_repeat_diff[0] * args.diff_scale).clamp(0, 1),
                sample_dir / "frames" / f"cpu_repeat_absdiff_x{args.diff_scale:g}_000.png",
            )
            _save_png(
                (gpu_repeat_diff[0] * args.diff_scale).clamp(0, 1),
                sample_dir / "frames" / f"gpu_repeat_absdiff_x{args.diff_scale:g}_000.png",
            )
        with open(sample_dir / "params.json", "w", encoding="utf-8") as file:
            json.dump(_summarize_params(params_cpu), file, ensure_ascii=False, indent=2)
        record = {
            "index": index,
            "input": str(video_path),
            "seed": seed,
            "shape_tchw": list(video_bchw.shape),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "cpu_video": str(sample_dir / "lq_cpu.mp4"),
            "gpu_video": str(sample_dir / "lq_gpu.mp4"),
        }
        if args.repeat_check and cpu_repeat_diff is not None and gpu_repeat_diff is not None:
            record.update(
                {
                    "cpu_repeat_mean_abs_diff": float(cpu_repeat_diff.mean()),
                    "cpu_repeat_max_abs_diff": float(cpu_repeat_diff.max()),
                    "gpu_repeat_mean_abs_diff": float(gpu_repeat_diff.mean()),
                    "gpu_repeat_max_abs_diff": float(gpu_repeat_diff.max()),
                }
            )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default="wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_half.yaml")
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--gpu_device", default="cuda:0")
    parser.add_argument("--diff_scale", type=float, default=8.0)
    parser.add_argument("--repeat_check", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
