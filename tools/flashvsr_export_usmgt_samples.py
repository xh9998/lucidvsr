#!/usr/bin/env python3
"""Export visual checks for the isolated v5.3.5 USMGT Stage1 data path."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import torch

from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53_usmgt import (
    ConsistentClipGTSharpen,
    FlashVSRTarStreamingDatasetV53USMGT,
)


def _tensor_to_uint8_video(tensor: torch.Tensor):
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim != 4:
        raise ValueError(f"expected [T,C,H,W], got {tuple(tensor.shape)}")
    return (tensor.permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype("uint8")


def _write_mp4(path: Path, tensor: torch.Tensor, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, _tensor_to_uint8_video(tensor), fps=fps, codec="libx264", pixelformat="yuv420p")


def export_dataset_samples(args: argparse.Namespace) -> None:
    dataset = FlashVSRTarStreamingDatasetV53USMGT(
        image_tar_root_url=args.image_manifest,
        takano_video_tar_url=args.video_manifest,
        yubari_video_tar_url="",
        takano_video_prob=1.0,
        yubari_video_prob=0.0,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        image_branch_num_frames=args.image_branch_num_frames,
        stride=1,
        max_source_frames=args.max_source_frames,
        enable_degradation=True,
        degradation_config_path=args.degradation_config,
        degradation_seed=args.seed,
        shuffle_buffer=1,
        global_seed=args.seed,
        output_tensors=True,
        gt_sharpen=True,
        gt_sharpen_backend=args.gt_sharpen_backend,
        gt_sharpen_device=args.gt_sharpen_device,
        degradation_device=args.degradation_device,
    )
    iterator = iter(dataset)
    for idx in range(args.num_samples):
        sample = next(iterator)
        stem = f"{idx:02d}_{Path(str(sample.get('sample_id', 'sample'))).stem}"
        _write_mp4(args.output_dir / "degraded_video" / f"{stem}_gt_usm.mp4", sample["video"], fps=args.fps)
        _write_mp4(args.output_dir / "degraded_video" / f"{stem}_lq_degraded.mp4", sample["lq_video"], fps=args.fps)
        _write_mp4(args.output_dir / "degraded_image_branch" / f"{stem}_image_gt_usm.mp4", sample["image_video"], fps=args.fps)
        _write_mp4(args.output_dir / "degraded_image_branch" / f"{stem}_image_lq_degraded.mp4", sample["image_lq_video"], fps=args.fps)


def export_sharpness_compare(args: argparse.Namespace) -> None:
    raw_dataset = FlashVSRTarStreamingDatasetV53USMGT(
        image_tar_root_url=args.image_manifest,
        takano_video_tar_url=args.video_manifest,
        yubari_video_tar_url="",
        takano_video_prob=1.0,
        yubari_video_prob=0.0,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        image_branch_num_frames=args.image_branch_num_frames,
        stride=1,
        max_source_frames=args.max_source_frames,
        enable_degradation=False,
        degradation_config_path=args.degradation_config,
        degradation_seed=args.seed,
        shuffle_buffer=1,
        global_seed=args.seed,
        output_tensors=True,
        gt_sharpen=False,
    )
    sharpener = ConsistentClipGTSharpen(
        backend=args.gt_sharpen_backend,
        device=args.gt_sharpen_device,
    )
    iterator = raw_dataset.validation_video_iterator()
    for idx in range(args.num_sharpness):
        raw = next(iterator)
        raw_video = raw["video"]
        # Reconstruct PIL frames for the same USM implementation used by the dataset.
        frames_np = _tensor_to_uint8_video(raw_video)
        from PIL import Image

        raw_frames = [Image.fromarray(frame) for frame in frames_np]
        sharp_frames = sharpener.sharpen_batch(raw_frames)
        sharp_tensor = torch.stack([raw_dataset._pil_to_tensor(frame) for frame in sharp_frames], dim=0)
        stem = f"{idx:02d}_{Path(str(raw.get('sample_id', 'sample'))).stem}"
        _write_mp4(args.output_dir / "sharpness_compare" / f"{stem}_gt_raw.mp4", raw_video, fps=args.fps)
        _write_mp4(args.output_dir / "sharpness_compare" / f"{stem}_gt_usm.mp4", sharp_tensor, fps=args.fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--degradation-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=89)
    parser.add_argument("--image-branch-num-frames", type=int, default=5)
    parser.add_argument("--max-source-frames", type=int, default=160)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-sharpness", type=int, default=5)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026051602)
    parser.add_argument("--gt-sharpen-backend", default="torch")
    parser.add_argument("--gt-sharpen-device", default="auto")
    parser.add_argument("--degradation-device", default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_dataset_samples(args)
    export_sharpness_compare(args)
    print(f"exported to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
