#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5


def _read_manifest_entries(path: str, limit: int) -> list[str]:
    entries: list[str] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
            if len(entries) >= limit:
                break
    if len(entries) < limit:
        raise ValueError(f"Manifest {path} only has {len(entries)} usable entries, need {limit}")
    return entries


def _save_preview(tensor: torch.Tensor, path: Path, fps: int) -> None:
    frames = v5._tensor_video_to_pil_frames(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(path), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed LQ/GT tensor pairs for Stage3 overfit debugging.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--degradation_config_path", required=True)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--max_source_frames", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026052500)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # The base streaming dataset only iterates direct video urls, not manifest
    # urls. Expand the small overfit manifest here so exported samples are
    # exactly the fixed videos we want.
    video_urls = _read_manifest_entries(args.manifest, args.num_samples)
    dataset = FlashVSRStreamingDataset(
        internal_url=",".join(video_urls),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=1,
        max_source_frames=args.max_source_frames,
        image_internal_url=None,
        image_dataset_prob=0.0,
        enable_degradation=True,
        degradation_seed=args.seed,
        global_seed=args.seed,
        degradation_config_path=args.degradation_config_path,
        shuffle_buffer=1,
        output_tensors=True,
    )

    metadata = {
        "manifest": args.manifest,
        "degradation_config_path": args.degradation_config_path,
        "num_samples": args.num_samples,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "max_source_frames": args.max_source_frames,
        "seed": args.seed,
        "video_urls": video_urls,
    }
    samples = []
    iterator = iter(dataset)
    for index in range(args.num_samples):
        sample = next(iterator)
        sample_id = str(sample.get("sample_id", f"sample_{index:02d}"))
        sample_seed = int(sample.get("sample_seed", args.seed))
        payload = {
            "video": sample["video"].cpu(),
            "lq_video": sample["lq_video"].cpu(),
            "sample_seed": sample_seed,
            "sample_id": sample_id,
        }
        tensor_path = output_root / f"sample_{index:02d}.pt"
        torch.save(payload, tensor_path)
        _save_preview(payload["video"], output_root / "gt" / f"sample_{index:02d}.mp4", args.fps)
        _save_preview(payload["lq_video"], output_root / "lq" / f"sample_{index:02d}.mp4", args.fps)
        samples.append(
            {
                "index": index,
                "sample_id": sample_id,
                "sample_seed": sample_seed,
                "tensor_path": str(tensor_path),
                "video_shape": list(payload["video"].shape),
                "lq_shape": list(payload["lq_video"].shape),
            }
        )

    metadata["samples"] = samples
    with open(output_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"[done] fixed_lqgt_root={output_root} samples={len(samples)}")


if __name__ == "__main__":
    main()
