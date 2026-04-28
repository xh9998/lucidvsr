import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

import torch
from PIL import Image

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    video = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in video:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _hash_tensor(video: torch.Tensor) -> str:
    return hashlib.sha256(video.detach().cpu().float().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_url", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=160)
    parser.add_argument("--global_seed", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_parquet_records", type=int, default=256)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    dataset = FlashVSRParquetTarDatasetV2(
        metadata_url=args.metadata_url,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        metadata_source="takano",
        enable_degradation=False,
        global_seed=args.global_seed,
        output_tensors=True,
        max_parquet_records=args.max_parquet_records,
    )

    iterator = iter(dataset)
    summary: Dict[str, Any] = {
        "global_seed": args.global_seed,
        "num_samples": args.num_samples,
        "metadata_url": args.metadata_url,
        "samples": [],
    }

    for sample_index in range(args.num_samples):
        sample = next(iterator)
        sample_dir = os.path.join(args.save_dir, f"sample_{sample_index:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        video = sample["video"]
        lq_video = sample["lq_video"]
        hr_frames = _tensor_to_pil_frames(video)
        lq_frames = _tensor_to_pil_frames(lq_video)
        hr_frames[0].save(os.path.join(sample_dir, "hr_first.png"))
        lq_frames[0].save(os.path.join(sample_dir, "lq_first.png"))
        save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=8, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=8, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

        sample_info = {
            "sample_id": sample["sample_id"],
            "sample_seed": int(sample["sample_seed"].item()),
            "source_dataset": sample["source_dataset"],
            "media_path": sample["media_path"],
            "tar_member_path": sample["tar_member_path"],
            "video_hash": _hash_tensor(video),
            "lq_video_hash": _hash_tensor(lq_video),
        }
        with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        summary["samples"].append(sample_info)

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
