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


def _save_sample(sample: Dict[str, Any], sample_dir: str):
    os.makedirs(sample_dir, exist_ok=True)
    video = sample["video"]
    lq_video = sample["lq_video"]
    hr_frames = _tensor_to_pil_frames(video)
    lq_frames = _tensor_to_pil_frames(lq_video)
    hr_frames[0].save(os.path.join(sample_dir, "hr_first.png"))
    lq_frames[0].save(os.path.join(sample_dir, "lq_first.png"))
    save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=8, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=8, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    info = {
        "sample_id": sample["sample_id"],
        "sample_seed": int(sample["sample_seed"].item()),
        "source_dataset": sample["source_dataset"],
        "source_type": sample.get("source_type"),
        "media_path": sample["media_path"],
        "tar_member_path": sample["tar_member_path"],
        "caption_text": sample.get("caption_text"),
        "video_hash": _hash_tensor(video),
        "lq_video_hash": _hash_tensor(lq_video),
        "metadata": sample.get("metadata"),
    }
    with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["takano", "image", "yubari", "mixed"], required=True)
    parser.add_argument("--takano_metadata_url", type=str, default=None)
    parser.add_argument("--image_metadata_url", type=str, default=None)
    parser.add_argument("--image_internal_url", type=str, default=None)
    parser.add_argument("--image_dataset_prob", type=float, default=0.0)
    parser.add_argument("--image_as_single_frame", action="store_true")
    parser.add_argument("--yubari_video_metadata_url", type=str, default=None)
    parser.add_argument("--yubari_sidecar_metadata_url", type=str, default=None)
    parser.add_argument("--yubari_video_tar_url", type=str, default=None)
    parser.add_argument("--yubari_sidecar_tar_url", type=str, default=None)
    parser.add_argument("--yubari_shard_start", type=int, default=None)
    parser.add_argument("--yubari_shard_end", type=int, default=None)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=160)
    parser.add_argument("--enable_degradation", action="store_true")
    parser.add_argument("--degradation_config_path", type=str, default=None)
    parser.add_argument("--global_seed", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_parquet_records", type=int, default=128)
    parser.add_argument("--max_yubari_records", type=int, default=128)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    takano_metadata_url = args.takano_metadata_url if args.mode in ("takano", "mixed") else None
    image_internal_url = args.image_internal_url if args.mode in ("image", "mixed") else None
    image_metadata_url = args.image_metadata_url if args.mode in ("image", "mixed") else None
    yubari_video_metadata_url = args.yubari_video_metadata_url if args.mode in ("yubari", "mixed") else None
    yubari_sidecar_metadata_url = args.yubari_sidecar_metadata_url if args.mode in ("yubari", "mixed") else None
    yubari_video_tar_url = args.yubari_video_tar_url if args.mode in ("yubari", "mixed") else None
    yubari_sidecar_tar_url = args.yubari_sidecar_tar_url if args.mode in ("yubari", "mixed") else None

    dataset = FlashVSRParquetTarDatasetV2(
        metadata_url=takano_metadata_url,
        metadata_source="takano",
        image_metadata_url=image_metadata_url,
        image_internal_url=image_internal_url,
        image_dataset_prob=args.image_dataset_prob,
        image_as_single_frame=args.image_as_single_frame,
        yubari_video_metadata_url=yubari_video_metadata_url,
        yubari_sidecar_metadata_url=yubari_sidecar_metadata_url,
        yubari_video_tar_url=yubari_video_tar_url,
        yubari_sidecar_tar_url=yubari_sidecar_tar_url,
        yubari_shard_start=args.yubari_shard_start,
        yubari_shard_end=args.yubari_shard_end,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=args.enable_degradation,
        degradation_config_path=args.degradation_config_path,
        global_seed=args.global_seed,
        output_tensors=True,
        max_parquet_records=args.max_parquet_records,
        max_yubari_records=args.max_yubari_records,
    )

    iterator = iter(dataset)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "global_seed": args.global_seed,
        "num_samples": args.num_samples,
        "takano_metadata_url": takano_metadata_url,
        "image_internal_url": image_internal_url,
        "image_metadata_url": image_metadata_url,
        "image_dataset_prob": args.image_dataset_prob,
        "yubari_video_metadata_url": yubari_video_metadata_url,
        "yubari_sidecar_metadata_url": yubari_sidecar_metadata_url,
        "yubari_video_tar_url": yubari_video_tar_url,
        "yubari_sidecar_tar_url": yubari_sidecar_tar_url,
        "yubari_shard_start": dataset.yubari_shard_start,
        "yubari_shard_end": dataset.yubari_shard_end,
        "enable_degradation": args.enable_degradation,
        "degradation_config_path": args.degradation_config_path,
        "samples": [],
    }
    for sample_index in range(args.num_samples):
        sample = next(iterator)
        sample_dir = os.path.join(args.save_dir, f"sample_{sample_index:03d}")
        summary["samples"].append(_save_sample(sample, sample_dir))
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
