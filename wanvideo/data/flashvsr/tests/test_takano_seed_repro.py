import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset


def _frame_hash(frames: List[Image.Image]) -> str:
    hasher = hashlib.sha256()
    for frame in frames:
        frame = frame.convert("RGB")
        hasher.update(np.asarray(frame, dtype=np.uint8).tobytes())
    return hasher.hexdigest()


def _save_frames(frames: List[Image.Image], path: str, fps: int = 8):
    save_video(frames, path, fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def _sample_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    video = sample["video"]
    lq_video = sample["lq_video"]
    return {
        "sample_id": sample.get("sample_id"),
        "source_type": sample.get("source_type"),
        "source_dataset": sample.get("source_dataset"),
        "media_path": sample.get("media_path"),
        "tar_member_path": sample.get("tar_member_path"),
        "caption_text": sample.get("caption_text"),
        "sample_seed": int(sample.get("sample_seed", -1)),
        "video_hash": _frame_hash(video),
        "lq_video_hash": _frame_hash(lq_video),
        "num_frames": len(video),
        "frame_size": list(video[0].size) if video else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal_url", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=160)
    parser.add_argument("--enable_degradation", action="store_true")
    parser.add_argument("--degradation_config_path", type=str, default=None)
    parser.add_argument("--global_seed", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = FlashVSRStreamingDataset(
        internal_url=args.internal_url,
        metadata_url=None,
        metadata_source="takano",
        image_internal_url=None,
        image_dataset_prob=0.0,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=args.enable_degradation,
        degradation_config_path=args.degradation_config_path,
        global_seed=args.global_seed,
        output_tensors=False,
    )

    iterator = iter(dataset)
    summary: Dict[str, Any] = {
        "global_seed": args.global_seed,
        "num_samples": args.num_samples,
        "internal_url": args.internal_url,
        "samples": [],
    }

    for sample_index in range(args.num_samples):
        sample = next(iterator)
        sample_dir = os.path.join(args.save_dir, f"sample_{sample_index:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        hr_frames = sample["video"]
        lq_frames = sample["lq_video"]
        hr_frames[0].save(os.path.join(sample_dir, "hr_first.png"))
        lq_frames[0].save(os.path.join(sample_dir, "lq_first.png"))
        _save_frames(hr_frames, os.path.join(sample_dir, "hr.mp4"))
        _save_frames(lq_frames, os.path.join(sample_dir, "lq.mp4"))

        sample_info = _sample_summary(sample)
        with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(sample_info, f, ensure_ascii=False, indent=2)
        summary["samples"].append(sample_info)

    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
