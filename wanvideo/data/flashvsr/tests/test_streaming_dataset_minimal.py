import argparse
import os
from typing import List

import torch
from PIL import Image

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    video = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in video:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _save_sample(save_dir: str, sample_index: int, video: torch.Tensor, lq_video: torch.Tensor, fps: int = 8):
    os.makedirs(save_dir, exist_ok=True)
    hr_frames = _tensor_to_pil_frames(video)
    lq_frames = _tensor_to_pil_frames(lq_video)

    hr_frames[0].save(os.path.join(save_dir, f"sample_{sample_index:03d}_hr_first.png"))
    lq_frames[0].save(os.path.join(save_dir, f"sample_{sample_index:03d}_lq_first.png"))
    save_video(hr_frames, os.path.join(save_dir, f"sample_{sample_index:03d}_hr_clip.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    save_video(lq_frames, os.path.join(save_dir, f"sample_{sample_index:03d}_lq_clip.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal_url", type=str, default=None)
    parser.add_argument("--metadata_url", type=str, default=None)
    parser.add_argument("--metadata_source", type=str, default="auto")
    parser.add_argument("--image_internal_url", type=str, default=None)
    parser.add_argument("--image_dataset_prob", type=float, default=0.0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=160)
    parser.add_argument("--enable_degradation", action="store_true")
    parser.add_argument("--degradation_config_path", type=str, default=None)
    parser.add_argument("--global_seed", type=int, default=None)
    parser.add_argument("--max_parquet_records", type=int, default=None)
    parser.add_argument("--min_overall_score", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    dataset = FlashVSRStreamingDataset(
        internal_url=args.internal_url,
        metadata_url=args.metadata_url,
        metadata_source=args.metadata_source,
        image_internal_url=args.image_internal_url,
        image_dataset_prob=args.image_dataset_prob,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=args.enable_degradation,
        degradation_config_path=args.degradation_config_path,
        global_seed=args.global_seed,
        max_parquet_records=args.max_parquet_records,
        min_overall_score=args.min_overall_score,
        output_tensors=True,
    )

    iterator = iter(dataset)
    for sample_index in range(args.num_samples):
        sample = next(iterator)
        print(
            f"sample_index={sample_index} "
            f"sample_id={sample.get('sample_id')} "
            f"sample_seed={sample.get('sample_seed')}"
        )
        if args.save_dir:
            _save_sample(args.save_dir, sample_index, sample["video"], sample["lq_video"])


if __name__ == "__main__":
    main()
