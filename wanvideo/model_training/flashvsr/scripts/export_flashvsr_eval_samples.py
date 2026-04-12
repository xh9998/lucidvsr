import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize, LoadVideo, ToAbsolutePath
from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.model_training.flashvsr.train_flashvsr_stage1 import (
    _tensor_video_to_pil_frames,
    collect_fixed_validation_samples,
    parse_flashvsr_args,
)


def build_dataset(args):
    if args.dataset_mode == "streaming":
        return FlashVSRStreamingDataset(
            internal_url=args.internal_url,
            metadata_url=args.metadata_url,
            metadata_source=args.metadata_source,
            max_parquet_records=args.max_parquet_records,
            min_overall_score=args.min_overall_score,
            require_qwen35_parse_success=args.require_qwen35_parse_success,
            image_internal_url=args.image_internal_url,
            image_dataset_prob=args.image_dataset_prob,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            stride=args.stride,
            max_source_frames=args.max_source_frames,
            enable_degradation=args.enable_degradation,
            degradation_config_path=args.degradation_config_path,
            degradation_seed=args.degradation_seed,
            hq_prefix_frames=args.hq_prefix_frames,
            control_dropout_prob=args.control_dropout_prob,
            shuffle_buffer=args.shuffle_buffer,
            global_seed=args.global_seed,
            output_tensors=True,
        )
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16)),
            "lq_video": ToAbsolutePath(args.dataset_base_path)
            >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16)),
        },
    )


def clone_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    cached: Dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            cached[key] = value.detach().cpu().clone()
        else:
            cached[key] = deepcopy(value)
    return cached


def main():
    parser = argparse.ArgumentParser(description="从 FlashVSR 训练集导出固定 hq/lq 测试视频。")
    parser.add_argument("--config", type=str, required=True, help="训练 yaml 配置路径。")
    parser.add_argument("--output_dir", type=str, required=True, help="导出目录。")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    train_args = parse_flashvsr_args(["--config", args.config])
    dataset = build_dataset(train_args)
    samples = collect_fixed_validation_samples(dataset, args.num_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        sample = clone_sample(sample)
        sample_dir = output_dir / f"sample_{idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        hr_tensor = sample["video"]
        lq_tensor = sample["lq_video"]
        hr_frames = _tensor_video_to_pil_frames(hr_tensor)
        lq_frames = _tensor_video_to_pil_frames(lq_tensor)
        save_video(hr_frames, str(sample_dir / "hr.mp4"), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        save_video(lq_frames, str(sample_dir / "lq.mp4"), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        torch.save(hr_tensor, sample_dir / "hr.pt")
        torch.save(lq_tensor, sample_dir / "lq.pt")

        meta = {
            "index": idx,
            "sample_seed": int(sample.get("sample_seed", torch.tensor(-1)).item() if torch.is_tensor(sample.get("sample_seed")) else sample.get("sample_seed", -1)),
            "video_shape": list(hr_tensor.shape),
            "lq_shape": list(lq_tensor.shape),
            "config": os.path.abspath(args.config),
        }
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"saved_samples={len(samples)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
