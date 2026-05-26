import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2
from wanvideo.data.flashvsr.tests.export_inference_testset6_aliyun_x4_lq import (
    AliyunVideoCompressionX4LQ,
    _degrade_x4,
    _save_video,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEGRADATION_CONFIG = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml"
)
DEFAULT_TAKANO_METADATA = ",".join(
    [
        "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00000.parquet",
        "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00001.parquet",
        "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00000.parquet",
        "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00001.parquet",
    ]
)
DEFAULT_YUBARI_ROOT = "conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"


def _build_dataset(
    *,
    source: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    takano_metadata: str,
    yubari_root: str,
    max_records: int,
) -> FlashVSRParquetTarDatasetV2:
    common: Dict[str, Any] = dict(
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(160, num_frames),
        enable_degradation=False,
        global_seed=seed,
        output_tensors=False,
        image_dataset_prob=0.0,
    )
    if source == "takano":
        return FlashVSRParquetTarDatasetV2(
            metadata_source="takano",
            metadata_url=takano_metadata,
            takano_dataset_prob=1.0,
            yubari_dataset_prob=0.0,
            max_parquet_records=max_records,
            **common,
        )
    if source == "yubari":
        return FlashVSRParquetTarDatasetV2(
            metadata_source="yubari",
            metadata_url=None,
            yubari_video_tar_url=yubari_root,
            takano_dataset_prob=0.0,
            yubari_dataset_prob=1.0,
            max_yubari_records=max_records,
            **common,
        )
    raise ValueError(f"Unsupported source: {source}")


def _collect_unique(dataset: FlashVSRParquetTarDatasetV2, *, count: int, source: str) -> List[Dict[str, Any]]:
    iterator = iter(dataset)
    samples: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    while len(samples) < count and attempts < count * 300:
        attempts += 1
        sample = next(iterator)
        sample_id = str(sample.get("sample_id") or f"{source}_{attempts}")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        sample["source_dataset"] = source
        samples.append(sample)
        print(f"[collect] source={source} {len(samples)}/{count} sample_id={sample_id}", flush=True)
    if len(samples) < count:
        raise RuntimeError(f"Only collected {len(samples)} samples for {source}, need {count}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast 20-video light Aliyun x4-LQ testset exporter.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--takano_metadata", default=DEFAULT_TAKANO_METADATA)
    parser.add_argument("--yubari_root", default=DEFAULT_YUBARI_ROOT)
    parser.add_argument("--degradation_config_path", default=DEFAULT_DEGRADATION_CONFIG)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--num_per_source", type=int, default=10)
    parser.add_argument("--max_records", type=int, default=2000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    degrader = AliyunVideoCompressionX4LQ(config_path=args.degradation_config_path)

    summary: Dict[str, Any] = {
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "lq_height": args.height // 4,
        "lq_width": args.width // 4,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "degradation_config_path": args.degradation_config_path,
        "lq_rule": "light Aliyun degradation, final bicubic restore disabled; LQ is 1/4 GT size.",
        "sources": {},
        "samples": [],
    }

    for source, seed_offset in (("takano", 101), ("yubari", 202)):
        dataset = _build_dataset(
            source=source,
            seed=args.seed + seed_offset,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            takano_metadata=args.takano_metadata,
            yubari_root=args.yubari_root,
            max_records=args.max_records,
        )
        samples = _collect_unique(dataset, count=args.num_per_source, source=source)
        gt_dir = output_root / source / "gt"
        lq_dir = output_root / source / "lq"
        gt_dir.mkdir(parents=True, exist_ok=True)
        lq_dir.mkdir(parents=True, exist_ok=True)
        summary["sources"][source] = {"count": len(samples)}
        for index, sample in enumerate(samples):
            prefix = f"{source}_{index:02d}"
            frames: List[Image.Image] = sample["video"]
            sample_seed = int(sample.get("sample_seed", args.seed + index))
            lq_frames = _degrade_x4(degrader, frames, seed=sample_seed)
            gt_path = gt_dir / f"{prefix}_gt.mp4"
            lq_path = lq_dir / f"{prefix}_lq.mp4"
            _save_video(frames, str(gt_path), fps=args.fps)
            _save_video(lq_frames, str(lq_path), fps=args.fps)
            item = {
                "prefix": prefix,
                "source_dataset": source,
                "sample_id": sample.get("sample_id"),
                "sample_seed": sample_seed,
                "gt_path": str(gt_path),
                "lq_path": str(lq_path),
                "gt_size": [args.width, args.height],
                "lq_size": [args.width // 4, args.height // 4],
            }
            summary["samples"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(summary['samples'])} samples to {output_root}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
