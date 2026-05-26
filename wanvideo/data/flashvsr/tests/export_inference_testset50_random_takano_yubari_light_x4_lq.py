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
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
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
DEFAULT_TAKANO_MANIFEST = "/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_20250205_test_4k_tar_manifest.txt"
DEFAULT_TAKANO_MANIFEST_S3 = "s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt"
DEFAULT_YUBARI_ROOT = "conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"
DEFAULT_OUTPUT_ROOT = "/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_light_x4_lq_20260518"


def _load_manifest_entries(path: str, *, seed: int, limit: int) -> List[str]:
    entries: List[str] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    if not entries:
        raise ValueError(f"Manifest has no usable entries: {path}")
    rng = random.Random(seed)
    rng.shuffle(entries)
    return entries[:limit]


def _build_dataset(
    *,
    source: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    takano_metadata: str,
    takano_manifest: str,
    yubari_root: str,
    max_records: int,
) -> Any:
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
        internal_url = ",".join(_load_manifest_entries(takano_manifest, seed=seed, limit=max_records))
        return FlashVSRStreamingDataset(
            internal_url=internal_url,
            image_internal_url=None,
            metadata_url=None,
            metadata_source="takano20250205_test_4k",
            shuffle_buffer=200,
            hq_prefix_frames=0,
            control_dropout_prob=0.0,
            degradation_seed=None,
            **common,
        )
    if source == "yubari":
        return FlashVSRStreamingDataset(
            internal_url=yubari_root,
            image_internal_url=None,
            metadata_url=None,
            metadata_source="auto",
            shuffle_buffer=200,
            hq_prefix_frames=0,
            control_dropout_prob=0.0,
            degradation_seed=None,
            **common,
        )
    raise ValueError(f"Unsupported source: {source}")


def _collect_unique(dataset: Any, *, count: int, source: str) -> List[Dict[str, Any]]:
    iterator = iter(dataset)
    samples: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    while len(samples) < count and attempts < count * 500:
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
    parser = argparse.ArgumentParser(description="Export random 25 Takano + 25 Yubari light Aliyun x4-LQ 89f testset.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--takano_metadata", default=DEFAULT_TAKANO_METADATA)
    parser.add_argument("--takano_manifest", default=DEFAULT_TAKANO_MANIFEST)
    parser.add_argument("--takano_manifest_s3", default=DEFAULT_TAKANO_MANIFEST_S3)
    parser.add_argument("--yubari_root", default=DEFAULT_YUBARI_ROOT)
    parser.add_argument("--degradation_config_path", default=DEFAULT_DEGRADATION_CONFIG)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--num_per_source", type=int, default=25)
    parser.add_argument("--max_records", type=int, default=5000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    gt_dir = output_root / "gt"
    lq_dir = output_root / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)
    degrader = AliyunVideoCompressionX4LQ(config_path=args.degradation_config_path)
    takano_manifest = Path(args.takano_manifest)
    if not takano_manifest.is_file():
        takano_manifest.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"conductor s3 cp {args.takano_manifest_s3} {args.takano_manifest}"
        print(f"[manifest] missing local Takano manifest, running: {cmd}", flush=True)
        if os.system(cmd) != 0:
            raise RuntimeError(f"Failed to fetch Takano manifest from {args.takano_manifest_s3}")

    summary: Dict[str, Any] = {
        "name": output_root.name,
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "lq_height": args.height // 4,
        "lq_width": args.width // 4,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "num_per_source": args.num_per_source,
        "selection_rule": "random order from dataset iterator; no visual filtering or scoring",
        "degradation_config_path": args.degradation_config_path,
        "lq_rule": "light Aliyun degradation; LQ is 1/4 GT size.",
        "sources": {
            "takano": {"metadata": args.takano_metadata},
            "takano_manifest": str(takano_manifest),
            "takano_manifest_s3": args.takano_manifest_s3,
            "yubari": {"root": args.yubari_root},
        },
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
            takano_manifest=str(takano_manifest),
            yubari_root=args.yubari_root,
            max_records=args.max_records,
        )
        samples = _collect_unique(dataset, count=args.num_per_source, source=source)
        for index, sample in enumerate(samples):
            prefix = f"{source}_{index:02d}"
            frames: List[Image.Image] = sample["video"]
            sample_seed = int(sample.get("sample_seed", args.seed + seed_offset + index))
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
