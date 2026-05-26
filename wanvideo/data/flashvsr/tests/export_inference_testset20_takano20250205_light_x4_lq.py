import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

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
DEFAULT_TAKANO_MANIFEST = "/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_20250205_test_4k_tar_manifest.txt"
DEFAULT_TAKANO_MANIFEST_S3 = "s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt"
DEFAULT_OUTPUT_ROOT = "/mnt/task_wrapper/user_output/artifacts/data/inference/testset20_89f_takano20250205_light_x4_lq_20260518"


def _load_excluded_ids(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    summary_path = Path(path)
    if not summary_path.is_file():
        return set()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    excluded = set()
    for item in samples:
        sample_id = item.get("sample_id")
        if sample_id:
            excluded.add(str(sample_id))
    return excluded


def _load_manifest_entries(path: str, *, seed: int, limit: int = 256) -> List[str]:
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


def _build_dataset(*, manifest: str, seed: int, height: int, width: int, num_frames: int) -> FlashVSRStreamingDataset:
    # FlashVSRStreamingDataset keeps local manifest paths in video_manifest_urls,
    # but __iter__ currently only enters video mode when video_urls is non-empty.
    # Expand here instead of patching the shared training dataset.
    internal_url = ",".join(_load_manifest_entries(manifest, seed=seed))
    return FlashVSRStreamingDataset(
        internal_url=internal_url,
        image_internal_url=None,
        image_dataset_prob=0.0,
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(160, num_frames),
        enable_degradation=False,
        degradation_seed=None,
        hq_prefix_frames=0,
        control_dropout_prob=0.0,
        shuffle_buffer=200,
        global_seed=seed,
        output_tensors=False,
        metadata_url=None,
        metadata_source="takano20250205_test_4k",
    )


def _collect_unique(
    dataset: FlashVSRStreamingDataset,
    *,
    count: int,
    excluded_ids: Set[str],
) -> List[Dict[str, Any]]:
    iterator = iter(dataset)
    samples: List[Dict[str, Any]] = []
    seen = set(excluded_ids)
    attempts = 0
    while len(samples) < count and attempts < count * 500:
        attempts += 1
        sample = next(iterator)
        sample_id = str(sample.get("sample_id") or f"takano20250205_attempt{attempts}")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        sample["source_dataset"] = "takano20250205_test_4k"
        samples.append(sample)
        print(f"[collect] {len(samples)}/{count} sample_id={sample_id}", flush=True)
    if len(samples) < count:
        raise RuntimeError(f"Only collected {len(samples)} unique samples, need {count}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 20-video Takano-20250205 light Aliyun x4-LQ 89f testset.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--takano_manifest", default=DEFAULT_TAKANO_MANIFEST)
    parser.add_argument("--takano_manifest_s3", default=DEFAULT_TAKANO_MANIFEST_S3)
    parser.add_argument("--degradation_config_path", default=DEFAULT_DEGRADATION_CONFIG)
    parser.add_argument("--exclude_summary_json", default="")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.takano_manifest)
    if not manifest_path.is_file():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"conductor s3 cp {args.takano_manifest_s3} {args.takano_manifest}"
        print(f"[manifest] missing local manifest, running: {cmd}", flush=True)
        status = os.system(cmd)
        if status != 0:
            raise RuntimeError(f"Failed to fetch manifest from {args.takano_manifest_s3}")

    output_root = Path(args.output_root)
    gt_dir = output_root / "gt"
    lq_dir = output_root / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    excluded_ids = _load_excluded_ids(args.exclude_summary_json)
    dataset = _build_dataset(
        manifest=str(manifest_path),
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    samples = _collect_unique(dataset, count=args.count, excluded_ids=excluded_ids)
    degrader = AliyunVideoCompressionX4LQ(config_path=args.degradation_config_path)

    summary: Dict[str, Any] = {
        "name": "testset20_89f_takano20250205_light_x4_lq_20260518",
        "output_root": str(output_root),
        "source_s3": "s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/",
        "takano_manifest": str(manifest_path),
        "takano_manifest_s3": args.takano_manifest_s3,
        "height": args.height,
        "width": args.width,
        "lq_height": args.height // 4,
        "lq_width": args.width // 4,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "degradation_config_path": args.degradation_config_path,
        "lq_rule": "Light Aliyun x4 degradation; final bicubic restore disabled; LQ is 1/4 GT size.",
        "excluded_count": len(excluded_ids),
        "samples": [],
    }

    for index, sample in enumerate(samples):
        prefix = f"takano20250205_{index:02d}"
        frames = sample["video"]
        sample_seed = int(sample.get("sample_seed", args.seed + index))
        lq_frames = _degrade_x4(degrader, frames, seed=sample_seed)
        gt_path = gt_dir / f"{prefix}_gt.mp4"
        lq_path = lq_dir / f"{prefix}_lq.mp4"
        _save_video(frames, str(gt_path), fps=args.fps)
        _save_video(lq_frames, str(lq_path), fps=args.fps)
        item = {
            "prefix": prefix,
            "source_dataset": sample.get("source_dataset"),
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
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(summary['samples'])} samples to {output_root}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
