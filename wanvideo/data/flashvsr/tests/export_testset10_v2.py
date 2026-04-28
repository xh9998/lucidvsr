import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2


REPO_ROOT = Path(__file__).resolve().parents[4]
DEGRADATION_CONFIG_PATH = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_realesrgan_with_second.yaml"
)
TAKANO_METADATA_URL = (
    "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00000.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00001.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00000.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00001.parquet"
)
YUBARI_VIDEO_TAR_URL = "conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"


@dataclass
class Candidate:
    sample: Dict[str, Any]
    score: float


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    tensor = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in tensor:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _degradation_score(sample: Dict[str, Any]) -> float:
    video = sample["video"].detach().float()
    lq_video = sample["lq_video"].detach().float()
    mse = torch.mean((video - lq_video) ** 2).item()
    return float(mse)


def _build_dataset(source: str, *, seed: int, height: int, width: int, num_frames: int):
    common_kwargs = dict(
        metadata_source="takano",
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(160, num_frames),
        enable_degradation=True,
        degradation_config_path=DEGRADATION_CONFIG_PATH,
        global_seed=seed,
        output_tensors=True,
        max_parquet_records=256,
        max_yubari_records=256,
    )
    if source == "takano":
        return FlashVSRParquetTarDatasetV2(
            metadata_url=TAKANO_METADATA_URL,
            takano_dataset_prob=1.0,
            **common_kwargs,
        )
    if source == "yubari":
        return FlashVSRParquetTarDatasetV2(
            metadata_url=None,
            yubari_video_tar_url=YUBARI_VIDEO_TAR_URL,
            yubari_dataset_prob=1.0,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported source: {source}")


def _collect_candidates(
    source: str,
    *,
    count: int,
    pool_size: int,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
) -> List[Candidate]:
    dataset = _build_dataset(source, seed=seed, height=height, width=width, num_frames=num_frames)
    iterator = iter(dataset)
    candidates: List[Candidate] = []
    seen_ids = set()
    max_trials = max(pool_size * 8, 64)
    for _ in range(max_trials):
        if len(candidates) >= pool_size:
            break
        sample = next(iterator)
        sample_id = str(sample.get("sample_id"))
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        candidates.append(Candidate(sample=sample, score=_degradation_score(sample)))
    if len(candidates) < count:
        raise RuntimeError(f"Not enough {source} candidates: got {len(candidates)}, need {count}")
    candidates.sort(key=lambda item: item.score)
    return candidates


def _pick_spread(candidates: List[Candidate], count: int) -> List[Candidate]:
    if len(candidates) < count:
        raise ValueError(f"Not enough candidates to pick {count}")
    if count == 1:
        return [candidates[len(candidates) // 2]]
    indices = []
    last_index = len(candidates) - 1
    for idx in range(count):
        pick_index = round(idx * last_index / (count - 1))
        indices.append(pick_index)
    deduped: List[Candidate] = []
    seen = set()
    for idx in indices:
        sample_id = candidates[idx].sample["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        deduped.append(candidates[idx])
    cursor = 0
    while len(deduped) < count and cursor < len(candidates):
        sample_id = candidates[cursor].sample["sample_id"]
        if sample_id not in seen:
            seen.add(sample_id)
            deduped.append(candidates[cursor])
        cursor += 1
    return deduped[:count]


def _save_one(sample: Dict[str, Any], *, prefix: str, gt_dir: str, lq_dir: str, fps: int) -> Dict[str, Any]:
    video = sample["video"]
    lq_video = sample["lq_video"]
    gt_frames = _tensor_to_pil_frames(video)
    lq_frames = _tensor_to_pil_frames(lq_video)
    gt_name = f"{prefix}_gt.mp4"
    lq_name = f"{prefix}_lq.mp4"
    save_video(gt_frames, os.path.join(gt_dir, gt_name), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    save_video(lq_frames, os.path.join(lq_dir, lq_name), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return {
        "prefix": prefix,
        "gt_path": os.path.join(gt_dir, gt_name),
        "lq_path": os.path.join(lq_dir, lq_name),
        "sample_id": sample.get("sample_id"),
        "source_dataset": sample.get("source_dataset"),
        "sample_seed": int(sample["sample_seed"].item()) if torch.is_tensor(sample.get("sample_seed")) else sample.get("sample_seed"),
        "media_path": sample.get("media_path"),
        "tar_member_path": sample.get("tar_member_path"),
        "score_mse": _degradation_score(sample),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--num_per_source", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=20)
    args = parser.parse_args()

    gt_dir = os.path.join(args.output_dir, "gt")
    lq_dir = os.path.join(args.output_dir, "lq")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    summary: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "num_per_source": args.num_per_source,
        "pool_size": args.pool_size,
        "samples": [],
    }

    source_settings: List[Tuple[str, int]] = [("yubari", args.seed), ("takano", args.seed + 1000)]
    for source_name, seed in source_settings:
        candidates = _collect_candidates(
            source_name,
            count=args.num_per_source,
            pool_size=args.pool_size,
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        selected = _pick_spread(candidates, args.num_per_source)
        for index, candidate in enumerate(selected):
            prefix = f"{source_name}_{index:02d}"
            summary["samples"].append(
                _save_one(candidate.sample, prefix=prefix, gt_dir=gt_dir, lq_dir=lq_dir, fps=args.fps)
            )

    summary["samples"].sort(key=lambda item: item["prefix"])
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": args.output_dir, "num_samples": len(summary["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
