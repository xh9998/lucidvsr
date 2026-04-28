import argparse
import json
import os
from pathlib import Path

import torch

from wanvideo.data.flashvsr.datasets.joint_batching_v1 import collate_image_video_joint_v1
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v4_lora import FlashVSRStage1TrainingModule


REPO_ROOT = Path(__file__).resolve().parents[4]
DEGRADATION_CONFIG_PATH = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_realesrgan_with_second.yaml"
)


def _build_dataset(
    *,
    mode: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    media_cache_dir: str | None,
    parquet_cache_dir: str | None,
):
    if parquet_cache_dir:
        os.environ["FLASHVSR_PARQUET_CACHE_DIR"] = parquet_cache_dir
    common_kwargs = dict(
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(num_frames, 17),
        enable_degradation=True,
        degradation_config_path=DEGRADATION_CONFIG_PATH,
        global_seed=seed,
        output_tensors=True,
        max_parquet_records=8,
        max_yubari_records=8,
        media_cache_dir=media_cache_dir,
    )
    if mode == "image":
        return FlashVSRParquetTarDatasetV2(
            metadata_url=None,
            metadata_source="takano",
            image_metadata_url="s3://takano-assets/20231106/high_resolution/metadata_split_parquet/train/",
            image_dataset_prob=1.0,
            image_as_single_frame=True,
            **common_kwargs,
        )
    if mode == "takano":
        return FlashVSRParquetTarDatasetV2(
            metadata_url=(
                "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/,"
                "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/"
            ),
            metadata_source="takano",
            takano_dataset_prob=1.0,
            **common_kwargs,
        )
    if mode == "yubari":
        return FlashVSRParquetTarDatasetV2(
            metadata_url=None,
            metadata_source="takano",
            yubari_video_tar_url="conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/",
            yubari_dataset_prob=1.0,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _next_sample(dataset, expected_source: str):
    iterator = iter(dataset)
    for _ in range(16):
        sample = next(iterator)
        if sample.get("source_dataset") == expected_source:
            return sample
    raise RuntimeError(f"Failed to get sample from source_dataset={expected_source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--media_cache_dir", type=str, default=None)
    parser.add_argument("--parquet_cache_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--video_source", type=str, default="yubari", choices=("takano", "yubari"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    image_dataset = _build_dataset(
        mode="image",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        media_cache_dir=args.media_cache_dir,
        parquet_cache_dir=args.parquet_cache_dir,
    )
    video_dataset = _build_dataset(
        mode=args.video_source,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed + 1,
        media_cache_dir=args.media_cache_dir,
        parquet_cache_dir=args.parquet_cache_dir,
    )
    image_sample = _next_sample(image_dataset, "image")
    video_sample = _next_sample(video_dataset, args.video_source)
    batch = collate_image_video_joint_v1([image_sample, video_sample])

    module = FlashVSRStage1TrainingModule(
        model_paths=args.model_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models="lq_proj_in",
        lora_base_model="dit",
        lora_target_modules="q,k,v,o",
        lora_rank=8,
        lq_proj_layer_num=1,
        lq_proj_scale=5.0,
        zero_init_lq_proj_in=True,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        image_video_joint_packed=True,
        device=args.device,
    )
    module.train()
    with torch.no_grad():
        loss = module.forward(batch)
    payload = {
        "loss": float(loss.detach().float().item()),
        "sequence_lengths": batch["sequence_lengths"].tolist(),
        "segment_lengths": batch["segment_lengths"],
        "sources": batch.get("source_dataset"),
        "sample_ids": batch.get("sample_id"),
    }
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
