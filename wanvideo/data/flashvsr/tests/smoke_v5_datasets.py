import argparse
import json
import traceback
from pathlib import Path

from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v5 import FlashVSRTarStreamingDatasetV5
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53


def _sample_summary(sample):
    item = {}
    for key in (
        "sample_kind",
        "source_type",
        "sample_id",
        "segment_lengths",
        "image_group_size",
        "target_num_frames",
        "image_sample_id",
        "image_source_type",
    ):
        if key in sample:
            item[key] = sample.get(key)
    for key in ("video", "lq_video", "image_video", "image_lq_video"):
        if key in sample:
            item[f"{key}_shape"] = tuple(sample[key].shape)
    if "sample_seed" in sample:
        seed = sample["sample_seed"]
        item["sample_seed_shape"] = tuple(seed.shape) if hasattr(seed, "shape") else str(type(seed))
    return item


def _pull_v5_samples(dataset):
    rng = dataset._make_iteration_rng()
    print("  pulling video sample for v5", flush=True)
    video_sample = dataset._wrap_video_sample(next(dataset._video_iterator(rng=rng)))
    print("  pulling grouped image sample for v5", flush=True)
    image_sample = next(dataset._group_image_iterator(rng=rng))
    return [video_sample, image_sample]


def _pull_v53_samples(dataset):
    rng = dataset._make_iteration_rng()
    print("  pulling video sample for v53", flush=True)
    video_sample = next(dataset._video_iterator(rng=rng))
    print("  pulling image sample for v53", flush=True)
    image_sample = next(dataset._image_iterator(rng=rng))
    return [
        {
            "video": video_sample["video"],
            "lq_video": video_sample["lq_video"],
            "sample_id": video_sample.get("sample_id"),
            "source_type": video_sample.get("source_type", "video"),
        },
        {
            "image_video": image_sample["video"],
            "image_lq_video": image_sample["lq_video"],
            "image_sample_id": image_sample.get("sample_id"),
            "image_source_type": image_sample.get("source_type", "image"),
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_variant", type=int, default=2)
    parser.add_argument(
        "--yubari_video_tar_url",
        default="conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=160)
    parser.add_argument("--shuffle_buffer", type=int, default=64)
    parser.add_argument("--global_seed", type=int, default=20260421)
    parser.add_argument("--disable_degradation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        yubari_video_tar_url=args.yubari_video_tar_url,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        stride=args.stride,
        max_source_frames=args.max_source_frames,
        enable_degradation=not args.disable_degradation,
        global_seed=args.global_seed,
        shuffle_buffer=args.shuffle_buffer,
        output_tensors=True,
    )

    checks = [
        (
            "5_1",
            lambda: FlashVSRTarStreamingDatasetV5(
                picked17k_image_tar_url=args.manifest_path,
                picked17k_dataset_prob=0.5,
                **common,
            ),
        ),
        (
            "5_2",
            lambda: FlashVSRTarStreamingDatasetV5(
                picked17k_image_tar_url=args.manifest_path,
                picked17k_dataset_prob=0.5,
                **common,
            ),
        ),
        (
            "5_3",
            lambda: FlashVSRTarStreamingDatasetV53(
                image_tar_root_url=args.manifest_path,
                **common,
            ),
        ),
    ]

    exit_code = 0
    for tag, build_dataset in checks:
        log_path = output_dir / f"{tag}.json"
        try:
            print(f"{tag}: building dataset", flush=True)
            dataset = build_dataset()
            print(f"{tag}: dataset built", flush=True)
            if tag in ("5_1", "5_2"):
                samples = [_sample_summary(sample) for sample in _pull_v5_samples(dataset)]
            else:
                samples = [_sample_summary(sample) for sample in _pull_v53_samples(dataset)]
            payload = {"status": "ok", "tag": tag, "samples": samples}
            print(f"{tag}: ok -> {log_path}", flush=True)
        except Exception as error:
            exit_code = 1
            payload = {
                "status": "error",
                "tag": tag,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            print(f"{tag}: error -> {log_path}: {error}", flush=True)
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
