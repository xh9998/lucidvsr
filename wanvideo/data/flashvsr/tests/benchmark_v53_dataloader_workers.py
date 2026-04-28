#!/usr/bin/env python3
import argparse
import inspect
import json
import time
from pathlib import Path

import torch
import yaml

from diffsynth.diffusion.runner import _PreBatchedIterableDataset
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--in_order", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiprocessing_context", default=None, choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--touch_cuda_before_iter", action="store_true")
    parser.add_argument("--disable_degradation", action="store_true")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def _flatten_config(config_data):
    merged = {}
    for value in config_data.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def main():
    bench_args = parse_args()
    with open(bench_args.config, "r", encoding="utf-8") as file:
        cfg = _flatten_config(yaml.safe_load(file) or {})
    if cfg.get("dataset_mode") != "tar_v53":
        raise ValueError(f"This benchmark currently targets tar_v53, got {cfg.get('dataset_mode')!r}")

    batch_size = int(bench_args.batch_size or cfg.get("batch_size", 1))
    dataset_kwargs = dict(
        yubari_video_tar_url=cfg.get("yubari_video_tar_url"),
        takano_video_tar_url=cfg.get("takano_video_tar_url"),
        image_tar_root_url=cfg.get("image_tar_url") or cfg.get("picked17k_image_tar_url"),
        yubari_video_prob=cfg.get("yubari_video_prob"),
        takano_video_prob=cfg.get("takano_video_prob"),
        height=cfg.get("height"),
        width=cfg.get("width"),
        num_frames=cfg.get("num_frames"),
        stride=cfg.get("stride", 1),
        max_source_frames=cfg.get("max_source_frames", 160),
        enable_degradation=(False if bench_args.disable_degradation else cfg.get("enable_degradation", True)),
        degradation_seed=cfg.get("degradation_seed"),
        hq_prefix_frames=cfg.get("hq_prefix_frames", 0),
        control_dropout_prob=cfg.get("control_dropout_prob", 0.0),
        shuffle_buffer=cfg.get("shuffle_buffer", 100),
        global_seed=cfg.get("global_seed"),
        output_tensors=True,
    )
    if "image_branch_num_frames" in inspect.signature(FlashVSRTarStreamingDatasetV53).parameters:
        dataset_kwargs["image_branch_num_frames"] = cfg.get("image_branch_num_frames")
    print("[worker-bench] constructing dataset", flush=True)
    dataset = FlashVSRTarStreamingDatasetV53(**dataset_kwargs)

    collate_fn = getattr(dataset, "custom_collate_fn", lambda x: x[0])
    dataloader_dataset = dataset
    dataloader_batch_size = batch_size
    dataloader_collate_fn = collate_fn
    if batch_size > 1:
        dataloader_dataset = _PreBatchedIterableDataset(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_batch_size = 1
        dataloader_collate_fn = lambda x: x[0]

    dataloader_kwargs = {
        "batch_size": dataloader_batch_size,
        "collate_fn": dataloader_collate_fn,
        "num_workers": bench_args.workers,
    }
    if bench_args.workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, bench_args.prefetch_factor)
        dataloader_kwargs["persistent_workers"] = bench_args.persistent_workers
        if bench_args.multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = bench_args.multiprocessing_context
        if "in_order" in inspect.signature(torch.utils.data.DataLoader).parameters:
            dataloader_kwargs["in_order"] = bench_args.in_order

    print("[worker-bench] dataset constructed", flush=True)
    print(
        "[worker-bench] "
        f"workers={bench_args.workers} batch_size={batch_size} "
        f"prefetch={dataloader_kwargs.get('prefetch_factor')} "
        f"persistent={dataloader_kwargs.get('persistent_workers', False)} "
        f"mp_context={dataloader_kwargs.get('multiprocessing_context')} "
        f"in_order={dataloader_kwargs.get('in_order')}",
        flush=True,
    )
    dataloader = torch.utils.data.DataLoader(dataloader_dataset, **dataloader_kwargs)
    print("[worker-bench] dataloader constructed", flush=True)

    if bench_args.touch_cuda_before_iter:
        torch.empty(1, device="cuda")
        print("[worker-bench] touched cuda before iterator", flush=True)

    print("[worker-bench] creating iterator", flush=True)
    iterator = iter(dataloader)
    print("[worker-bench] iterator created", flush=True)
    timings = []
    first_batch_seconds = None
    total_batches = bench_args.warmup_batches + bench_args.num_batches
    for index in range(total_batches):
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        if first_batch_seconds is None:
            first_batch_seconds = elapsed
        if index >= bench_args.warmup_batches:
            timings.append(elapsed)
        if index < 3 or index == total_batches - 1:
            keys = sorted(batch.keys()) if isinstance(batch, dict) else type(batch).__name__
            print(f"[worker-bench] batch={index} seconds={elapsed:.4f} keys={keys}", flush=True)

    result = {
        "workers": bench_args.workers,
        "batch_size": batch_size,
        "prefetch_factor": dataloader_kwargs.get("prefetch_factor"),
        "persistent_workers": dataloader_kwargs.get("persistent_workers", False),
        "multiprocessing_context": dataloader_kwargs.get("multiprocessing_context"),
        "in_order": dataloader_kwargs.get("in_order"),
        "warmup_batches": bench_args.warmup_batches,
        "measured_batches": bench_args.num_batches,
        "first_batch_seconds": first_batch_seconds,
        "mean_seconds": sum(timings) / len(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "batches_per_second": len(timings) / sum(timings),
    }
    print("[worker-bench-result] " + json.dumps(result, sort_keys=True), flush=True)
    if bench_args.output_json:
        output_path = Path(bench_args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
