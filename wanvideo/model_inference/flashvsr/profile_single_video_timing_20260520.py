#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is unavailable in this environment.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.core import load_state_dict
from diffsynth.utils.data import VideoData, save_video
from wanvideo.model_inference.flashvsr.color_fix import apply_color_fix
from wanvideo.model_inference.flashvsr.infer_flashvsr_full_cloud_padded import (
    count_frames_and_fps,
    pad_video_to_length,
    smallest_8n_minus_3_geq,
    trim_video,
)
from wanvideo.model_inference.flashvsr.infer_flashvsr_stage1_v5_3_aligned import (
    add_common_args as add_stage1_args,
    build_flashvsr_stage1_pipe,
    load_lq_video_frames as load_stage1_lq_video_frames,
    resize_frames_to_match,
)
from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1 import (
    add_common_args as add_stage2_args,
    build_stage2_pipe,
    load_lq_video_frames as load_stage2_lq_video_frames,
)
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5
from wanvideo.model_training.flashvsr.train_flashvsr_stage2_v6_1_lora import (
    flashvsr_stage2_streaming_model_fn,
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(stats: Dict[str, float], name: str):
    cuda_sync()
    start = time.perf_counter()
    yield
    cuda_sync()
    stats[name] = stats.get(name, 0.0) + (time.perf_counter() - start)


def merge_existing_records(records: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    json_path = os.path.join(output_dir, "single_video_timing_detailed.json")
    merged: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                existing = json.load(f)
            for record in existing:
                method = record.get("method")
                if method:
                    merged[method] = record
        except Exception as exc:
            print(f"[warn] failed to read existing report for merge: {exc}")
    for record in records:
        method = record.get("method")
        if method:
            merged[method] = record
    return list(merged.values())


def write_reports(records: List[Dict[str, Any]], output_dir: str, frames: int):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "single_video_timing_detailed.json")
    csv_path = os.path.join(output_dir, "single_video_timing_detailed.csv")
    md_path = os.path.join(output_dir, "single_video_timing_detailed.md")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    keys = ["method", "frames", "status", "total_wall_s", "per_frame_wall_s"]
    extra_keys = sorted({k for r in records for k in r if k not in set(keys)})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra_keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return "" if v is None else str(v)

    with open(md_path, "w") as f:
        f.write("# Single 89f Video Timing Profile\n\n")
        f.write(f"- frames: {frames}\n")
        f.write("- note: per-frame table is wall-clock seconds divided by 89, not measured independent frame execution.\n\n")
        f.write("## Seconds Per Video\n\n")
        header = ["method", "total_wall_s"] + extra_keys
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for record in records:
            f.write("| " + " | ".join(fmt(record.get(k)) for k in header) + " |\n")
        f.write("\n## Seconds Per Frame\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
        for record in records:
            row = dict(record)
            for k, v in list(row.items()):
                if isinstance(v, (int, float)) and k not in {"frames", "status"}:
                    row[k] = float(v) / float(frames)
            f.write("| " + " | ".join(fmt(row.get(k)) for k in header) + " |\n")
    print(f"[report] {json_path}")
    print(f"[report] {csv_path}")
    print(f"[report] {md_path}")


def ns_from_args(args, **kwargs):
    data = vars(args).copy()
    data.update(kwargs)
    return SimpleNamespace(**data)


@torch.no_grad()
def profile_stage1(args, method: str, checkpoint_path: str) -> Dict[str, Any]:
    stats: Dict[str, float] = {}
    total_start = time.perf_counter()
    stage_args = ns_from_args(args, checkpoint_path=checkpoint_path)
    with timed(stats, "model_load_total_s"):
        pipe = build_flashvsr_stage1_pipe(stage_args)

    with timed(stats, "video_read_resize_s"):
        lq_video, effective_height, effective_width = load_stage1_lq_video_frames(
            args.input_video, args.height, args.width, args.input_bicubic_upscale
        )
        lq_video = lq_video[: args.num_frames]
    if not lq_video:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")

    with timed(stats, "scheduler_setup_s"):
        pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    inputs_shared = {
        "input_video": None,
        "lq_video": lq_video,
        "seed": args.seed,
        "rand_device": "cpu",
        "height": effective_height,
        "width": effective_width,
        "num_frames": len(lq_video),
        "cfg_scale": 1.0,
        "cfg_merge": False,
        "tiled": args.tiled,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "framewise_decoding": False,
        "vace_reference_image": None,
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "lq_proj_scale": pipe.lq_proj_scale,
    }
    inputs_posi: Dict[str, Any] = {}
    inputs_nega: Dict[str, Any] = {}
    with timed(stats, "pre_units_noise_prompt_lqproj_s"):
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    if "latents" not in inputs_shared:
        inputs_shared["latents"] = inputs_shared["noise"]
    with timed(stats, "onload_dit_s"):
        pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    dit_total = 0.0
    scheduler_total = 0.0
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        cuda_sync()
        t0 = time.perf_counter()
        noise_pred = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        cuda_sync()
        dit_total += time.perf_counter() - t0
        cuda_sync()
        t0 = time.perf_counter()
        inputs_shared["latents"] = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"])
        cuda_sync()
        scheduler_total += time.perf_counter() - t0
    stats["dit_sampling_s"] = dit_total
    stats["scheduler_step_s"] = scheduler_total
    with timed(stats, "post_units_s"):
        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    with timed(stats, "onload_vae_s"):
        pipe.load_models_to_device(["vae"])
    with timed(stats, "vae_decode_s"):
        video = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=args.tiled, tile_size=(30, 52), tile_stride=(15, 26))
    with timed(stats, "vae_quantize_to_pil_s"):
        video = pipe.vae_output_to_video(video)
    with timed(stats, "color_fix_s"):
        if not args.disable_color_fix:
            color_ref_video = resize_frames_to_match(lq_video, video)
            video = apply_color_fix(video, color_ref_video, method=args.color_fix_method)
    output_video = os.path.join(args.output_dir, f"{method}_single.mp4")
    with timed(stats, "video_save_s"):
        save_video(video, output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    with timed(stats, "offload_cleanup_s"):
        pipe.load_models_to_device([])
    stats["total_wall_s"] = time.perf_counter() - total_start
    return make_record(method, stats, args.num_frames, output_video)


@torch.no_grad()
def profile_stage2_like(args, method: str, checkpoint_path: str, steps: int) -> Dict[str, Any]:
    stats: Dict[str, float] = {}
    total_start = time.perf_counter()
    stage_args = ns_from_args(args, checkpoint_path=checkpoint_path, num_inference_steps=steps)
    with timed(stats, "model_load_total_s"):
        pipe = build_stage2_pipe(stage_args)
    with timed(stats, "video_read_resize_s"):
        lq_video, effective_height, effective_width = load_stage2_lq_video_frames(
            args.input_video, args.height, args.width, args.input_bicubic_upscale
        )
        lq_video = lq_video[: args.num_frames]
    if int(len(lq_video)) % 8 != 1:
        raise ValueError(f"Stage2 expects num_frames % 8 == 1, got {len(lq_video)}")
    with timed(stats, "scheduler_setup_s"):
        pipe.scheduler.set_timesteps(steps, denoising_strength=1.0, shift=5.0)
    with timed(stats, "preprocess_video_tensor_s"):
        lq_tensor = pipe.preprocess_video(lq_video).to(device=pipe.device, dtype=pipe.torch_dtype)
        lq_tensor = lq_tensor[:, :, : len(lq_video)]
    latent_frames = max(1, (int(len(lq_video)) - 1) // 4)
    with timed(stats, "noise_init_s"):
        latents = pipe.generate_noise(
            (1, pipe.vae.model.z_dim, latent_frames, int(effective_height) // pipe.vae.upsampling_factor, int(effective_width) // pipe.vae.upsampling_factor),
            seed=args.seed,
            rand_device="cpu",
        ).to(device=pipe.device, dtype=pipe.torch_dtype)
    with timed(stats, "prompt_context_s"):
        context = v5.FlashVSRUnit_FixedPrompt().process(pipe)["context"]
    with timed(stats, "onload_dit_lqproj_s"):
        pipe.load_models_to_device(("dit", "lq_proj_in"))
    models = {"dit": pipe.dit}
    lqproj_total = 0.0
    dit_total = 0.0
    scheduler_total = 0.0
    chunks = 0
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep_tensor = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.lq_proj_in.clear_cache()
        pre_cache_k = [None] * len(pipe.dit.blocks)
        pre_cache_v = [None] * len(pipe.dit.blocks)
        updated_chunks = []
        process_total_num = (int(len(lq_video)) - 1) // 8 - 2
        for cur_process_idx in range(process_total_num):
            chunks += 1
            cuda_sync()
            t0 = time.perf_counter()
            if cur_process_idx == 0:
                lq_latents = None
                for inner_idx in range(7):
                    start = max(0, inner_idx * 4 - 3)
                    end = (inner_idx + 1) * 4 - 3
                    cur = pipe.lq_proj_in.stream_forward(lq_tensor[:, :, start:end])
                    if cur is None:
                        continue
                    if lq_latents is None:
                        lq_latents = cur
                    else:
                        for layer_idx in range(len(lq_latents)):
                            lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                cur_latents = latents[:, :, :6]
            else:
                lq_latents = None
                for inner_idx in range(2):
                    start = cur_process_idx * 8 + 17 + inner_idx * 4
                    end = cur_process_idx * 8 + 21 + inner_idx * 4
                    cur = pipe.lq_proj_in.stream_forward(lq_tensor[:, :, start:end])
                    if cur is None:
                        continue
                    if lq_latents is None:
                        lq_latents = cur
                    else:
                        for layer_idx in range(len(lq_latents)):
                            lq_latents[layer_idx] = torch.cat([lq_latents[layer_idx], cur[layer_idx]], dim=1)
                cur_latents = latents[:, :, 4 + cur_process_idx * 2 : 6 + cur_process_idx * 2]
            cuda_sync()
            lqproj_total += time.perf_counter() - t0
            cuda_sync()
            t0 = time.perf_counter()
            noise_pred, pre_cache_k, pre_cache_v = flashvsr_stage2_streaming_model_fn(
                **models,
                latents=cur_latents,
                timestep=timestep_tensor,
                context=context,
                lq_latents=lq_latents,
                lq_proj_scale=pipe.lq_proj_scale,
                pre_cache_k=pre_cache_k,
                pre_cache_v=pre_cache_v,
                cur_process_idx=cur_process_idx,
                topk_ratio=args.stage2_topk_ratio,
                kv_ratio=args.stage2_kv_ratio,
            )
            cuda_sync()
            dit_total += time.perf_counter() - t0
            cuda_sync()
            t0 = time.perf_counter()
            updated_chunks.append(pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], cur_latents))
            cuda_sync()
            scheduler_total += time.perf_counter() - t0
        latents = torch.cat(updated_chunks, dim=2)
    stats["lq_projector_stream_s"] = lqproj_total
    stats["dit_sampling_s"] = dit_total
    stats["scheduler_step_s"] = scheduler_total
    stats["stream_chunks"] = float(chunks)
    with timed(stats, "onload_vae_s"):
        pipe.load_models_to_device(["vae"])
    with timed(stats, "vae_decode_s"):
        video = pipe.vae.decode(latents, device=pipe.device, tiled=args.tiled, tile_size=(30, 52), tile_stride=(15, 26))
    with timed(stats, "vae_quantize_to_pil_s"):
        video = pipe.vae_output_to_video(video)
    with timed(stats, "color_fix_s"):
        if not args.disable_color_fix:
            aligned = min(len(video), len(lq_video))
            video = apply_color_fix(video[:aligned], lq_video[:aligned], method=args.color_fix_method)
    output_video = os.path.join(args.output_dir, f"{method}_single.mp4")
    with timed(stats, "video_save_s"):
        save_video(video, output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    with timed(stats, "offload_cleanup_s"):
        pipe.load_models_to_device([])
    stats["total_wall_s"] = time.perf_counter() - total_start
    return make_record(method, stats, args.num_frames, output_video)


def profile_flashvsr_official(args) -> Dict[str, Any]:
    stats: Dict[str, float] = {}
    total_start = time.perf_counter()
    repo = Path(args.flashvsr_repo).resolve()
    infer_py = repo / "examples" / "WanVSR" / "infer_flashvsr_full_cloud.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"missing official infer script: {infer_py}")
    output_video = os.path.join(args.output_dir, "flashvsr_official_single.mp4")
    with tempfile.TemporaryDirectory(prefix="profile_flashvsr_official_") as tmpdir:
        padded_dir = os.path.join(tmpdir, "inputs")
        temp_output_dir = os.path.join(tmpdir, "out")
        os.makedirs(padded_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.input_video))[0]
        with timed(stats, "pad_read_resize_save_s"):
            original_frames, fps = count_frames_and_fps(args.input_video)
            target_frames = smallest_8n_minus_3_geq(original_frames)
            padded_input = os.path.join(padded_dir, f"{stem}.mp4")
            pad_video_to_length(args.input_video, padded_input, target_frames, fps)
        cmd = [
            os.environ.get("PYTHON_BIN", sys.executable),
            str(infer_py),
            "--input_path",
            padded_dir,
            "--output_path",
            temp_output_dir,
            "--model_dir",
            args.flashvsr_model_dir,
            "--seed",
            str(args.seed),
            "--scale",
            "4",
            "--sparse_ratio",
            str(args.stage2_topk_ratio),
            "--kv_ratio",
            str(args.stage2_kv_ratio),
            "--local_range",
            "11",
            "--quality",
            "6",
            "--tiled",
        ]
        with timed(stats, "official_subprocess_total_s"):
            subprocess.run(cmd, check=True, cwd=str(repo / "examples" / "WanVSR"))
        produced = os.path.join(temp_output_dir, f"FlashVSR_v1.1_Full_{stem}_seed{args.seed}.mp4")
        with timed(stats, "trim_save_s"):
            trim_video(produced, output_video, original_frames, fps)
    stats["total_wall_s"] = time.perf_counter() - total_start
    return make_record("flashvsr_official", stats, args.num_frames, output_video)


def profile_seedvr_subprocess(args, method: str, model_kind: str, model_dir: str) -> Dict[str, Any]:
    stats: Dict[str, float] = {}
    total_start = time.perf_counter()
    output_dir = os.path.join(args.output_dir, method)
    input_dir = os.path.join(args.output_dir, f"_input_{method}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    input_link = os.path.join(input_dir, os.path.basename(args.input_video))
    if os.path.lexists(input_link):
        os.unlink(input_link)
    os.symlink(args.input_video, input_link)
    log_file = os.path.join(args.output_dir, f"{method}.log")
    cmd = [
        "bash",
        os.path.join(Path(__file__).resolve().parent, "history", "run_seedvr_dir_20260421.sh"),
    ]
    env = os.environ.copy()
    env.update(
        {
            "SEEDVR_PYTHON": args.seedvr_python,
            "MODEL_KIND": model_kind,
            "MODEL_DIR": model_dir,
            "INPUT_DIR": input_dir,
            "OUTPUT_DIR": output_dir,
            "LOG_FILE": log_file,
            "CUDA_DEVICE": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            "RES_H": str(args.height),
            "RES_W": str(args.width),
            "OUT_FPS": str(args.fps),
            "SEED": str(args.seed),
            "MASTER_PORT": str(args.seedvr_master_port),
            "PYTHONPATH_PREFIX": "/mnt/task_runtime/lucidvsr/third_party_compat",
        }
    )
    with timed(stats, "seedvr_subprocess_total_s"):
        subprocess.run(cmd, check=True, env=env)
    stats["total_wall_s"] = time.perf_counter() - total_start
    outputs = [str(p) for p in Path(output_dir).glob("*.mp4")]
    return make_record(method, stats, args.num_frames, outputs[0] if outputs else "")


def make_record(method: str, stats: Dict[str, float], frames: int, output_video: Optional[str]) -> Dict[str, Any]:
    total = float(stats.get("total_wall_s", 0.0))
    record: Dict[str, Any] = {
        "method": method,
        "frames": frames,
        "status": 0,
        "total_wall_s": total,
        "per_frame_wall_s": total / float(frames),
        "output_video": output_video or "",
    }
    for key, value in stats.items():
        if key != "total_wall_s":
            record[key] = float(value)
    known = sum(v for k, v in stats.items() if k != "total_wall_s" and isinstance(v, (int, float)) and not k.endswith("chunks"))
    record["unattributed_overhead_s"] = max(0.0, total - known)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Detailed single-video timing profile for PPT benchmark methods.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default="flashvsr_official,stage1_535_step10000,stage1_usmgt_step3000,stage2_v641_step6000,stage3_v7d32_step2000")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--prompt_tensor_path", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--lq_proj_checkpoint", type=str, default=None)
    parser.add_argument("--lq_proj_scale", type=float, default=1.0)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--disable_lora", action="store_true", default=False)
    parser.add_argument("--disable_projection", action="store_true", default=False)
    parser.add_argument("--lq_proj_temporal_mode", default="nonstreaming_aligned", choices=("streaming", "nonstreaming", "nonstreaming_aligned"))
    parser.add_argument("--stage2_attention_mode", default="block_sparse_chunk_causal", choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"))
    parser.add_argument("--stage2_topk_ratio", type=float, default=2.0)
    parser.add_argument("--stage2_local_num", type=int, default=-1)
    parser.add_argument("--stage2_kv_ratio", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--disable_color_fix", action="store_true", default=False)
    parser.add_argument("--color_fix_method", default="adain", choices=("adain", "wavelet"))
    parser.add_argument("--input_bicubic_upscale", type=float, default=4.0)
    parser.add_argument("--stage1_535_ckpt", required=True)
    parser.add_argument("--stage1_usmgt_ckpt", required=True)
    parser.add_argument("--stage2_641_ckpt", required=True)
    parser.add_argument("--stage3_d32_ckpt", required=True)
    parser.add_argument("--flashvsr_repo", default="/mnt/task_runtime/FlashVSR")
    parser.add_argument("--flashvsr_model_dir", default="/mnt/models/FlashVSR-v1.1")
    parser.add_argument("--seedvr_python", default="/mnt/conda_envs/seedvr/bin/python")
    parser.add_argument("--seedvr3b_model_dir", default="/mnt/models/SeedVR-3B")
    parser.add_argument("--seedvr2_3b_model_dir", default="/mnt/models/SeedVR2-3B")
    parser.add_argument("--seedvr_master_port", type=int, default=29731)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        print(f"[profile] method={method}")
        if method == "flashvsr_official":
            records.append(profile_flashvsr_official(args))
        elif method == "seedvr3b":
            records.append(profile_seedvr_subprocess(args, method, "seedvr1", args.seedvr3b_model_dir))
        elif method == "seedvr2_3b":
            records.append(profile_seedvr_subprocess(args, method, "seedvr2", args.seedvr2_3b_model_dir))
        elif method == "stage1_535_step10000":
            records.append(profile_stage1(args, method, args.stage1_535_ckpt))
        elif method == "stage1_usmgt_step3000":
            records.append(profile_stage1(args, method, args.stage1_usmgt_ckpt))
        elif method == "stage2_v641_step6000":
            records.append(profile_stage2_like(args, method, args.stage2_641_ckpt, 50))
        elif method == "stage3_v7d32_step2000":
            records.append(profile_stage2_like(args, method, args.stage3_d32_ckpt, 1))
        else:
            raise ValueError(f"unsupported internal/official method: {method}")
        write_reports(merge_existing_records(records, args.output_dir), args.output_dir, args.num_frames)
    print("[done]")


if __name__ == "__main__":
    main()
