#!/usr/bin/env python3
"""Gate 1 probe: compare Stage1 teacher path with Stage3 G_real wrapper.

This tool is intentionally standalone. It does not modify or import through any
training entrypoint side effects beyond constructing the existing modules.

The comparable quantity is the DMD teacher one-step x0 prediction:
  - build clean latents from the fixed GT batch using the Stage3 wrapper units;
  - add the same noise at the same timestep;
  - run a Stage1 fixed-prompt teacher path and the Stage3 G_real wrapper path;
  - decode both x0 predictions and write tensor/video statistics to JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import ModelConfig, load_state_dict  # noqa: E402
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5  # noqa: E402
from wanvideo.model_training.flashvsr import train_flashvsr_stage3_v7_d4_4_lora as d44  # noqa: E402


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _cfg(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    for section_name in ("model", "lora", "runtime", "train", "data", "validation", "output"):
        section = _section(config, section_name)
        for key in keys:
            if key in section:
                return section[key]
    return default


def _parse_model_configs(model_paths: Any, base_model_dir: str | None) -> list[ModelConfig]:
    paths: list[str]
    if model_paths:
        if isinstance(model_paths, str):
            text = model_paths.strip()
            if text.startswith("["):
                paths = [str(item) for item in json.loads(text)]
            else:
                paths = [part.strip() for part in text.split(",") if part.strip()]
        elif isinstance(model_paths, (list, tuple)):
            paths = [str(item) for item in model_paths]
        else:
            raise TypeError(f"Unsupported model_paths type: {type(model_paths).__name__}")
    elif base_model_dir:
        paths = [
            os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(base_model_dir, "Wan2.1_VAE.pth"),
        ]
    else:
        raise ValueError("Need either model_paths in config or --base_model_dir.")
    return [ModelConfig(path=path) for path in paths]


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping[str(name).lower()]


def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    value = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "abs_mean": float(value.abs().mean()),
    }


def _tensor_diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    diff = lhs_f - rhs_f
    flat_l = lhs_f.flatten()
    flat_r = rhs_f.flatten()
    denom = flat_l.norm() * flat_r.norm()
    cosine = float(torch.dot(flat_l, flat_r).div(denom.clamp_min(1e-12)).detach().cpu())
    return {
        "shape_lhs": list(lhs.shape),
        "shape_rhs": list(rhs.shape),
        "max_abs": float(diff.abs().max().detach().cpu()),
        "mean_abs": float(diff.abs().mean().detach().cpu()),
        "mse": float(diff.pow(2).mean().detach().cpu()),
        "cosine": cosine,
    }


def _frames_rgb_stats(frames: list[Any]) -> dict[str, Any]:
    if not frames:
        return {"num_frames": 0}
    import numpy as np

    arrays = [np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0 for frame in frames]
    stacked = np.stack(arrays, axis=0)
    return {
        "num_frames": int(stacked.shape[0]),
        "shape": list(stacked.shape),
        "rgb_mean": [float(x) for x in stacked.mean(axis=(0, 1, 2))],
        "rgb_std": [float(x) for x in stacked.std(axis=(0, 1, 2))],
        "brightness_mean": float(stacked.mean()),
        "brightness_std": float(stacked.mean(axis=-1).std()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
    }


def _save_input_video(tensor_tchw: torch.Tensor, path: Path, fps: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = v5._tensor_video_to_pil_frames(tensor_tchw.detach().cpu().float().clamp(0, 1))
    v5.save_video(frames, str(path), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return {"path": str(path), "rgb": _frames_rgb_stats(frames), "tensor": _tensor_stats(tensor_tchw)}


def _decode_latents(pipe, latents: torch.Tensor, path: Path, fps: int, tiled: bool) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        pipe.load_models_to_device(["vae"])
        video = pipe.vae.decode(
            latents.to(device=pipe.device, dtype=pipe.torch_dtype),
            device=pipe.device,
            tiled=bool(tiled),
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        frames = pipe.vae_output_to_video(video)
    v5.save_video(frames, str(path), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    return {
        "path": str(path),
        "latent": _tensor_stats(latents),
        "decoded_tensor": _tensor_stats(video),
        "rgb": _frames_rgb_stats(frames),
    }


def _load_fixed_batch(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.fixed_batch:
        sample_path = Path(args.fixed_batch)
    else:
        sample_path = Path(args.fixed_lqgt_root) / f"sample_{int(args.sample_index):02d}.pt"
    payload = torch.load(sample_path, map_location="cpu")
    if "video" not in payload or "lq_video" not in payload:
        raise KeyError(f"Fixed sample must contain video/lq_video: {sample_path}")
    video = payload["video"].detach().cpu().float()
    lq_video = payload["lq_video"].detach().cpu().float()
    if video.ndim != 4 or lq_video.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got video={tuple(video.shape)} lq={tuple(lq_video.shape)}")
    data = {
        "video": video.unsqueeze(0),
        "lq_video": lq_video.unsqueeze(0),
        "sample_seed": payload.get("sample_seed"),
        "sample_id": payload.get("sample_id", sample_path.stem),
    }
    meta = {
        "sample_path": str(sample_path),
        "sample_seed": payload.get("sample_seed"),
        "sample_id": payload.get("sample_id", sample_path.stem),
        "video": _tensor_stats(video),
        "lq_video": _tensor_stats(lq_video),
    }
    return data, meta


def _infer_lq_proj_layer_num(state: dict[str, torch.Tensor]) -> int | None:
    indices: list[int] = []
    prefix = "linear_layers."
    for key in state:
        if key.startswith(prefix):
            item = key[len(prefix):].split(".", 1)[0]
            if item.isdigit():
                indices.append(int(item))
    return (max(indices) + 1) if indices else None


def _build_stage1_teacher_pipe(args: argparse.Namespace, config: dict[str, Any], checkpoint: str):
    ckpt = load_state_dict(checkpoint, device="cpu")
    lq_proj_state, lora_state, other_state = v5.flashvsr_stage1_split_exported_state(ckpt)
    layer_num = args.lq_proj_layer_num or _cfg(config, "lq_proj_layer_num", default=None) or _infer_lq_proj_layer_num(lq_proj_state)
    if layer_num is None:
        raise ValueError("Cannot infer lq_proj_layer_num for Stage1 teacher.")
    model_configs = _parse_model_configs(_cfg(config, "model_paths", default=None), args.base_model_dir)
    pipe = v5.WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=_torch_dtype(args.torch_dtype),
        device=args.device,
        model_configs=model_configs,
        prompt_tensor_path=args.prompt_tensor_path or _cfg(config, "prompt_tensor_path", default=None),
        lq_proj_layer_num=int(layer_num),
        lq_proj_temporal_mode=args.stage1_lq_proj_temporal_mode or _cfg(config, "lq_proj_temporal_mode", default="nonstreaming_aligned"),
    )
    pipe.lq_proj_scale = float(args.lq_proj_scale if args.lq_proj_scale is not None else _cfg(config, "lq_proj_scale", default=1.0))
    if lq_proj_state:
        missing, unexpected = pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    else:
        missing, unexpected = [], []
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)
    pipe.eval()
    return pipe, {
        "checkpoint": checkpoint,
        "lq_proj_keys": len(lq_proj_state),
        "lora_keys": len(lora_state),
        "other_keys": len(other_state),
        "lq_proj_layer_num": int(layer_num),
        "lq_proj_temporal_mode": getattr(pipe.lq_proj_in, "temporal_mode", None),
        "lq_proj_scale": float(pipe.lq_proj_scale),
        "missing_lq_proj_keys": list(missing),
        "unexpected_lq_proj_keys": list(unexpected),
    }


def _build_stage3_greal_module(args: argparse.Namespace, config: dict[str, Any], checkpoint: str):
    module = d44.FlashVSRStage3BTrainingModule(
        model_paths=_cfg(config, "model_paths", default=None),
        model_id_with_origin_paths=_cfg(config, "model_id_with_origin_paths", default=None),
        prompt_tensor_path=args.prompt_tensor_path or _cfg(config, "prompt_tensor_path", default=None),
        trainable_models=_cfg(config, "trainable_models", default="lq_proj_in"),
        lora_base_model=_cfg(config, "lora_base_model", default="dit"),
        lora_target_modules=_cfg(config, "lora_target_modules", default="q,k,v,o"),
        lora_rank=int(_cfg(config, "lora_rank", default=384)),
        lora_checkpoint=None,
        lq_proj_checkpoint=None,
        resume_stage1_checkpoint=checkpoint,
        lq_proj_layer_num=int(args.lq_proj_layer_num or _cfg(config, "lq_proj_layer_num", default=1)),
        lq_proj_scale=float(args.lq_proj_scale if args.lq_proj_scale is not None else _cfg(config, "lq_proj_scale", default=1.0)),
        zero_init_lq_proj_in=False,
        freeze_lq_proj_in=True,
        use_gradient_checkpointing=bool(_cfg(config, "use_gradient_checkpointing", default=True)),
        use_gradient_checkpointing_offload=bool(_cfg(config, "use_gradient_checkpointing_offload", default=False)),
        stage2_attention_mode=args.stage3_real_attention_mode or _cfg(config, "stage3_real_attention_mode", default="dense_full"),
        stage2_topk_ratio=float(_cfg(config, "stage2_topk_ratio", default=2.0)),
        stage2_local_num=int(_cfg(config, "stage2_local_num", default=-1)),
        lq_proj_temporal_mode=args.stage3_real_lq_proj_temporal_mode or _cfg(config, "stage3_real_lq_proj_temporal_mode", default="nonstreaming_aligned"),
        stage3_recon_num_latents=int(_cfg(config, "stage3_recon_num_latents", default=2)),
        stage3_flow_weight=1.0,
        stage3_mse_weight=0.0,
        stage3_lpips_weight=0.0,
        stage3_lpips_net=str(_cfg(config, "stage3_lpips_net", default="vgg")),
        stage3_first_frame_pixel_weight=float(_cfg(config, "stage3_first_frame_pixel_weight", default=4.0)),
        stage3_first_frame_lpips_weight=float(_cfg(config, "stage3_first_frame_lpips_weight", default=4.0)),
        stage3_decoder_cpu_offload=bool(_cfg(config, "stage3_decoder_cpu_offload", default=True)),
        stage3_compute_z_pred=False,
        stage3_fake_fm_weight=0.0,
        stage3_fake_update_ratio=1,
        stage3_fake_checkpoint=None,
        fp8_models=_cfg(config, "fp8_models", default=None),
        offload_models=_cfg(config, "offload_models", default=None),
        device=args.device,
    )
    d44._freeze_stage3c_probe_model(module)
    module.eval()
    return module, {
        "checkpoint": checkpoint,
        "attention_mode": args.stage3_real_attention_mode or _cfg(config, "stage3_real_attention_mode", default="dense_full"),
        "lq_proj_temporal_mode": getattr(module.pipe.lq_proj_in, "temporal_mode", None),
        "lq_proj_scale": float(module.pipe.lq_proj_scale),
        "lq_latent_alignment": d44._stage3d31_teacher_lq_alignment_mode(module.pipe),
    }


def _prepare_stage3_inputs(module, data: dict[str, Any]):
    pipe = module.pipe
    inputs = module.get_pipeline_inputs(data)
    merged_inputs = module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    with torch.inference_mode():
        for unit in pipe.units:
            merged_inputs = pipe.unit_runner(unit, pipe, *merged_inputs)
    merged: dict[str, Any] = {}
    merged.update(merged_inputs[0])
    merged.update(merged_inputs[1])
    merged["lq_latent_alignment"] = d44._stage3d31_teacher_lq_alignment_mode(pipe)
    return merged


def _prepare_stage1_inputs(pipe, data: dict[str, Any], height: int, width: int, num_frames: int, seed: int):
    inputs_shared = {
        "input_video": None,
        "lq_video": data["lq_video"],
        "seed": seed,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": 1.0,
        "cfg_merge": False,
        "tiled": False,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "framewise_decoding": False,
        "vace_reference_image": None,
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "lq_proj_scale": pipe.lq_proj_scale,
    }
    inputs_posi: dict[str, Any] = {}
    inputs_nega: dict[str, Any] = {}
    with torch.inference_mode():
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)
    merged: dict[str, Any] = {}
    merged.update(inputs_shared)
    merged.update(inputs_posi)
    return merged


def _run_stage1_x0(pipe, merged: dict[str, Any], clean_latents: torch.Tensor, dmd_point: dict[str, torch.Tensor]):
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    run_inputs = dict(merged)
    run_inputs["input_latents"] = clean_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
    run_inputs["latents"] = dmd_point["noisy_latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
    timestep = dmd_point["timestep"].to(device=pipe.device, dtype=pipe.torch_dtype)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    with torch.inference_mode():
        pipe.load_models_to_device(pipe.in_iteration_models)
        noise_pred = pipe.model_fn(**models, **run_inputs, timestep=timestep)
        x0_pred = pipe.scheduler.step(noise_pred, timestep, run_inputs["latents"], to_final=True)
    return x0_pred.detach(), noise_pred.detach()


def _build_dmd_point(pipe, clean_latents: torch.Tensor, seed: int, timestep_id: int | None):
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    if timestep_id is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        timestep_id_tensor = torch.randint(0, len(pipe.scheduler.timesteps), (1,), generator=generator)
    else:
        timestep_id_tensor = torch.tensor([int(timestep_id)], dtype=torch.long)
    timestep = pipe.scheduler.timesteps[timestep_id_tensor].to(dtype=pipe.torch_dtype, device=pipe.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1)
    noise = torch.randn(clean_latents.shape, generator=generator, dtype=torch.float32).to(
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timestep)
    return {
        "timestep": timestep.detach(),
        "timestep_id": timestep_id_tensor.detach(),
        "noise": noise.detach(),
        "noisy_latents": noisy_latents.detach(),
    }


def _summarize_lq_latents(lq_latents: Any) -> dict[str, Any] | None:
    if not isinstance(lq_latents, (list, tuple)):
        return None
    return {
        "num_layers": len(lq_latents),
        "layers": [_tensor_stats(layer) for layer in lq_latents[: min(4, len(lq_latents))]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate1 Stage1 teacher vs Stage3 G_real equivalence probe.")
    parser.add_argument("--fixed_batch", default="", help="Path to sample_XX.pt containing video/lq_video.")
    parser.add_argument("--fixed_lqgt_root", default="", help="Root containing sample_XX.pt files.")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--stage1_config", default="", help="Stage1 YAML. Used for model paths/prompt/lq mode.")
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage3_config", required=True, help="Stage3 D4.4 YAML.")
    parser.add_argument("--stage3_real_checkpoint", default="", help="Defaults to --stage1_checkpoint.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model_dir", default="", help="Fallback base model dir if YAML has no model_paths.")
    parser.add_argument("--prompt_tensor_path", default="", help="Override prompt tensor path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026052601)
    parser.add_argument("--timestep_id", type=int, default=None)
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--lq_proj_scale", type=float, default=None)
    parser.add_argument("--stage1_lq_proj_temporal_mode", default="")
    parser.add_argument("--stage3_real_lq_proj_temporal_mode", default="")
    parser.add_argument("--stage3_real_attention_mode", default="")
    parser.add_argument("--decode_tiled", action="store_true", default=False)
    parser.add_argument("--save_tensors", action="store_true", default=False)
    args = parser.parse_args()

    if not args.fixed_batch and not args.fixed_lqgt_root:
        raise ValueError("Pass --fixed_batch or --fixed_lqgt_root.")

    out_dir = Path(args.output_dir)
    (out_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1_teacher").mkdir(parents=True, exist_ok=True)
    (out_dir / "stage3_greal").mkdir(parents=True, exist_ok=True)

    stage1_config = _read_yaml(args.stage1_config)
    stage3_config = _read_yaml(args.stage3_config)
    stage3_real_checkpoint = args.stage3_real_checkpoint or args.stage1_checkpoint

    data, sample_meta = _load_fixed_batch(args)
    video = data["video"][0]
    lq_video = data["lq_video"][0]
    height = int(video.shape[-2])
    width = int(video.shape[-1])
    num_frames = int(video.shape[0])

    report: dict[str, Any] = {
        "tool": "stage3_gate1_teacher_equivalence_probe",
        "sample": sample_meta,
        "args": vars(args),
        "inputs": {
            "gt_video": _save_input_video(video, out_dir / "inputs" / "gt.mp4", args.fps),
            "lq_video": _save_input_video(lq_video, out_dir / "inputs" / "lq.mp4", args.fps),
        },
    }

    stage1_pipe, stage1_meta = _build_stage1_teacher_pipe(args, stage1_config or stage3_config, args.stage1_checkpoint)
    stage3_module, stage3_meta = _build_stage3_greal_module(args, stage3_config, stage3_real_checkpoint)
    report["stage1_teacher"] = {"meta": stage1_meta}
    report["stage3_greal"] = {"meta": stage3_meta}

    with torch.inference_mode():
        stage3_merged = _prepare_stage3_inputs(stage3_module, data)
        clean_latents = stage3_merged["input_latents"].detach()
        dmd_point = _build_dmd_point(stage3_module.pipe, clean_latents, args.seed, args.timestep_id)

        stage1_merged = _prepare_stage1_inputs(stage1_pipe, data, height, width, num_frames, args.seed)
        stage1_x0, stage1_noise_pred = _run_stage1_x0(stage1_pipe, stage1_merged, clean_latents, dmd_point)
        stage3_x0, returned_point = d44._stage3c_probe_predict_x0(
            stage3_module,
            data,
            clean_latents,
            dataset_load_from_cache=False,
            probe_name="gate1_greal",
            dmd_point=dmd_point,
            return_dmd_point=True,
        )

    report["shared_point"] = {
        "clean_latents": _tensor_stats(clean_latents),
        "noise": _tensor_stats(dmd_point["noise"]),
        "noisy_latents": _tensor_stats(dmd_point["noisy_latents"]),
        "timestep": _tensor_stats(dmd_point["timestep"]),
        "timestep_id": int(dmd_point["timestep_id"].flatten()[0].item()),
    }
    report["stage1_teacher"].update(
        {
            "lq_latents": _summarize_lq_latents(stage1_merged.get("lq_latents")),
            "noise_pred": _tensor_stats(stage1_noise_pred),
            "x0": _tensor_stats(stage1_x0),
        }
    )
    report["stage3_greal"].update(
        {
            "lq_latents": _summarize_lq_latents(stage3_merged.get("lq_latents")),
            "x0": _tensor_stats(stage3_x0),
            "returned_timestep_id": int(returned_point["timestep_id"].flatten()[0].detach().cpu().item()),
        }
    )
    report["comparison"] = {
        "x0_stage1_minus_stage3": _tensor_diff_stats(stage1_x0, stage3_x0),
        "lq_latent_alignment_note": (
            "Stage1 path uses WanFixedPromptFlashVSRStage1Pipeline; "
            "Stage3 path uses D4.4 G_real wrapper. Both share clean/noisy latents and timestep."
        ),
    }

    report["stage1_teacher"]["decoded_x0"] = _decode_latents(
        stage1_pipe,
        stage1_x0,
        out_dir / "stage1_teacher" / "x0_decode.mp4",
        args.fps,
        args.decode_tiled,
    )
    report["stage3_greal"]["decoded_x0"] = _decode_latents(
        stage3_module.pipe,
        stage3_x0,
        out_dir / "stage3_greal" / "x0_decode.mp4",
        args.fps,
        args.decode_tiled,
    )

    if args.save_tensors:
        tensor_dir = out_dir / "tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "clean_latents": clean_latents.detach().cpu(),
                "noise": dmd_point["noise"].detach().cpu(),
                "noisy_latents": dmd_point["noisy_latents"].detach().cpu(),
                "timestep": dmd_point["timestep"].detach().cpu(),
                "timestep_id": dmd_point["timestep_id"].detach().cpu(),
                "stage1_x0": stage1_x0.detach().cpu(),
                "stage1_noise_pred": stage1_noise_pred.detach().cpu(),
                "stage3_greal_x0": stage3_x0.detach().cpu(),
            },
            tensor_dir / "gate1_tensors.pt",
        )
        report["tensors_path"] = str(tensor_dir / "gate1_tensors.pt")

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"[stage3_gate1] wrote {report_path}", flush=True)
    print(
        "[stage3_gate1] x0_diff "
        + " ".join(f"{key}={value}" for key, value in report["comparison"]["x0_stage1_minus_stage3"].items() if isinstance(value, (int, float))),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
