#!/usr/bin/env python3
"""Gate 2 condition-alignment dump tool for Stage3 DMD probes.

This is a standalone debug tool. It intentionally does not modify or import-
patch the production v7-D4.4 training entrypoint. It builds student/G_real/
G_fake from the same YAML/CLI arguments, runs one fixed batch and one fixed
DMD point, then writes a JSON report describing whether real/fake inputs are
identical except for model parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.models import wan_video_dit_stage2_v6_1 as stage2_attn
from wanvideo.model_training.flashvsr import train_flashvsr_stage3_v7_d4_4_lora as stage3


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    try:
        raw = value.view(torch.uint8).numpy().tobytes()
    except Exception:
        raw = value.float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_stats(tensor: torch.Tensor, *, include_hash: bool = True) -> dict[str, Any]:
    value = tensor.detach()
    value_float = value.float()
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "min": float(value_float.min().detach().cpu().item()),
        "max": float(value_float.max().detach().cpu().item()),
        "mean": float(value_float.mean().detach().cpu().item()),
        "std": float(value_float.std().detach().cpu().item()) if value.numel() > 1 else 0.0,
        "abs_mean": float(value_float.abs().mean().detach().cpu().item()),
        "finite": bool(torch.isfinite(value_float).all().detach().cpu().item()),
    }
    if include_hash:
        result["sha256"] = _tensor_hash(value)
    return result


def _maybe_tensor_stats(value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_stats(value)
    if isinstance(value, (list, tuple)) and all(torch.is_tensor(item) for item in value):
        return [_tensor_stats(item) for item in value]
    return _jsonable(value)


def _hashes_for(value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_hash(value)
    if isinstance(value, (list, tuple)) and all(torch.is_tensor(item) for item in value):
        return [_tensor_hash(item) for item in value]
    return None


def _batch_to_device(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            result[key] = value.to(device=device)
        else:
            result[key] = value
    return result


def _ensure_batched_sample(sample: dict[str, Any]) -> dict[str, Any]:
    result = dict(sample)
    for key in ("video", "lq_video"):
        tensor = result.get(key)
        if torch.is_tensor(tensor) and tensor.ndim == 4:
            result[key] = tensor.unsqueeze(0)
    return result


def _load_or_build_batch(args, debug_args) -> tuple[dict[str, Any], dict[str, Any]]:
    if debug_args.batch_pt:
        payload = torch.load(debug_args.batch_pt, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"--batch_pt must contain a dict, got {type(payload)!r}")
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise TypeError("--batch_pt payload['data'] must be a dict")
        return _ensure_batched_sample(data), {"source": "batch_pt", "path": debug_args.batch_pt}

    dataset = stage3.v6.FlashVSRStage2VideoOnlyDataset(
        yubari_video_tar_url=args.yubari_video_tar_url,
        takano_video_tar_url=args.takano_video_tar_url,
        internal_url=args.internal_url,
        yubari_video_prob=args.yubari_video_prob,
        takano_video_prob=args.takano_video_prob,
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
    iterator = iter(dataset)
    sample = next(iterator)
    collate_fn = getattr(dataset, "custom_collate_fn", None)
    if collate_fn is not None:
        data = collate_fn([sample])
    else:
        data = _ensure_batched_sample(sample)
    return data, {
        "source": "dataset_first_sample",
        "source_dataset": sample.get("source_dataset"),
        "sample_seed": stage3.v5._serialize_sample_seed(sample.get("sample_seed")),
    }


def _build_stage3_model(args, *, role: str, checkpoint: str | None, attention_mode: str, lq_temporal_mode: str, device: str):
    if role == "student":
        checkpoint = args.resume_stage1_checkpoint
        attention_mode = args.stage2_attention_mode
        lq_temporal_mode = getattr(args, "lq_proj_temporal_mode", "streaming")
        freeze_lq = args.freeze_lq_proj_in
        zero_init = args.zero_init_lq_proj_in
        lq_checkpoint = args.lq_proj_checkpoint
        lora_checkpoint = args.lora_checkpoint
    else:
        if checkpoint is None:
            raise ValueError(f"{role} checkpoint is required for Gate2 dump")
        freeze_lq = True if role == "real" else not bool(args.stage3_fake_train_lq_proj_in)
        zero_init = False
        lq_checkpoint = None
        lora_checkpoint = None

    model = stage3.FlashVSRStage3BTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=lora_checkpoint,
        lq_proj_checkpoint=lq_checkpoint,
        resume_stage1_checkpoint=checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        lq_proj_scale=args.lq_proj_scale,
        zero_init_lq_proj_in=zero_init,
        freeze_lq_proj_in=freeze_lq,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        stage2_attention_mode=attention_mode,
        stage2_topk_ratio=args.stage2_topk_ratio,
        stage2_local_num=args.stage2_local_num,
        lq_proj_temporal_mode=lq_temporal_mode,
        stage3_recon_num_latents=args.stage3_recon_num_latents,
        stage3_flow_weight=args.stage3_flow_weight,
        stage3_mse_weight=0.0,
        stage3_lpips_weight=0.0,
        stage3_lpips_net=args.stage3_lpips_net,
        stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
        stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
        stage3_decoder_cpu_offload=args.stage3_decoder_cpu_offload,
        stage3_compute_z_pred=True,
        stage3_fake_fm_weight=0.0,
        stage3_fake_update_ratio=args.stage3_fake_update_ratio,
        stage3_fake_checkpoint=None,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=device,
    )
    if role in ("real", "fake"):
        stage3._freeze_stage3c_probe_model(model)
    model.eval()
    return model


def _module_report(model) -> dict[str, Any]:
    module = stage3._unwrap_stage3c_model(model)
    pipe = module.pipe
    dit = getattr(pipe, "dit", None)
    lq_proj = getattr(pipe, "lq_proj_in", None)
    return {
        "pipe_device": str(pipe.device),
        "pipe_dtype": str(pipe.torch_dtype),
        "lq_proj_temporal_mode": getattr(lq_proj, "temporal_mode", None),
        "lq_proj_scale": float(getattr(pipe, "lq_proj_scale", 1.0)),
        "attention_mode": getattr(dit, "flashvsr_stage2_attention_mode", None),
        "topk_ratio": getattr(dit, "flashvsr_stage2_topk_ratio", None),
        "local_num": getattr(dit, "flashvsr_stage2_local_num", None),
        "trainable_param_count": int(sum(param.numel() for param in module.parameters() if param.requires_grad)),
    }


def _run_student_clean_latents(student, data: dict[str, Any], seed: int) -> torch.Tensor:
    module = stage3._unwrap_stage3c_model(student)
    module.stage3_mse_weight = 0.0
    module.stage3_lpips_weight = 0.0
    module.stage3_compute_z_pred = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    with torch.no_grad():
        _ = module(data)
    clean_latents = stage3._get_stage3c_last_z_pred(module)
    if clean_latents is None:
        raise RuntimeError("Student did not produce pipe._stage3_last_z_pred")
    return clean_latents.detach()


def _prepare_merged_inputs(model, data: dict[str, Any]) -> dict[str, Any]:
    module = stage3._unwrap_stage3c_model(model)
    pipe = module.pipe
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    inputs = module.get_pipeline_inputs(data)
    merged_inputs = module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        merged_inputs = pipe.unit_runner(unit, pipe, *merged_inputs)
    merged: dict[str, Any] = {}
    merged.update(merged_inputs[0])
    merged.update(merged_inputs[1])
    merged["lq_latent_alignment"] = stage3._stage3d31_teacher_lq_alignment_mode(pipe)
    return merged


def _make_dmd_point(pipe, clean_latents: torch.Tensor, merged: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    max_timestep_boundary = int(merged.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(merged.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    clean = clean_latents.detach().to(device=pipe.device, dtype=pipe.torch_dtype)
    noise = torch.randn_like(clean)
    noisy_latents = pipe.scheduler.add_noise(clean, noise, timestep)
    return {
        "timestep": timestep.detach(),
        "timestep_id": timestep_id.detach().to(device=pipe.device),
        "noise": noise.detach(),
        "noisy_latents": noisy_latents.detach(),
    }


def _mask_basic_stats(mask: torch.Tensor) -> dict[str, Any]:
    mask_bool = mask.detach().bool()
    total = int(mask_bool.numel())
    legal = int(mask_bool.sum().cpu().item())
    return {
        **_tensor_stats(mask_bool, include_hash=True),
        "legal_pairs": legal,
        "total_pairs": total,
        "legal_ratio": float(legal / max(total, 1)),
        "per_query_min": int(mask_bool.sum(dim=-1).min().cpu().item()) if mask_bool.ndim >= 1 else legal,
        "per_query_max": int(mask_bool.sum(dim=-1).max().cpu().item()) if mask_bool.ndim >= 1 else legal,
    }


def _planned_mask_report(model, noisy_latents: torch.Tensor) -> dict[str, Any]:
    module = stage3._unwrap_stage3c_model(model)
    pipe = module.pipe
    dit = pipe.dit
    mode = getattr(dit, "flashvsr_stage2_attention_mode", "unknown")
    report: dict[str, Any] = {
        "attention_mode": mode,
        "topk_ratio": getattr(dit, "flashvsr_stage2_topk_ratio", None),
        "local_num": getattr(dit, "flashvsr_stage2_local_num", None),
    }
    if mode == "dense_full":
        report["kind"] = "dense_full_no_mask"
        return report
    try:
        with torch.no_grad():
            patched = dit.patch_embedding(noisy_latents.to(device=pipe.device, dtype=pipe.torch_dtype))
        frames, height, width = [int(v) for v in patched.shape[2:]]
        win_f, win_h, win_w = stage2_attn.STAGE2_BLOCK_WINDOW
        height_blocks = height // win_h
        width_blocks = width // win_w
        num_heads = int(dit.blocks[0].self_attn.num_heads)
        causal_allowed = stage2_attn.build_stage2_chunk_block_mask(
            batch_size=int(noisy_latents.shape[0]),
            num_heads=num_heads,
            latent_frames=frames,
            height_blocks=height_blocks,
            width_blocks=width_blocks,
            device=pipe.device,
            local_num=getattr(dit, "flashvsr_stage2_local_num", None),
        )
        report.update(
            {
                "kind": "planned_allowed_mask",
                "patch_grid": [frames, height, width],
                "block_window": list(stage2_attn.STAGE2_BLOCK_WINDOW),
                "height_blocks": height_blocks,
                "width_blocks": width_blocks,
                "num_heads": num_heads,
                "causal_allowed": _mask_basic_stats(causal_allowed),
            }
        )
        if mode == "block_sparse_official_mask":
            spatial = stage2_attn.build_official_spatial_local_mask(
                height_blocks,
                width_blocks,
                device=pipe.device,
            )
            chunks = frames // win_f
            spatial_full = spatial.repeat(chunks, chunks).unsqueeze(0).unsqueeze(0)
            report["spatial_local"] = _mask_basic_stats(spatial)
            report["combined_allowed"] = _mask_basic_stats(causal_allowed & spatial_full)
    except Exception as exc:
        report["error"] = repr(exc)
    return report


class TopKMaskCapture:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._original_topk: Callable[..., torch.Tensor] | None = None
        self._original_official: Callable[..., torch.Tensor] | None = None

    def _wrap(self, name: str, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def wrapped(*args, **kwargs):
            selected = original(*args, **kwargs)
            allowed = kwargs.get("allowed_mask")
            record = {
                "selector": name,
                "topk_ratio": kwargs.get("topk_ratio"),
                "spatial_blocks": kwargs.get("spatial_blocks"),
                "selected": _mask_basic_stats(selected),
            }
            if torch.is_tensor(allowed):
                record["allowed"] = _mask_basic_stats(allowed)
            self.records.append(record)
            return selected

        return wrapped

    def __enter__(self):
        self._original_topk = stage2_attn._select_topk_blocks
        self._original_official = stage2_attn._select_topk_blocks_official
        stage2_attn._select_topk_blocks = self._wrap("_select_topk_blocks", self._original_topk)
        stage2_attn._select_topk_blocks_official = self._wrap("_select_topk_blocks_official", self._original_official)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original_topk is not None:
            stage2_attn._select_topk_blocks = self._original_topk
        if self._original_official is not None:
            stage2_attn._select_topk_blocks_official = self._original_official
        return False


def _run_probe_dump(
    *,
    name: str,
    model,
    data: dict[str, Any],
    clean_latents: torch.Tensor,
    dmd_point: dict[str, torch.Tensor],
    seed: int,
    skip_model_fn: bool,
) -> dict[str, Any]:
    module = stage3._unwrap_stage3c_model(model)
    pipe = module.pipe
    was_training = module.training
    module.eval()
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        merged = _prepare_merged_inputs(module, data)
        clean = clean_latents.detach().to(device=pipe.device, dtype=pipe.torch_dtype)
        noisy = dmd_point["noisy_latents"].to(device=pipe.device, dtype=pipe.torch_dtype)
        timestep = dmd_point["timestep"].to(device=pipe.device, dtype=pipe.torch_dtype)
        merged["input_latents"] = clean
        merged["latents"] = noisy
        models = {model_name: getattr(pipe, model_name) for model_name in pipe.in_iteration_models}

        report: dict[str, Any] = {
            "name": name,
            "module": _module_report(module),
            "merged_keys": sorted(merged.keys()),
            "condition": {
                "context": _maybe_tensor_stats(merged.get("context")),
                "lq_latents": _maybe_tensor_stats(merged.get("lq_latents")),
                "lq_latent_alignment": merged.get("lq_latent_alignment"),
                "lq_video": _maybe_tensor_stats(merged.get("lq_video")),
                "input_video": _maybe_tensor_stats(merged.get("input_video")),
            },
            "dmd_point": {
                "input_latents": _tensor_stats(clean),
                "latents": _tensor_stats(noisy),
                "timestep": _tensor_stats(timestep),
                "timestep_id": _maybe_tensor_stats(dmd_point.get("timestep_id")),
                "noise": _tensor_stats(dmd_point["noise"].to(device=pipe.device, dtype=pipe.torch_dtype)),
            },
            "planned_mask": _planned_mask_report(module, noisy),
        }
        if skip_model_fn:
            report["model_fn_skipped"] = True
            return report

        with torch.no_grad(), TopKMaskCapture() as mask_capture:
            started = time.time()
            noise_pred = pipe.model_fn(**models, **merged, timestep=timestep)
            x0_pred = pipe.scheduler.step(noise_pred, timestep, noisy, to_final=True)
            elapsed = time.time() - started
        report["model"] = {
            "elapsed_sec": elapsed,
            "noise_pred": _tensor_stats(noise_pred),
            "x0_pred": _tensor_stats(x0_pred),
        }
        report["actual_topk_masks"] = {
            "count": len(mask_capture.records),
            "records": mask_capture.records[:8],
            "unique_selected_hashes": sorted({item["selected"]["sha256"] for item in mask_capture.records}),
        }
        return report
    finally:
        module.train(was_training)


def _get_path(report: dict[str, Any], path: str) -> Any:
    value: Any = report
    for item in path.split("."):
        if isinstance(value, dict):
            value = value.get(item)
        elif isinstance(value, list):
            try:
                value = value[int(item)]
            except Exception:
                return None
        else:
            return None
    return value


def _alignment_report(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compare_paths = [
        "condition.context.sha256",
        "condition.lq_latents.0.sha256",
        "condition.lq_latent_alignment",
        "dmd_point.input_latents.sha256",
        "dmd_point.latents.sha256",
        "dmd_point.timestep.sha256",
        "dmd_point.noise.sha256",
        "planned_mask.kind",
        "planned_mask.causal_allowed.sha256",
        "planned_mask.combined_allowed.sha256",
        "model.noise_pred.shape",
        "model.x0_pred.shape",
    ]
    result: dict[str, Any] = {"real_fake": {}, "student_real": {}, "all_same": {}}
    pairs = {
        "real_fake": ("real", "fake"),
        "student_real": ("student", "real"),
    }
    for pair_name, (left_name, right_name) in pairs.items():
        left = reports.get(left_name, {})
        right = reports.get(right_name, {})
        entries = {}
        all_equal = True
        for path in compare_paths:
            left_value = _get_path(left, path)
            right_value = _get_path(right, path)
            equal = left_value == right_value
            entries[path] = {"equal": equal, "left": left_value, "right": right_value}
            all_equal = all_equal and equal
        result[pair_name] = {
            "all_checked_fields_equal": all_equal,
            "fields": entries,
        }
    result["conclusion"] = {
        "real_fake_condition_aligned": bool(result["real_fake"]["all_checked_fields_equal"]),
        "note": (
            "real_fake_condition_aligned only checks hashes/shapes for DMD point, context, "
            "first LQ-projector layer, planned mask and output shapes. Full per-layer "
            "LQ hashes are available under each role's condition.lq_latents."
        ),
    }
    return result


def _parse_args(argv=None):
    debug_parser = argparse.ArgumentParser(add_help=False)
    debug_parser.add_argument("--output_json", required=True)
    debug_parser.add_argument("--device", default="cuda")
    debug_parser.add_argument("--batch_pt", default="")
    debug_parser.add_argument("--seed", type=int, default=20260526)
    debug_parser.add_argument("--dmd_seed", type=int, default=20260527)
    debug_parser.add_argument("--mask_seed", type=int, default=20260528)
    debug_parser.add_argument("--skip_model_fn", action="store_true")
    debug_args, stage_argv = debug_parser.parse_known_args(argv)
    stage_args = stage3.parse_stage3_args(stage_argv)
    return debug_args, stage_args


def main(argv=None) -> None:
    debug_args, args = _parse_args(argv)
    output_json = Path(debug_args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    device = debug_args.device
    data_cpu, batch_meta = _load_or_build_batch(args, debug_args)
    data = _batch_to_device(data_cpu, torch.device(device))

    student = _build_stage3_model(
        args,
        role="student",
        checkpoint=args.resume_stage1_checkpoint,
        attention_mode=args.stage2_attention_mode,
        lq_temporal_mode=getattr(args, "lq_proj_temporal_mode", "streaming"),
        device=device,
    )
    real = _build_stage3_model(
        args,
        role="real",
        checkpoint=args.stage3_real_checkpoint,
        attention_mode=args.stage3_real_attention_mode,
        lq_temporal_mode=args.stage3_real_lq_proj_temporal_mode,
        device=device,
    )
    fake = _build_stage3_model(
        args,
        role="fake",
        checkpoint=args.stage3_fake_checkpoint,
        attention_mode=args.stage3_fake_attention_mode,
        lq_temporal_mode=args.stage3_fake_lq_proj_temporal_mode,
        device=device,
    )

    clean_latents = _run_student_clean_latents(student, data, debug_args.seed)
    merged_for_point = _prepare_merged_inputs(real, data)
    dmd_point = _make_dmd_point(real.pipe, clean_latents, merged_for_point, debug_args.dmd_seed)

    role_reports = {
        "student": _run_probe_dump(
            name="student",
            model=student,
            data=data,
            clean_latents=clean_latents,
            dmd_point=dmd_point,
            seed=debug_args.mask_seed,
            skip_model_fn=debug_args.skip_model_fn,
        ),
        "real": _run_probe_dump(
            name="real",
            model=real,
            data=data,
            clean_latents=clean_latents,
            dmd_point=dmd_point,
            seed=debug_args.mask_seed,
            skip_model_fn=debug_args.skip_model_fn,
        ),
        "fake": _run_probe_dump(
            name="fake",
            model=fake,
            data=data,
            clean_latents=clean_latents,
            dmd_point=dmd_point,
            seed=debug_args.mask_seed,
            skip_model_fn=debug_args.skip_model_fn,
        ),
    }
    report = {
        "tool": "stage3_gate2_condition_alignment_dump",
        "config": getattr(args, "config", None),
        "batch": {
            **batch_meta,
            "video": _maybe_tensor_stats(data_cpu.get("video")),
            "lq_video": _maybe_tensor_stats(data_cpu.get("lq_video")),
        },
        "seeds": {
            "student_seed": int(debug_args.seed),
            "dmd_seed": int(debug_args.dmd_seed),
            "mask_seed": int(debug_args.mask_seed),
        },
        "roles": role_reports,
        "alignment": _alignment_report(role_reports),
    }
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(report), file, ensure_ascii=False, indent=2)
    print(f"[stage3_gate2_condition_alignment_dump] wrote {output_json}", flush=True)
    print(json.dumps(report["alignment"]["conclusion"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
