#!/usr/bin/env python3
"""Check whether the Stage3 teacher wrapper preserves Stage1 v5.3.5 LQ projector semantics."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffsynth.core.loader.file import load_state_dict  # noqa: E402
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5  # noqa: E402


def _tensor_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> Dict[str, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "lhs_norm": float(lhs.float().norm().item()),
        "rhs_norm": float(rhs.float().norm().item()),
    }


def _build_projector(mode: str, state: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype):
    linear_weight = state.get("linear_layers.0.weight")
    if linear_weight is None:
        raise KeyError("Checkpoint lq_proj_in state does not contain linear_layers.0.weight")
    projector = v5.FlashVSRLQProjIn(
        in_dim=3,
        out_dim=int(linear_weight.shape[0]),
        layer_num=1,
        zero_init_output=False,
        temporal_mode=mode,
    ).to(device=device, dtype=dtype)
    result = projector.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"load_state_dict mismatch for {mode}: {result}")
    projector.eval()
    return projector


def _run_projector(projector, video: torch.Tensor) -> List[torch.Tensor]:
    with torch.inference_mode():
        outputs = projector(video)
    if outputs is None:
        raise RuntimeError(f"{projector.temporal_mode} projector returned None")
    return [item.detach().cpu() for item in outputs]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=int, default=89)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("=== Stage1/Stage3 Teacher Projector Equivalence Check ===", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"input_shape=(1, 3, {args.frames}, {args.height}, {args.width}) device={device} dtype={dtype}", flush=True)
    print("code_fact_stage1_pipeline_lq_proj_temporal_mode=configurable", flush=True)
    print("code_fact_stage2_stage3_wrapper_lq_proj_temporal_mode=streaming_hardcoded", flush=True)
    print("expected_stage1_v5_3_5_run_hint=nonstreamproj_aligned23", flush=True)

    state_dict = load_state_dict(args.checkpoint, device="cpu")
    lq_state, lora_state, other_state = v5.flashvsr_stage1_split_exported_state(state_dict)
    print(
        f"checkpoint_keys lq_proj={len(lq_state)} lora={len(lora_state)} other={len(other_state)}",
        flush=True,
    )
    if not lq_state:
        raise RuntimeError("No lq_proj_in state found in checkpoint.")

    torch.manual_seed(args.seed)
    video = torch.randn(1, 3, args.frames, args.height, args.width, device=device, dtype=dtype)

    modes = ["nonstreaming_aligned", "streaming", "nonstreaming"]
    outputs: Dict[str, List[torch.Tensor]] = {}
    for mode in modes:
        projector = _build_projector(mode, lq_state, device, dtype)
        outputs[mode] = _run_projector(projector, video)
        shapes = [tuple(item.shape) for item in outputs[mode]]
        print(f"mode={mode} output_shapes={shapes}", flush=True)
        del projector
        torch.cuda.empty_cache()

    stage1 = outputs["nonstreaming_aligned"][0]
    stage3 = outputs["streaming"][0]
    if tuple(stage1.shape) != tuple(stage3.shape):
        print(
            "RESULT=FAIL_NOT_EQUIVALENT "
            f"reason=shape_mismatch stage1_nonstreaming_aligned={tuple(stage1.shape)} "
            f"stage3_wrapper_streaming={tuple(stage3.shape)}",
            flush=True,
        )
        return 2

    stats = _tensor_stats(stage1, stage3)
    print(
        "compare_nonstreaming_aligned_vs_streaming "
        + " ".join(f"{key}={value:.8g}" for key, value in stats.items()),
        flush=True,
    )
    if stats["max_abs"] == 0.0:
        print("RESULT=PASS_PROJECTOR_EQUIVALENT", flush=True)
        return 0
    print("RESULT=FAIL_NOT_EQUIVALENT reason=value_mismatch", flush=True)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
