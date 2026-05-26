#!/usr/bin/env python3
"""Analyze Stage3 DMD tensor dumps for sign and normalization sanity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _mean_sq(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().item())


def _mean_abs(x: torch.Tensor) -> float:
    return float(x.float().abs().mean().item())


def _candidate_report(name: str, z: torch.Tensor, grad: torch.Tensor, real_x0: torch.Tensor, fake_x0: torch.Tensor) -> dict[str, Any]:
    rows = []
    base_real = _mean_sq(z, real_x0)
    base_fake = _mean_sq(z, fake_x0)
    for eps in (0.001, 0.01, 0.05, 0.1):
        z_next = z.float() - float(eps) * grad.float()
        real_dist = _mean_sq(z_next, real_x0)
        fake_dist = _mean_sq(z_next, fake_x0)
        rows.append(
            {
                "eps": eps,
                "mse_to_real": real_dist,
                "mse_to_fake": fake_dist,
                "delta_real": real_dist - base_real,
                "delta_fake": fake_dist - base_fake,
            }
        )
    return {
        "name": name,
        "grad_abs_mean": _mean_abs(grad),
        "grad_mean": float(grad.float().mean().item()),
        "base_mse_to_real": base_real,
        "base_mse_to_fake": base_fake,
        "steps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to dmd_tensors.pt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = torch.load(args.dump, map_location="cpu")
    z = payload["clean_latents"].float()
    real_x0 = payload["real_x0"].float()
    fake_x0 = payload["fake_x0"].float()
    p_real = payload["p_real"].float()
    p_fake = payload["p_fake"].float()

    reduce_dims = tuple(range(1, p_real.ndim))
    norm_per_sample_real = p_real.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    norm_global_real = p_real.abs().mean().clamp_min(1e-6)
    norm_per_sample_fake = p_fake.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
    raw = p_real - p_fake

    candidates = {
        "current_preal_minus_pfake_per_sample_real_norm": raw / norm_per_sample_real,
        "flipped_pfake_minus_preal_per_sample_real_norm": -raw / norm_per_sample_real,
        "current_global_real_norm": raw / norm_global_real,
        "current_per_sample_fake_norm": raw / norm_per_sample_fake,
        "raw_no_norm": raw,
    }
    report = {
        "dump": args.dump,
        "raw_abs_mean": _mean_abs(raw),
        "p_real_abs_mean": _mean_abs(p_real),
        "p_fake_abs_mean": _mean_abs(p_fake),
        "real_fake_mse": _mean_sq(real_x0, fake_x0),
        "candidates": [_candidate_report(name, z, grad, real_x0, fake_x0) for name, grad in candidates.items()],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"[dmd_sign_norm_probe] wrote {output}", flush=True)
    for item in report["candidates"]:
        step = item["steps"][1]
        print(
            f"{item['name']} grad_abs={item['grad_abs_mean']:.6f} "
            f"eps=0.01 delta_real={step['delta_real']:.8f} delta_fake={step['delta_fake']:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
