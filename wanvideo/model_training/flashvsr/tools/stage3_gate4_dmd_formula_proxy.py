#!/usr/bin/env python3
"""Offline Gate-4 DMD formula proxy analysis.

This tool intentionally does not import or modify any training entrypoint.  It
loads an existing tensor dump that contains the student one-step prediction and
the G_real/G_fake x0 predictions, then compares DMD formula variants offline.

Expected tensors, with flexible aliases:
  - student_z_pred / z_pred / clean_latents / student_x0
  - g_real_x0 / real_x0 / real_dmd_x0
  - g_fake_x0 / fake_x0 / fake_dmd_x0
  - noisy_latents is optional and is only reported when present

The proxy update follows the D44/DMD2 student loss direction:
  dmd_grad = ((z - real_x0) - (z - fake_x0)) / norm
  proxy_after = z - proxy_step * dmd_grad

Good variants should usually decrease distance to G_real and increase distance
to G_fake under this one-step proxy.  dfake is intentionally not a variable here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch


ALIASES = {
    "student_z_pred": (
        "student_z_pred",
        "z_pred",
        "clean_latents",
        "student_x0",
        "student_clean_latents",
        "student",
    ),
    "g_real_x0": (
        "g_real_x0",
        "real_x0",
        "real_dmd_x0",
        "g_real",
        "teacher_real_x0",
    ),
    "g_fake_x0": (
        "g_fake_x0",
        "fake_x0",
        "fake_dmd_x0",
        "g_fake",
        "teacher_fake_x0",
    ),
    "noisy_latents": (
        "noisy_latents",
        "dmd_noisy_latents",
        "x_t",
        "latents_noisy",
    ),
    "timestep": (
        "timestep",
        "timesteps",
        "dmd_timestep",
        "t",
    ),
}


def _flatten_mapping(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict/list containers while keeping terminal values."""

    flat: Dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = str(key)
            full = f"{prefix}.{name}" if prefix else name
            flat[full] = value
            flat.update(_flatten_mapping(value, full))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            full = f"{prefix}.{idx}" if prefix else str(idx)
            flat[full] = value
            flat.update(_flatten_mapping(value, full))
    return flat


def _load_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, Mapping):
            return dict(obj)
        return {path.stem: obj}
    if suffix == ".npz":
        import numpy as np

        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            obj = json.load(file)
        if isinstance(obj, Mapping):
            return dict(obj)
        return {path.stem: obj}
    raise ValueError(f"unsupported input file suffix: {path}")


def _iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.pt", "*.pth", "*.npz", "*.json"):
        yield from sorted(path.rglob(pattern))


def _load_dump(path: Path) -> Dict[str, Any]:
    """Load a dump file or merge tensor-like files from a directory."""

    if path.is_file():
        return _load_file(path)

    merged: Dict[str, Any] = {}
    archive_candidates: list[tuple[Path, Dict[str, Any]]] = []
    for file_path in _iter_candidate_files(path):
        try:
            loaded = _load_file(file_path)
        except Exception:
            continue
        archive_candidates.append((file_path, loaded))
        for key, value in loaded.items():
            merged.setdefault(key, value)
            merged.setdefault(file_path.stem, value)
            merged.setdefault(f"{file_path.stem}.{key}", value)

    # Prefer a single archive that already contains all required keys.
    for file_path, loaded in archive_candidates:
        flat = _flatten_mapping(loaded)
        if all(_find_key(flat, ALIASES[name]) is not None for name in ("student_z_pred", "g_real_x0", "g_fake_x0")):
            result = dict(loaded)
            result["_source_file"] = str(file_path)
            return result

    merged["_source_file"] = str(path)
    return merged


def _find_key(flat: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    lowered = {key.lower(): key for key in flat}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]

    # Allow nested names such as "dmd.student_z_pred" or file stems.
    for alias in aliases:
        alias_l = alias.lower()
        for key in flat:
            parts = key.lower().replace("/", ".").split(".")
            if alias_l in parts or key.lower().endswith(f".{alias_l}"):
                return key
    return None


def _as_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.detach().cpu().float()
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have batch plus feature dims, got shape {tuple(tensor.shape)}")
    return tensor


def _extract_tensors(raw: Mapping[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    flat = _flatten_mapping(raw)
    tensors: Dict[str, torch.Tensor] = {}
    sources: Dict[str, str] = {}
    for canonical in ("student_z_pred", "g_real_x0", "g_fake_x0", "noisy_latents", "timestep"):
        key = _find_key(flat, ALIASES[canonical])
        if key is None:
            if canonical in {"noisy_latents", "timestep"}:
                continue
            aliases = ", ".join(ALIASES[canonical])
            raise KeyError(f"missing required tensor {canonical}; accepted aliases: {aliases}")
        try:
            tensor = _as_tensor(flat[key], canonical)
        except Exception as exc:
            if canonical in {"noisy_latents", "timestep"}:
                continue
            raise ValueError(f"failed to convert {canonical} from key {key}: {exc}") from exc
        tensors[canonical] = tensor
        sources[canonical] = key
    return tensors, sources


def _check_shapes(tensors: Mapping[str, torch.Tensor]) -> None:
    z_shape = tuple(tensors["student_z_pred"].shape)
    for name in ("g_real_x0", "g_fake_x0"):
        shape = tuple(tensors[name].shape)
        if shape != z_shape:
            raise ValueError(f"shape mismatch: student_z_pred {z_shape}, {name} {shape}")


def _reduce_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, tensor.ndim))


def _per_sample_mean(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().mean(dim=_reduce_dims(tensor))


def _per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _per_sample_mean((a.float() - b.float()).pow(2))


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    return _per_sample_mean(tensor.float().pow(2)).sqrt()


def _per_sample_absmean(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().abs().mean(dim=_reduce_dims(tensor))


def _per_sample_absmax(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().abs().amax(dim=_reduce_dims(tensor))


def _safe_float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected scalar tensor")
        value = value.detach().cpu().float().item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return value
    return value


def _variant_grad(
    variant: str,
    z: torch.Tensor,
    real_x0: torch.Tensor,
    fake_x0: torch.Tensor,
    *,
    epsilon: float,
    strict_loss_max: float,
    strict_grad_absmean_max: float,
    d44_grad_absmean_max: float,
    d44_loss_max: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    p_real = z.float() - real_x0.float()
    p_fake = z.float() - fake_x0.float()
    raw = p_real - p_fake
    reduce_dims = _reduce_dims(raw)

    norm_factor: torch.Tensor | None
    grad = raw
    notes: Dict[str, Any] = {
        "normalization": "none",
        "sign": "normal",
        "loss_clamped": False,
        "grad_absmean_clamped": False,
        "grad_zeroed": False,
    }

    if variant in {"d44_current", "dmd2_absmean", "reverse_sign", "strict_clamp"}:
        norm_factor = p_real.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(epsilon)
        grad = raw / norm_factor
        notes["normalization"] = "per_sample_abs_p_real_mean"
    elif variant == "raw_no_norm":
        norm_factor = None
    else:
        raise ValueError(f"unknown variant: {variant}")

    if variant == "reverse_sign":
        grad = -grad
        notes["sign"] = "reversed"

    grad = torch.nan_to_num(grad)

    if variant == "d44_current":
        if d44_grad_absmean_max > 0.0:
            current = grad.detach().float().abs().mean()
            if torch.isfinite(current) and float(current.item()) > d44_grad_absmean_max:
                grad = grad * (d44_grad_absmean_max / current.clamp_min(epsilon))
                notes["grad_absmean_clamped"] = True
        if d44_loss_max > 0.0:
            current_loss = 0.5 * grad.detach().float().pow(2).mean()
            if torch.isfinite(current_loss) and float(current_loss.item()) > d44_loss_max:
                scale = torch.sqrt(torch.tensor(d44_loss_max, dtype=torch.float32) / current_loss.clamp_min(1e-12))
                grad = grad * scale.to(dtype=grad.dtype)
                notes["loss_clamped"] = True

    if variant == "strict_clamp":
        if strict_grad_absmean_max > 0.0:
            current = grad.detach().float().abs().mean()
            if torch.isfinite(current) and float(current.item()) > strict_grad_absmean_max:
                grad = grad * (strict_grad_absmean_max / current.clamp_min(epsilon))
                notes["grad_absmean_clamped"] = True
        if strict_loss_max > 0.0:
            current_loss = 0.5 * grad.detach().float().pow(2).mean()
            if torch.isfinite(current_loss) and float(current_loss.item()) > strict_loss_max:
                scale = torch.sqrt(torch.tensor(strict_loss_max, dtype=torch.float32) / current_loss.clamp_min(1e-12))
                grad = grad * scale.to(dtype=grad.dtype)
                notes["loss_clamped"] = True

    if norm_factor is None:
        norm_per_sample = torch.ones(z.shape[0], dtype=torch.float32)
    else:
        norm_per_sample = norm_factor.detach().float().reshape(z.shape[0], -1).mean(dim=1)
    notes["norm_factor_per_sample"] = norm_per_sample
    return grad, notes


def _summarize(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().cpu().float().reshape(-1)
    return {
        "mean": _safe_float(values.mean()),
        "min": _safe_float(values.min()),
        "max": _safe_float(values.max()),
        "std": _safe_float(values.std(unbiased=False)) if values.numel() > 1 else 0.0,
    }


def _cosine_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.float().reshape(a.shape[0], -1)
    b_flat = b.float().reshape(b.shape[0], -1)
    numerator = (a_flat * b_flat).sum(dim=1)
    denominator = a_flat.norm(dim=1).clamp_min(1e-12) * b_flat.norm(dim=1).clamp_min(1e-12)
    return numerator / denominator


def analyze(
    tensors: Mapping[str, torch.Tensor],
    *,
    variants: Sequence[str],
    proxy_step: float,
    epsilon: float,
    strict_loss_max: float,
    strict_grad_absmean_max: float,
    d44_grad_absmean_max: float,
    d44_loss_max: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    z = tensors["student_z_pred"].float()
    real_x0 = tensors["g_real_x0"].float()
    fake_x0 = tensors["g_fake_x0"].float()
    p_real = z - real_x0
    p_fake = z - fake_x0

    rows: list[Dict[str, Any]] = []
    sample_rows: list[Dict[str, Any]] = []
    before_real = _per_sample_mse(z, real_x0)
    before_fake = _per_sample_mse(z, fake_x0)

    for variant in variants:
        grad, notes = _variant_grad(
            variant,
            z,
            real_x0,
            fake_x0,
            epsilon=epsilon,
            strict_loss_max=strict_loss_max,
            strict_grad_absmean_max=strict_grad_absmean_max,
            d44_grad_absmean_max=d44_grad_absmean_max,
            d44_loss_max=d44_loss_max,
        )
        after = z - float(proxy_step) * grad
        after_real = _per_sample_mse(after, real_x0)
        after_fake = _per_sample_mse(after, fake_x0)
        move = after - z
        grad_absmean = _per_sample_absmean(grad)
        grad_rms = _per_sample_rms(grad)
        grad_absmax = _per_sample_absmax(grad)
        norm_factor = notes.pop("norm_factor_per_sample")

        real_delta = after_real - before_real
        fake_delta = after_fake - before_fake
        move_to_real_cos = _cosine_per_sample(move, real_x0 - z)
        move_away_fake_cos = _cosine_per_sample(move, z - fake_x0)
        dmd_loss_unweighted = 0.5 * _per_sample_mean(grad.pow(2))

        row = {
            "variant": variant,
            "proxy_step": float(proxy_step),
            "batch_size": int(z.shape[0]),
            "shape": list(z.shape),
            "dfake_fixed": 5,
            "dist_to_real_before_mse_mean": _safe_float(before_real.mean()),
            "dist_to_real_after_mse_mean": _safe_float(after_real.mean()),
            "dist_to_real_delta_mse_mean": _safe_float(real_delta.mean()),
            "dist_to_fake_before_mse_mean": _safe_float(before_fake.mean()),
            "dist_to_fake_after_mse_mean": _safe_float(after_fake.mean()),
            "dist_to_fake_delta_mse_mean": _safe_float(fake_delta.mean()),
            "real_distance_decreased_fraction": _safe_float((real_delta < 0).float().mean()),
            "fake_distance_increased_fraction": _safe_float((fake_delta > 0).float().mean()),
            "grad_absmean_mean": _safe_float(grad_absmean.mean()),
            "grad_rms_mean": _safe_float(grad_rms.mean()),
            "grad_absmax_mean": _safe_float(grad_absmax.mean()),
            "norm_factor_mean": _safe_float(norm_factor.mean()),
            "norm_factor_min": _safe_float(norm_factor.min()),
            "norm_factor_max": _safe_float(norm_factor.max()),
            "dmd_loss_unweighted_mean": _safe_float(dmd_loss_unweighted.mean()),
            "move_to_real_cos_mean": _safe_float(move_to_real_cos.mean()),
            "move_away_fake_cos_mean": _safe_float(move_away_fake_cos.mean()),
            **{key: value for key, value in notes.items() if not isinstance(value, torch.Tensor)},
        }
        rows.append(row)

        for sample_idx in range(z.shape[0]):
            sample_rows.append(
                {
                    "variant": variant,
                    "sample": int(sample_idx),
                    "proxy_step": float(proxy_step),
                    "dist_to_real_before_mse": _safe_float(before_real[sample_idx]),
                    "dist_to_real_after_mse": _safe_float(after_real[sample_idx]),
                    "dist_to_real_delta_mse": _safe_float(real_delta[sample_idx]),
                    "dist_to_fake_before_mse": _safe_float(before_fake[sample_idx]),
                    "dist_to_fake_after_mse": _safe_float(after_fake[sample_idx]),
                    "dist_to_fake_delta_mse": _safe_float(fake_delta[sample_idx]),
                    "grad_absmean": _safe_float(grad_absmean[sample_idx]),
                    "grad_rms": _safe_float(grad_rms[sample_idx]),
                    "grad_absmax": _safe_float(grad_absmax[sample_idx]),
                    "norm_factor": _safe_float(norm_factor[sample_idx]),
                    "dmd_loss_unweighted": _safe_float(dmd_loss_unweighted[sample_idx]),
                    "move_to_real_cos": _safe_float(move_to_real_cos[sample_idx]),
                    "move_away_fake_cos": _safe_float(move_away_fake_cos[sample_idx]),
                    "loss_clamped": bool(row["loss_clamped"]),
                    "grad_absmean_clamped": bool(row["grad_absmean_clamped"]),
                    "normalization": row["normalization"],
                    "sign": row["sign"],
                }
            )

    return rows, sample_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path, help="Tensor dump file or directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for JSON/CSV reports.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["d44_current", "dmd2_absmean", "reverse_sign", "strict_clamp", "raw_no_norm"],
        choices=["d44_current", "dmd2_absmean", "reverse_sign", "strict_clamp", "raw_no_norm"],
        help="Formula variants to compare. dfake is fixed to 5 and is not a variant.",
    )
    parser.add_argument("--proxy-step", type=float, default=1.0, help="Offline proxy update scale: z_after = z - step * grad.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Minimum normalization denominator.")
    parser.add_argument("--d44-grad-absmean-max", type=float, default=0.0, help="Optional D44 grad absmean clamp.")
    parser.add_argument("--d44-loss-max", type=float, default=0.0, help="Optional D44 unweighted loss clamp.")
    parser.add_argument("--strict-grad-absmean-max", type=float, default=5.0, help="Strict-clamp variant grad absmean max.")
    parser.add_argument("--strict-loss-max", type=float, default=1.0, help="Strict-clamp variant unweighted loss max.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = _load_dump(args.dump)
    tensors, sources = _extract_tensors(raw)
    _check_shapes(tensors)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, sample_rows = analyze(
        tensors,
        variants=args.variants,
        proxy_step=args.proxy_step,
        epsilon=args.epsilon,
        strict_loss_max=args.strict_loss_max,
        strict_grad_absmean_max=args.strict_grad_absmean_max,
        d44_grad_absmean_max=args.d44_grad_absmean_max,
        d44_loss_max=args.d44_loss_max,
    )

    meta = {
        "input": str(args.dump),
        "source_file": raw.get("_source_file"),
        "tensor_sources": sources,
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "dfake_fixed": 5,
        "variants": list(args.variants),
        "proxy_step": float(args.proxy_step),
        "clamp_args": {
            "d44_grad_absmean_max": float(args.d44_grad_absmean_max),
            "d44_loss_max": float(args.d44_loss_max),
            "strict_grad_absmean_max": float(args.strict_grad_absmean_max),
            "strict_loss_max": float(args.strict_loss_max),
        },
    }
    report = {
        "meta": meta,
        "summary": rows,
        "per_sample": sample_rows,
    }
    json_path = args.out / "gate4_dmd_formula_proxy.json"
    csv_path = args.out / "gate4_dmd_formula_proxy_summary.csv"
    per_sample_csv_path = args.out / "gate4_dmd_formula_proxy_per_sample.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    _write_csv(csv_path, rows)
    _write_csv(per_sample_csv_path, sample_rows)

    print(f"[gate4] wrote {json_path}")
    print(f"[gate4] wrote {csv_path}")
    print(f"[gate4] wrote {per_sample_csv_path}")
    for row in rows:
        print(
            "[gate4] "
            f"{row['variant']}: "
            f"real_delta={row['dist_to_real_delta_mse_mean']:.6g} "
            f"fake_delta={row['dist_to_fake_delta_mse_mean']:.6g} "
            f"grad_absmean={row['grad_absmean_mean']:.6g} "
            f"loss_clamped={row['loss_clamped']}"
        )


if __name__ == "__main__":
    main()
