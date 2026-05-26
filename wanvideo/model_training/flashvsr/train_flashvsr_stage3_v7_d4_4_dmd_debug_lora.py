"""Stage3 v7-D4.4 DMD tensor-dump debug entrypoint.

This file intentionally does not modify the production v7-D4.4 trainer. It
imports the overfit trainer, replaces only the DMD student-loss helper, and
then runs the normal overfit main path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from wanvideo.model_training.flashvsr import train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora as base


def _dump_step_enabled(global_step: int) -> bool:
    raw_steps = os.environ.get("FLASHVSR_STAGE3_DMD_TENSOR_DUMP_STEPS", "1,2,5,10,20,50,100,200,500,1000")
    steps = {int(item.strip()) for item in raw_steps.split(",") if item.strip()}
    return int(global_step) in steps


def _summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
        "abs_mean": float(value.abs().mean().item()),
    }


def _dump_dmd_tensors(
    *,
    output_dir: str,
    global_step: int,
    clean_latents: torch.Tensor,
    real_x0: torch.Tensor,
    fake_x0: torch.Tensor,
    p_real: torch.Tensor,
    p_fake: torch.Tensor,
    dmd_grad: torch.Tensor,
    dmd_point: dict[str, torch.Tensor],
    weight_factor: torch.Tensor,
    dmd_loss: torch.Tensor,
) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return
    out = Path(output_dir) / f"step_{int(global_step):06d}"
    out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "clean_latents": clean_latents.detach().cpu(),
        "real_x0": real_x0.detach().cpu(),
        "fake_x0": fake_x0.detach().cpu(),
        "p_real": p_real.detach().cpu(),
        "p_fake": p_fake.detach().cpu(),
        "dmd_grad": dmd_grad.detach().cpu(),
        "noisy_latents": dmd_point["noisy_latents"].detach().cpu(),
        "noise": dmd_point["noise"].detach().cpu(),
        "timestep": dmd_point["timestep"].detach().cpu(),
        "timestep_id": dmd_point["timestep_id"].detach().cpu(),
        "weight_factor": weight_factor.detach().cpu(),
    }
    torch.save(tensors, out / "dmd_tensors.pt")
    meta = {
        "global_step": int(global_step),
        "clean_latents": _summary(clean_latents),
        "real_x0": _summary(real_x0),
        "fake_x0": _summary(fake_x0),
        "real_minus_fake": _summary(real_x0.float() - fake_x0.float()),
        "p_real": _summary(p_real),
        "p_fake": _summary(p_fake),
        "p_real_minus_p_fake": _summary(p_real.float() - p_fake.float()),
        "dmd_grad": _summary(dmd_grad),
        "weight_factor": _summary(weight_factor),
        "dmd_loss_unweighted": float(dmd_loss.detach().float().item()),
        "timestep": float(dmd_point["timestep"].detach().flatten()[0].float().cpu().item()),
        "timestep_id": int(dmd_point["timestep_id"].detach().flatten()[0].cpu().item()),
    }
    with (out / "dmd_stats.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    print(f"[stage3_dmd_tensor_dump] wrote {out}", flush=True)


def _debug_dmd_student_loss(
    real_model,
    fake_probe_model,
    data,
    clean_latents,
    args,
    global_step: int,
    *,
    dataset_load_from_cache: bool,
):
    dmd_weight = float(getattr(args, "stage3_dmd_weight", 0.0))
    if dmd_weight == 0.0 or real_model is None or fake_probe_model is None or clean_latents is None:
        return None, None, None, None
    dmd_every = max(1, int(getattr(args, "stage3_dmd_probe_every", 1)))
    if global_step % dmd_every != 0:
        return None, None, None, None

    with torch.no_grad():
        real_x0, dmd_point = base._stage3c_probe_predict_x0(
            real_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="real_dmd_debug",
            return_dmd_point=True,
        )
        fake_x0 = base._stage3c_probe_predict_x0(
            fake_probe_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="fake_dmd_debug",
            dmd_point=dmd_point,
        )
        z_detached = clean_latents.detach()
        p_real = z_detached.float() - real_x0.float()
        p_fake = z_detached.float() - fake_x0.float()
        reduce_dims = tuple(range(1, p_real.ndim))
        weight_factor = p_real.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        dmd_grad = torch.nan_to_num((p_real - p_fake) / weight_factor)
        dmd_grad_norm = dmd_grad.detach().float().abs().mean()
        dmd_skipped = torch.zeros((), device=clean_latents.device, dtype=torch.float32)
        dmd_loss_clamped = torch.zeros((), device=clean_latents.device, dtype=torch.float32)
        dmd_loss_unweighted = 0.5 * dmd_grad.detach().float().pow(2).mean()

        dump_dir = os.environ.get("FLASHVSR_STAGE3_DMD_TENSOR_DUMP_DIR", "").strip()
        if dump_dir and _dump_step_enabled(global_step):
            _dump_dmd_tensors(
                output_dir=dump_dir,
                global_step=global_step,
                clean_latents=clean_latents,
                real_x0=real_x0,
                fake_x0=fake_x0,
                p_real=p_real,
                p_fake=p_fake,
                dmd_grad=dmd_grad,
                dmd_point=dmd_point,
                weight_factor=weight_factor,
                dmd_loss=dmd_loss_unweighted,
            )

        grad_norm_max = float(getattr(args, "stage3_dmd_grad_norm_max", 0.0) or 0.0)
        spike_policy = str(getattr(args, "stage3_dmd_spike_policy", "none")).lower()
        if grad_norm_max > 0.0 and torch.isfinite(dmd_grad_norm) and float(dmd_grad_norm.item()) > grad_norm_max:
            if spike_policy == "skip":
                dmd_grad = torch.zeros_like(dmd_grad)
                dmd_skipped = torch.ones((), device=clean_latents.device, dtype=torch.float32)
            elif spike_policy == "clamp":
                dmd_grad = dmd_grad * (grad_norm_max / dmd_grad_norm.clamp_min(1e-6))
                dmd_grad_norm = dmd_grad.detach().float().abs().mean()
        dmd_loss_max = float(getattr(args, "stage3_dmd_loss_max", 0.0) or 0.0)
        if dmd_loss_max > 0.0:
            dmd_loss_unweighted = 0.5 * dmd_grad.detach().float().pow(2).mean()
            if torch.isfinite(dmd_loss_unweighted) and float(dmd_loss_unweighted.item()) > dmd_loss_max:
                scale = torch.sqrt(
                    torch.tensor(dmd_loss_max, device=dmd_grad.device, dtype=torch.float32)
                    / dmd_loss_unweighted.clamp_min(1e-12)
                )
                dmd_grad = dmd_grad * scale.to(device=dmd_grad.device, dtype=dmd_grad.dtype)
                dmd_grad_norm = dmd_grad.detach().float().abs().mean()
                dmd_loss_clamped = torch.ones((), device=clean_latents.device, dtype=torch.float32)

    target = (clean_latents.float() - dmd_grad.to(device=clean_latents.device, dtype=torch.float32)).detach()
    dmd_loss = 0.5 * F.mse_loss(clean_latents.float(), target, reduction="mean")
    return dmd_loss * dmd_weight, dmd_grad_norm, dmd_skipped, dmd_loss_clamped


def main() -> None:
    from wanvideo.model_training.flashvsr import train_flashvsr_stage3_v7_d4_4_overfit_lora as overfit_entry

    overfit_entry._install_fixed_sample_seed_patch()
    overfit_entry._install_gt_sharpen_patch()
    base._maybe_run_stage3c_dmd_student_loss = _debug_dmd_student_loss
    print("[stage3_dmd_debug] patched DMD student loss for tensor dump", flush=True)
    base.main()


if __name__ == "__main__":
    main()
