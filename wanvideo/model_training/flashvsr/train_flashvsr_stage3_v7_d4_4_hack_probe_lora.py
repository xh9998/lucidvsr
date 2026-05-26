"""Stage3 v7-D4.4 hack-probe entrypoint.

This file is intentionally a runtime wrapper around the isolated overfit
entrypoint.  It does not modify the production v7-D4.4 training file.  The
goal is diagnostic: isolate why DMD-only updates drift yellow/green or gray.

Enable one variant with:

    FLASHVSR_STAGE3_HACK_PROBE_VARIANT=fake_x0_equal_real

Supported variants:
  - fake_x0_equal_real
  - dmd_grad_scale0p1_clipnear
  - color_match_fake_x0_to_real
  - fake_score_percentile_clip
  - weight_factor_rms_detach
  - freeze_fake_lq_proj
  - fake_lr0p1
  - fake_lr0p01
"""

from __future__ import annotations

import os


def _install_fixed_sample_seed_patch() -> None:
    fixed_seed = os.environ.get("FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED")
    if fixed_seed in (None, ""):
        return

    from wanvideo.data.flashvsr.datasets import streaming_dataset

    seed = int(fixed_seed)

    def _fixed_next_sample_seed(self, rng):  # noqa: ARG001
        return seed

    streaming_dataset.FlashVSRStreamingDataset._next_sample_seed = _fixed_next_sample_seed
    print(f"[stage3_hack_probe] fixed degradation sample_seed={seed}", flush=True)


def _install_gt_sharpen_patch() -> None:
    enabled = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN", "").lower() in ("1", "true", "yes", "y")
    if not enabled:
        return

    from wanvideo.data.flashvsr.datasets import streaming_dataset
    from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53_usmgt import ConsistentClipGTSharpen

    backend = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN_BACKEND", "torch")
    device = os.environ.get("FLASHVSR_OVERFIT_GT_SHARPEN_DEVICE", "auto")
    sharpener = ConsistentClipGTSharpen(backend=backend, device=device)

    def _process_video_bytes_with_sharpen(self, video_bytes, sample_id, rng):
        frames = self._extract_frames(video_bytes)
        if frames is None:
            return None
        clip = self._select_clip(frames, rng=rng)
        if clip is None:
            return None
        sample_seed = self._next_sample_seed(rng)
        gt_clip = sharpener.sharpen_batch(clip)
        lq_video = self._build_lq_clip(gt_clip, rng=rng, sample_seed=sample_seed)
        return self._maybe_convert_output(
            {
                "video": gt_clip,
                "lq_video": lq_video,
                "sample_id": sample_id,
                "source_type": "video",
                "sample_seed": sample_seed,
            }
        )

    streaming_dataset.FlashVSRStreamingDataset._process_video_bytes = _process_video_bytes_with_sharpen
    print(
        "[stage3_hack_probe] enabled Stage1-USMGT-style GT sharpening "
        f"backend={backend} device={device}",
        flush=True,
    )


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0") or 0) == 0


def _stats_text(stage3, name, tensor) -> str:
    if tensor is None:
        return f"{name}=None"
    with stage3.torch.no_grad():
        x = tensor.detach().float()
        return (
            f"{name}.mean={float(x.mean().item()):.6g} "
            f"{name}.std={float(x.std().item()):.6g} "
            f"{name}.min={float(x.min().item()):.6g} "
            f"{name}.max={float(x.max().item()):.6g}"
        )


def _match_mean_std_per_sample(stage3, source, reference, eps: float = 1e-6):
    reduce_dims = tuple(range(1, source.ndim))
    src_mean = source.float().mean(dim=reduce_dims, keepdim=True)
    src_std = source.float().std(dim=reduce_dims, keepdim=True).clamp_min(eps)
    ref_mean = reference.float().mean(dim=reduce_dims, keepdim=True)
    ref_std = reference.float().std(dim=reduce_dims, keepdim=True).clamp_min(eps)
    matched = (source.float() - src_mean) / src_std * ref_std + ref_mean
    return matched.to(device=source.device, dtype=source.dtype)


def _clip_per_sample_quantile(stage3, tensor, percentile: float):
    """Winsorize a score tensor by per-sample absolute-value percentile."""
    if percentile <= 0 or percentile >= 1:
        return tensor
    flat = tensor.detach().float().abs().flatten(start_dim=1)
    threshold = stage3.torch.quantile(flat, percentile, dim=1, keepdim=True)
    view_shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    threshold = threshold.reshape(view_shape).clamp_min(1e-6)
    return tensor.clamp(min=-threshold, max=threshold)


def _install_hack_probe_patch(stage3) -> None:
    variant = os.environ.get("FLASHVSR_STAGE3_HACK_PROBE_VARIANT", "none").strip().lower()
    if variant in ("", "none", "off", "0"):
        print("[stage3_hack_probe] variant=none; delegating to normal overfit entrypoint", flush=True)
        return

    supported = {
        "fake_x0_equal_real",
        "dmd_grad_scale0p1_clipnear",
        "color_match_fake_x0_to_real",
        "fake_score_percentile_clip",
        "weight_factor_rms_detach",
        "freeze_fake_lq_proj",
        "fake_lr0p1",
        "fake_lr0p01",
    }
    if variant not in supported:
        raise ValueError(f"Unsupported FLASHVSR_STAGE3_HACK_PROBE_VARIANT={variant!r}; supported={sorted(supported)}")

    original_dmd_loss = stage3._maybe_run_stage3c_dmd_student_loss
    log_every = max(1, int(os.environ.get("FLASHVSR_STAGE3_HACK_PROBE_LOG_EVERY", "1")))
    grad_scale = float(os.environ.get("FLASHVSR_STAGE3_HACK_PROBE_DMD_GRAD_SCALE", "0.1"))
    grad_absmax = float(os.environ.get("FLASHVSR_STAGE3_HACK_PROBE_DMD_GRAD_ABSMAX", "0.25"))
    score_clip_percentile = float(os.environ.get("FLASHVSR_STAGE3_HACK_PROBE_SCORE_CLIP_PERCENTILE", "0.995"))

    if _rank0():
        expected = {
            "fake_x0_equal_real": (
                "Sets fake_x0 = real_x0 before DMD grad. If yellow/gray drift disappears, "
                "the damaging path is DMD real/fake score difference, not flow/recon or validation."
            ),
            "dmd_grad_scale0p1_clipnear": (
                "Keeps current real/fake scores but scales DMD grad by 0.1 and clips elementwise. "
                "If this mitigates drift, DMD direction may be roughly usable but update magnitude/spikes are unsafe."
            ),
            "color_match_fake_x0_to_real": (
                "Matches fake_x0 per-sample mean/std to real_x0 before DMD grad. If this mitigates yellow/green drift, "
                "global color/scale mismatch between real/fake score predictions is likely a primary cause."
            ),
            "fake_score_percentile_clip": (
                "Clips the real/fake score difference by per-sample percentile before normalization. "
                "If this mitigates drift, DMD is being dominated by local score outliers."
            ),
            "weight_factor_rms_detach": (
                "Uses a detached per-sample RMS weight factor instead of absmean. "
                "If this mitigates drift, DMD normalization scale is too small/unstable for video latents."
            ),
            "freeze_fake_lq_proj": (
                "Uses the normal DMD formula while the config freezes fake lq_proj_in. "
                "If this mitigates drift, fake-side LQ projector updates are corrupting the critic condition path."
            ),
            "fake_lr0p1": (
                "Uses the normal DMD formula with G_fake learning rate reduced 10x in config. "
                "If this mitigates drift, fake critic update scale is a primary issue."
            ),
            "fake_lr0p01": (
                "Uses the normal DMD formula with G_fake learning rate reduced 100x in config. "
                "If this mitigates drift, fake critic update scale is a primary issue."
            ),
        }[variant]
        print(f"[stage3_hack_probe] installed variant={variant}", flush=True)
        print(f"[stage3_hack_probe] expected_if_mitigates={expected}", flush=True)

    def _hack_dmd_student_loss(
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

        with stage3.torch.no_grad():
            real_x0, dmd_point = stage3._stage3c_probe_predict_x0(
                real_model,
                data,
                clean_latents,
                dataset_load_from_cache=dataset_load_from_cache,
                probe_name="real_dmd_hack_probe",
                return_dmd_point=True,
            )
            fake_x0 = stage3._stage3c_probe_predict_x0(
                fake_probe_model,
                data,
                clean_latents,
                dataset_load_from_cache=dataset_load_from_cache,
                probe_name="fake_dmd_hack_probe",
                dmd_point=dmd_point,
            )

            original_fake_x0 = fake_x0
            if variant == "fake_x0_equal_real":
                fake_x0 = real_x0.detach().clone()
                modified = "fake_x0 := real_x0"
            elif variant == "color_match_fake_x0_to_real":
                fake_x0 = _match_mean_std_per_sample(stage3, fake_x0, real_x0)
                modified = "fake_x0 mean/std matched to real_x0 per sample"
            else:
                modified = "DMD grad scaled/clipped after normal real/fake score"

            z_detached = clean_latents.detach()
            p_real = z_detached.float() - real_x0.float()
            p_fake = z_detached.float() - fake_x0.float()
            reduce_dims = tuple(range(1, p_real.ndim))
            score_diff = p_real - p_fake
            if variant == "fake_score_percentile_clip":
                score_diff = _clip_per_sample_quantile(stage3, score_diff, score_clip_percentile)
                modified = f"score_diff clipped by per-sample p{score_clip_percentile:g}"
            if variant == "weight_factor_rms_detach":
                weight_factor = p_real.detach().float().pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(1e-6)
                modified = "weight_factor := detached per-sample RMS(p_real)"
            else:
                weight_factor = p_real.detach().abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            dmd_grad = stage3.torch.nan_to_num(score_diff / weight_factor)

            if variant == "dmd_grad_scale0p1_clipnear":
                dmd_grad = dmd_grad * grad_scale
                if grad_absmax > 0:
                    dmd_grad = dmd_grad.clamp(min=-grad_absmax, max=grad_absmax)

            dmd_grad_norm = dmd_grad.detach().float().abs().mean()
            dmd_skipped = stage3.torch.zeros((), device=clean_latents.device, dtype=stage3.torch.float32)
            dmd_loss_clamped = stage3.torch.zeros((), device=clean_latents.device, dtype=stage3.torch.float32)

        target = (clean_latents.float() - dmd_grad.to(device=clean_latents.device, dtype=stage3.torch.float32)).detach()
        dmd_loss = 0.5 * stage3.F.mse_loss(clean_latents.float(), target, reduction="mean")

        if _rank0() and (int(global_step) % log_every == 0):
            with stage3.torch.no_grad():
                real_fake_mse = stage3.F.mse_loss(real_x0.float(), fake_x0.float(), reduction="mean")
                real_origfake_mse = stage3.F.mse_loss(real_x0.float(), original_fake_x0.float(), reduction="mean")
                print(
                    "[stage3_hack_probe_dmd] "
                    f"step={int(global_step)} variant={variant} modified='{modified}' "
                    f"dmd_loss_unweighted={float(dmd_loss.detach().float().item()):.6g} "
                    f"dmd_grad_absmean={float(dmd_grad_norm.detach().float().item()):.6g} "
                    f"dmd_grad_absmax={float(dmd_grad.detach().float().abs().max().item()):.6g} "
                    f"real_fake_mse_after={float(real_fake_mse.detach().float().item()):.6g} "
                    f"real_fake_mse_before={float(real_origfake_mse.detach().float().item()):.6g} "
                    f"{_stats_text(stage3, 'real_x0', real_x0)} "
                    f"{_stats_text(stage3, 'fake_x0_orig', original_fake_x0)} "
                    f"{_stats_text(stage3, 'fake_x0_used', fake_x0)}",
                    flush=True,
                )

        return dmd_loss * dmd_weight, dmd_grad_norm, dmd_skipped, dmd_loss_clamped

    stage3._maybe_run_stage3c_dmd_student_loss = _hack_dmd_student_loss
    stage3._stage3_hack_probe_original_dmd_loss = original_dmd_loss


def main() -> None:
    _install_fixed_sample_seed_patch()
    _install_gt_sharpen_patch()
    from wanvideo.model_training.flashvsr import train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora as stage3_main_module

    _install_hack_probe_patch(stage3_main_module)
    print("[stage3_hack_probe] using isolated train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py", flush=True)
    stage3_main_module.main()


if __name__ == "__main__":
    main()
