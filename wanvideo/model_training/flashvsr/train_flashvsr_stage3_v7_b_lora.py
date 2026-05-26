import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import accelerate
import torch
import torch.nn.functional as F
import yaml
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.diffusion import ModelLogger, launch_training_task
from wanvideo.model_training.flashvsr import train_flashvsr_stage1_v5_3_lora as v5
from wanvideo.model_training.flashvsr import train_flashvsr_stage2_v6_4_lora as v6


def _stage3_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    tensor_float = tensor.detach().cpu().float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(tensor_float.min()),
        "max": float(tensor_float.max()),
        "mean": float(tensor_float.mean()),
        "std": float(tensor_float.std()),
    }


def _stage3_dump_batch_videos(
    data: Dict[str, Any],
    *,
    output_dir: str,
    max_samples: int,
    fps: int,
) -> None:
    """Dump raw DataLoader batch tensors before pipeline/model preprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    video = data.get("video")
    lq_video = data.get("lq_video")
    sample_seed = data.get("sample_seed")
    payload: Dict[str, Any] = {
        "rank": os.environ.get("RANK", "0"),
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "keys": sorted(list(data.keys())),
        "fps": int(fps),
    }
    if torch.is_tensor(video):
        payload["video"] = _stage3_tensor_stats(video)
    if torch.is_tensor(lq_video):
        payload["lq_video"] = _stage3_tensor_stats(lq_video)
    if sample_seed is not None:
        payload["sample_seed"] = v5._serialize_sample_seed(sample_seed)

    with open(os.path.join(output_dir, "batch_meta.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    if not (torch.is_tensor(video) and torch.is_tensor(lq_video)):
        return
    if video.ndim != 5 or lq_video.ndim != 5:
        raise ValueError(f"Expected video/lq_video [B,T,C,H,W], got video={tuple(video.shape)} lq={tuple(lq_video.shape)}")
    count = min(int(max_samples), int(video.shape[0]), int(lq_video.shape[0]))
    for sample_idx in range(count):
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        gt_tensor = video[sample_idx].detach().cpu().float().clamp(0, 1)
        lq_tensor = lq_video[sample_idx].detach().cpu().float().clamp(0, 1)
        v5.save_video(
            v5._tensor_video_to_pil_frames(gt_tensor),
            os.path.join(sample_dir, "gt_before_model.mp4"),
            fps=fps,
            quality=5,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        v5.save_video(
            v5._tensor_video_to_pil_frames(lq_tensor),
            os.path.join(sample_dir, "lq_before_model.mp4"),
            fps=fps,
            quality=5,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "sample_index": int(sample_idx),
                    "gt": _stage3_tensor_stats(gt_tensor),
                    "lq": _stage3_tensor_stats(lq_tensor),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


def _flatten_video_for_lpips(video: torch.Tensor) -> torch.Tensor:
    """Convert BCHW video tensor from B,C,T,H,W to BT,C,H,W for frame LPIPS."""
    if video.ndim != 5:
        raise ValueError(f"Expected video tensor B,C,T,H,W, got {tuple(video.shape)}")
    return video.permute(0, 2, 1, 3, 4).reshape(-1, video.shape[1], video.shape[3], video.shape[4])


def _weighted_video_mse(pred: torch.Tensor, target: torch.Tensor, *, first_frame_weight: float) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Weighted MSE shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    per_frame = (pred.float() - target.float()).pow(2).mean(dim=(1, 3, 4))
    if per_frame.shape[1] > 0 and float(first_frame_weight) != 1.0:
        weights = torch.ones(per_frame.shape[1], device=per_frame.device, dtype=per_frame.dtype)
        weights[0] = float(first_frame_weight)
        per_frame = per_frame * weights.unsqueeze(0)
        return per_frame.mean()
    return per_frame.mean()


def _weighted_frame_loss(frame_loss: torch.Tensor, *, batch_size: int, num_frames: int, first_frame_weight: float) -> torch.Tensor:
    if frame_loss.numel() != batch_size * num_frames:
        raise ValueError(f"Frame loss size mismatch: got={frame_loss.numel()} expected={batch_size * num_frames}")
    frame_loss = frame_loss.reshape(batch_size, num_frames)
    if num_frames > 0 and float(first_frame_weight) != 1.0:
        weights = torch.ones(num_frames, device=frame_loss.device, dtype=frame_loss.dtype)
        weights[0] = float(first_frame_weight)
        return (frame_loss * weights.unsqueeze(0)).mean()
    return frame_loss.mean()


def _latent_window_to_frame_range(start: int, end: int) -> tuple[int, int]:
    """Map Wan latent-time [start, end) to decoded frame range.

    Wan keeps latent 0 as a special first-frame latent. Every later latent
    expands to 4 frames, so latent 0 -> frame [0, 1), latent 1 -> [1, 5), etc.
    """
    if start < 0 or end <= start:
        raise ValueError(f"Invalid latent window: start={start} end={end}")
    frame_start = 0 if start == 0 else 1 + 4 * (start - 1)
    frame_end = 1 + 4 * (end - 1)
    return frame_start, frame_end


def _sample_stage3_recon_window(latent_t: int, recon_num: int, device: torch.device) -> tuple[int, int]:
    recon_num = min(max(int(recon_num), 1), int(latent_t))
    max_start = int(latent_t) - recon_num
    if max_start <= 0:
        return 0, recon_num
    start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
    return start, start + recon_num


def _build_stage3_decode_window(
    z_pred: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Build a debug/reference full-prefix decoder input.

    v7-B's production path below does not backprop through this whole prefix.
    It advances the Wan decoder cache with prefix latents under no-grad, then
    decodes only [start:end) with grad. This helper remains useful for logging
    the paper-level window semantics.
    """
    if start < 0 or end <= start:
        raise ValueError(f"Invalid decode window: start={start} end={end}")
    if start == 0:
        return z_pred[:, :, :end].contiguous(), 0, 0, end, 0
    z_decode = torch.cat(
        [
            z_pred[:, :, :start].detach(),
            z_pred[:, :, start:end],
        ],
        dim=2,
    ).contiguous()
    frame_start, _ = _latent_window_to_frame_range(start, end)
    return z_decode, frame_start, 0, end, start


def _stage3_unscale_vae_latents(vae, z: torch.Tensor) -> torch.Tensor:
    scale = vae.scale
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
        return z / scale[1].view(1, vae.z_dim, 1, 1, 1) + scale[0].view(1, vae.z_dim, 1, 1, 1)
    scale = scale.to(dtype=z.dtype, device=z.device)
    return z / scale[1] + scale[0]


def _stage3_decode_selected_window_full_frame(pipe, z_pred: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Decode [0:end) causally, but keep gradients only for [start:end).

    This is the Stage3 author-aligned path. Prefix latents advance Wan VAE
    decoder cache under no-grad; selected latents reuse that detached cache and
    carry grad. No spatial tile split is used in the training loss path.
    """
    vae = pipe.vae
    model = vae.model
    model.clear_cache()

    if start > 0:
        with torch.no_grad():
            z_prefix = _stage3_unscale_vae_latents(vae, z_pred[:, :, :start].detach())
            x_prefix = model.conv2(z_prefix)
            for idx in range(start):
                model._conv_idx = [0]
                _, model._feat_map, model._conv_idx = model.decoder(
                    x_prefix[:, :, idx:idx + 1],
                    feat_cache=model._feat_map,
                    feat_idx=model._conv_idx,
                )

    z_selected = _stage3_unscale_vae_latents(vae, z_pred[:, :, start:end])
    x_selected = model.conv2(z_selected)
    outputs = []
    for idx in range(start, end):
        model._conv_idx = [0]
        out, model._feat_map, model._conv_idx = model.decoder(
            x_selected[:, :, idx - start:idx - start + 1],
            feat_cache=model._feat_map,
            feat_idx=model._conv_idx,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=2)


def _stage3_decode_selected_with_checkpoint(
    pipe,
    z_pred: torch.Tensor,
    start: int,
    end: int,
    *,
    cpu_offload: bool,
) -> torch.Tensor:
    def _decode_selected(latents: torch.Tensor) -> torch.Tensor:
        return _stage3_decode_selected_window_full_frame(pipe, latents, start, end)

    if not z_pred.requires_grad:
        return _decode_selected(z_pred)
    if cpu_offload:
        with torch.autograd.graph.save_on_cpu():
            return checkpoint(_decode_selected, z_pred, use_reentrant=False)
    return checkpoint(_decode_selected, z_pred, use_reentrant=False)


class _LazyLPIPS(torch.nn.Module):
    def __init__(self, net: str = "vgg"):
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as error:
            raise RuntimeError(
                "Stage3 v7-B requires the `lpips` package when stage3_lpips_weight > 0. "
                "Install lpips or set --stage3_lpips_weight 0 for a dry smoke."
            ) from error
        self.loss = lpips.LPIPS(net=net)
        self.loss.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred.float(), target.float()).reshape(-1)


def _get_stage3_lpips(pipe, net: str):
    # Do not assign this module through nn.Module.__setattr__ on `pipe`.
    # Stage3 creates LPIPS after DeepSpeed initialization; registering it as a
    # new frozen child module makes DeepSpeed checkpointing fail because the
    # parameter-name mapping was built before LPIPS existed.
    if hasattr(pipe, "_modules") and "_stage3_lpips" in pipe._modules:
        del pipe._modules["_stage3_lpips"]
    cache = pipe.__dict__.get("_stage3_lpips_cache")
    cache_net = pipe.__dict__.get("_stage3_lpips_cache_net")
    if cache is None or cache_net != net:
        cache = _LazyLPIPS(net=net).to(device=pipe.device)
        cache.eval()
        object.__setattr__(pipe, "_stage3_lpips_cache", cache)
        object.__setattr__(pipe, "_stage3_lpips_cache_net", net)
    return cache


def _stage3_lpips_video_loss(
    lpips_module: torch.nn.Module,
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    *,
    first_frame_weight: float,
    cpu_offload: bool,
) -> torch.Tensor:
    """Frame-wise LPIPS with checkpointing to keep Stage3 peak memory bounded."""
    if pred_video.shape != target_video.shape:
        raise ValueError(f"LPIPS video shape mismatch: pred={tuple(pred_video.shape)} gt={tuple(target_video.shape)}")
    num_frames = int(pred_video.shape[2])
    total = torch.zeros((), device=pred_video.device, dtype=torch.float32)
    weight_total = 0.0

    def _lpips_one(pred_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
        return lpips_module(pred_frame, target_frame).mean()

    for frame_idx in range(num_frames):
        pred_frame = pred_video[:, :, frame_idx].contiguous()
        target_frame = target_video[:, :, frame_idx].contiguous()
        if pred_frame.requires_grad:
            if cpu_offload:
                with torch.autograd.graph.save_on_cpu():
                    frame_loss = checkpoint(_lpips_one, pred_frame, target_frame, use_reentrant=False)
            else:
                frame_loss = checkpoint(_lpips_one, pred_frame, target_frame, use_reentrant=False)
        else:
            frame_loss = _lpips_one(pred_frame, target_frame)
        frame_weight = float(first_frame_weight) if frame_idx == 0 else 1.0
        total = total + frame_loss * frame_weight
        weight_total += frame_weight
    return total / max(weight_total, 1.0)


def Stage3BOneStepReconLoss(pipe, *, stage3_recon_num_latents: int, stage3_flow_weight: float,
                            stage3_mse_weight: float, stage3_lpips_weight: float,
                            stage3_lpips_net: str, stage3_first_frame_pixel_weight: float,
                            stage3_first_frame_lpips_weight: float,
                            stage3_decoder_cpu_offload: bool,
                            stage3_compute_z_pred: bool,
                            stage3_fake_fm_weight: float,
                            stage3_fake_update_ratio: int,
                            stage3_fake_checkpoint: Optional[str],
                            **inputs):
    if float(stage3_fake_fm_weight) != 0.0:
        raise RuntimeError(
            "Stage3 v7-B fake-FM is intentionally guarded off in this first "
            "split-out file. DMD2-style G_fake needs a separate trainable model "
            "and optimizer/update schedule; the current launch_training_task runner "
            "only owns one optimizer. Keep stage3_fake_fm_weight=0 for the v7-B "
            "scaffold, or implement the dedicated dual-optimizer runner before "
            "enabling this loss."
        )
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    target_latents = inputs["input_latents"]
    noise = torch.randn_like(target_latents)
    inputs["latents"] = pipe.scheduler.add_noise(target_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(target_latents, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    flow_loss = F.mse_loss(noise_pred.float(), training_target.float())
    flow_loss = flow_loss * pipe.scheduler.training_weight(timestep)

    need_reconstruction = float(stage3_mse_weight) != 0.0 or float(stage3_lpips_weight) != 0.0
    if bool(stage3_compute_z_pred) or need_reconstruction:
        # One-step prediction to sigma=0. This is the v7-B student output z_pred.
        z_pred = pipe.scheduler.step(noise_pred, timestep, inputs["latents"], to_final=True)
    else:
        z_pred = None

    if need_reconstruction:
        latent_t = int(z_pred.shape[2])
        recon_start, recon_end = _sample_stage3_recon_window(latent_t, int(stage3_recon_num_latents), z_pred.device)
        recon_num = recon_end - recon_start
        frame_start, frame_end = _latent_window_to_frame_range(recon_start, recon_end)
        _, _, context_latent_start, context_latent_end, detached_context_latents = _build_stage3_decode_window(
            z_pred,
            recon_start,
            recon_end,
        )
        local_frame_start = 0

        pipe.load_models_to_device(["vae"])
        x_pred = _stage3_decode_selected_with_checkpoint(
            pipe,
            z_pred,
            recon_start,
            recon_end,
            cpu_offload=bool(stage3_decoder_cpu_offload),
        ).to(device=pipe.device)
        x_pred = x_pred.to(device=pipe.device).contiguous()
        target_frames = int(x_pred.shape[2])
        local_frame_end = target_frames
        x_gt = pipe.preprocess_video(inputs["input_video"]).to(device=pipe.device, dtype=x_pred.dtype)
        x_gt = x_gt[:, :, frame_start:frame_end].contiguous()
        if x_gt.shape != x_pred.shape:
            raise ValueError(f"Stage3 v7-B decoded/GT mismatch: pred={tuple(x_pred.shape)} gt={tuple(x_gt.shape)}")

        effective_first_frame_pixel_weight = float(stage3_first_frame_pixel_weight) if frame_start == 0 else 1.0
        effective_first_frame_lpips_weight = float(stage3_first_frame_lpips_weight) if frame_start == 0 else 1.0
        mse_loss = _weighted_video_mse(
            x_pred,
            x_gt,
            first_frame_weight=effective_first_frame_pixel_weight,
        )
        if stage3_lpips_weight > 0:
            lpips_module = _get_stage3_lpips(pipe, stage3_lpips_net)
            lpips_loss = _stage3_lpips_video_loss(
                lpips_module,
                x_pred,
                x_gt,
                first_frame_weight=effective_first_frame_lpips_weight,
                cpu_offload=bool(stage3_decoder_cpu_offload),
            )
        else:
            lpips_loss = torch.zeros((), device=pipe.device, dtype=torch.float32)
    else:
        recon_start = recon_end = recon_num = 0
        frame_start = frame_end = 0
        context_latent_start = context_latent_end = detached_context_latents = 0
        local_frame_start = local_frame_end = target_frames = 0
        effective_first_frame_pixel_weight = 1.0
        effective_first_frame_lpips_weight = 1.0
        mse_loss = torch.zeros((), device=pipe.device, dtype=torch.float32)
        lpips_loss = torch.zeros((), device=pipe.device, dtype=torch.float32)

    total = (
        float(stage3_flow_weight) * flow_loss
        + float(stage3_mse_weight) * mse_loss
        + float(stage3_lpips_weight) * lpips_loss
    )
    if not hasattr(pipe, "_stage3_last_losses"):
        pipe._stage3_last_losses = {}
    pipe._stage3_last_losses = {
        "loss": float(total.detach().cpu()),
        "loss_flow": float(flow_loss.detach().cpu()),
        "loss_mse": float(mse_loss.detach().cpu()),
        "loss_lpips": float(lpips_loss.detach().cpu()),
        "loss_fake_fm": 0.0,
        "recon_latents": int(recon_num),
        "selected_latent_start": int(recon_start),
        "selected_latent_end": int(recon_end),
        "decoded_frame_start": int(frame_start),
        "decoded_frame_end": int(frame_end),
        "decoded_frames": int(target_frames),
        "decode_context_latent_start": int(context_latent_start),
        "decode_context_latent_end": int(context_latent_end),
        "decode_local_frame_start": int(local_frame_start),
        "decode_local_frame_end": int(local_frame_end),
        "decode_context_mode": "full_prefix",
        "decoder_cpu_offload": bool(stage3_decoder_cpu_offload),
        "detached_context_latents": int(detached_context_latents),
        "first_frame_pixel_weight": float(effective_first_frame_pixel_weight),
        "first_frame_lpips_weight": float(effective_first_frame_lpips_weight),
        "fake_fm_weight": float(stage3_fake_fm_weight),
        "fake_update_ratio": int(stage3_fake_update_ratio),
        "fake_checkpoint": stage3_fake_checkpoint,
        "compute_z_pred": bool(stage3_compute_z_pred),
        "need_reconstruction": bool(need_reconstruction),
    }
    if os.environ.get("FLASHVSR_STAGE3_DEBUG_LOSS") == "1" and os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            "[stage3_v7_b_loss] "
            f"loss={pipe._stage3_last_losses['loss']:.6f} "
            f"flow={pipe._stage3_last_losses['loss_flow']:.6f} "
            f"mse={pipe._stage3_last_losses['loss_mse']:.6f} "
            f"lpips={pipe._stage3_last_losses['loss_lpips']:.6f} "
            f"latent_window=[{recon_start},{recon_end}) "
            f"frame_window=[{frame_start},{frame_end}) "
            f"decode_latents=[{context_latent_start},{context_latent_end}) "
            f"local_frame_window=[{local_frame_start},{local_frame_end}) "
            f"recon_latents={recon_num} decoded_frames={target_frames} "
            "context_mode=full_prefix "
            f"decoder_cpu_offload={bool(stage3_decoder_cpu_offload)} "
            f"detached_context_latents={int(detached_context_latents)} "
            f"first_frame_pixel_weight={effective_first_frame_pixel_weight} "
            f"first_frame_lpips_weight={effective_first_frame_lpips_weight} "
            f"compute_z_pred={bool(stage3_compute_z_pred)} "
            f"need_reconstruction={bool(need_reconstruction)}",
            flush=True,
        )
    return total


class FlashVSRStage3BTrainingModule(v6.FlashVSRStage2TrainingModule):
    def __init__(
        self,
        *args,
        stage3_recon_num_latents: int = 2,
        stage3_flow_weight: float = 1.0,
        stage3_mse_weight: float = 1.0,
        stage3_lpips_weight: float = 2.0,
        stage3_lpips_net: str = "vgg",
        stage3_first_frame_pixel_weight: float = 4.0,
        stage3_first_frame_lpips_weight: float = 4.0,
        stage3_decoder_cpu_offload: bool = True,
        stage3_compute_z_pred: bool = True,
        stage3_fake_fm_weight: float = 0.0,
        stage3_fake_update_ratio: int = 5,
        stage3_fake_checkpoint: Optional[str] = None,
        debug_dump_training_batch_dir: Optional[str] = None,
        debug_dump_training_batch_limit: int = 0,
        debug_dump_training_batch_fps: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stage3_recon_num_latents = int(stage3_recon_num_latents)
        self.stage3_flow_weight = float(stage3_flow_weight)
        self.stage3_mse_weight = float(stage3_mse_weight)
        self.stage3_lpips_weight = float(stage3_lpips_weight)
        self.stage3_lpips_net = str(stage3_lpips_net)
        self.stage3_first_frame_pixel_weight = float(stage3_first_frame_pixel_weight)
        self.stage3_first_frame_lpips_weight = float(stage3_first_frame_lpips_weight)
        self.stage3_decoder_cpu_offload = bool(stage3_decoder_cpu_offload)
        self.stage3_compute_z_pred = bool(stage3_compute_z_pred)
        self.stage3_fake_fm_weight = float(stage3_fake_fm_weight)
        self.stage3_fake_update_ratio = int(stage3_fake_update_ratio)
        self.stage3_fake_checkpoint = stage3_fake_checkpoint
        self.debug_dump_training_batch_dir = debug_dump_training_batch_dir
        self.debug_dump_training_batch_limit = int(debug_dump_training_batch_limit)
        self.debug_dump_training_batch_fps = int(debug_dump_training_batch_fps)
        self._debug_dump_training_batch_done = False

    def forward(self, data, inputs=None):
        if (
            not self._debug_dump_training_batch_done
            and self.debug_dump_training_batch_dir
            and self.debug_dump_training_batch_limit > 0
            and os.environ.get("RANK", "0") == "0"
        ):
            _stage3_dump_batch_videos(
                data,
                output_dir=self.debug_dump_training_batch_dir,
                max_samples=self.debug_dump_training_batch_limit,
                fps=self.debug_dump_training_batch_fps,
            )
            self._debug_dump_training_batch_done = True
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        merged_inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            merged_inputs = self.pipe.unit_runner(unit, self.pipe, *merged_inputs)
        return Stage3BOneStepReconLoss(
            self.pipe,
            **merged_inputs[0],
            **merged_inputs[1],
            stage3_recon_num_latents=self.stage3_recon_num_latents,
            stage3_flow_weight=self.stage3_flow_weight,
            stage3_mse_weight=self.stage3_mse_weight,
            stage3_lpips_weight=self.stage3_lpips_weight,
            stage3_lpips_net=self.stage3_lpips_net,
            stage3_first_frame_pixel_weight=self.stage3_first_frame_pixel_weight,
            stage3_first_frame_lpips_weight=self.stage3_first_frame_lpips_weight,
            stage3_decoder_cpu_offload=self.stage3_decoder_cpu_offload,
            stage3_compute_z_pred=self.stage3_compute_z_pred,
            stage3_fake_fm_weight=self.stage3_fake_fm_weight,
            stage3_fake_update_ratio=self.stage3_fake_update_ratio,
            stage3_fake_checkpoint=self.stage3_fake_checkpoint,
        )


class FlashVSRStage3BValidationCallback(v6.FlashVSRStage2ValidationCallback):
    def __call__(self, accelerator, model, checkpoint_path: str, step: int):
        if not self.validation_samples:
            return
        validation_dir = os.path.join(self.output_path, "validation", f"step-{step}")
        os.makedirs(validation_dir, exist_ok=True)

        pipe = model.pipe
        scheduler_state = {
            "timesteps": pipe.scheduler.timesteps.clone() if getattr(pipe.scheduler, "timesteps", None) is not None else None,
            "training": getattr(pipe.scheduler, "training", None),
        }
        training_mode = model.training
        model.eval()
        try:
            for sample_index, sample in enumerate(self.validation_samples):
                sample_dir = os.path.join(validation_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                hr_tensor = sample["video"]
                lq_tensor = sample["lq_video"]
                hr_frames = v5._tensor_video_to_pil_frames(hr_tensor)
                lq_frames = v5._tensor_video_to_pil_frames(lq_tensor)
                v5.save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                v5.save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

                pipe.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
                inputs_shared = {
                    "input_video": None,
                    "lq_video": lq_tensor.unsqueeze(0),
                    "seed": self.seed_base + sample_index,
                    "rand_device": "cpu",
                    "height": int(hr_tensor.shape[-2]),
                    "width": int(hr_tensor.shape[-1]),
                    "num_frames": int(hr_tensor.shape[0]),
                    "cfg_scale": 1.0,
                    "cfg_merge": False,
                    "tiled": True,
                    "tile_size": (30, 52),
                    "tile_stride": (15, 26),
                    "framewise_decoding": False,
                    "lq_proj_scale": pipe.lq_proj_scale,
                }
                for unit in pipe.units:
                    inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, {}, {})
                pipe.load_models_to_device(pipe.in_iteration_models)
                models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
                timestep = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                noise_pred = pipe.model_fn(**models, **inputs_shared, timestep=timestep)
                latents = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[0], inputs_shared["latents"], to_final=True)
                pipe.load_models_to_device(["vae"])
                video = pipe.vae.decode(
                    latents,
                    device=pipe.device,
                    tiled=True,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
                sr_frames = pipe.vae_output_to_video(video)
                v5.save_video(sr_frames, os.path.join(sample_dir, "sr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "checkpoint_path": checkpoint_path,
                            "step": int(step),
                            "sample_index": int(sample_index),
                            "validation_mode": "stage3_v7_b_one_step_recon",
                            "input_num_frames": int(hr_tensor.shape[0]),
                            "output_num_frames": len(sr_frames),
                            "sample_seed": v5._serialize_sample_seed(sample.get("sample_seed")),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
        finally:
            model.train(training_mode)
            if scheduler_state["training"]:
                pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
            else:
                if scheduler_state["timesteps"] is not None:
                    pipe.scheduler.timesteps = scheduler_state["timesteps"]
                if scheduler_state["training"] is not None:
                    pipe.scheduler.training = scheduler_state["training"]


def _stage3_parser():
    parser = v6._stage2_parser()
    parser.add_argument(
        "--resume_stage2_checkpoint",
        type=str,
        default=None,
        help="Alias for --resume_stage1_checkpoint in Stage3; should point to a Stage2 sparse-causal checkpoint.",
    )
    parser.add_argument("--stage3_recon_num_latents", type=int, default=2)
    parser.add_argument("--stage3_flow_weight", type=float, default=1.0)
    parser.add_argument("--stage3_mse_weight", type=float, default=1.0)
    parser.add_argument("--stage3_lpips_weight", type=float, default=2.0)
    parser.add_argument("--stage3_lpips_net", type=str, default="vgg", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--stage3_first_frame_pixel_weight", type=float, default=4.0)
    parser.add_argument("--stage3_first_frame_lpips_weight", type=float, default=4.0)
    parser.add_argument("--stage3_decoder_cpu_offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage3_compute_z_pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug_dump_training_batch_dir",
        type=str,
        default=None,
        help="Optional rank0-only dump directory for raw DataLoader video/lq_video before model preprocessing.",
    )
    parser.add_argument("--debug_dump_training_batch_limit", type=int, default=0)
    parser.add_argument("--debug_dump_training_batch_fps", type=int, default=8)
    parser.add_argument(
        "--stage3_fake_checkpoint",
        type=str,
        default=None,
        help="Reserved for Stage3 v7-B G_fake. The current scaffold keeps fake-FM disabled until the dual-optimizer runner is added.",
    )
    parser.add_argument(
        "--stage3_fake_fm_weight",
        type=float,
        default=0.0,
        help="Must stay 0 in the v7-B scaffold. Non-zero requires a dedicated G_fake optimizer path.",
    )
    parser.add_argument("--stage3_fake_update_ratio", type=int, default=5)
    return parser


def parse_stage3_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = _stage3_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**v5._flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.resume_stage2_checkpoint is not None:
        if args.resume_stage1_checkpoint is not None and args.resume_stage1_checkpoint != args.resume_stage2_checkpoint:
            parser.error("--resume_stage1_checkpoint and --resume_stage2_checkpoint point to different files.")
        args.resume_stage1_checkpoint = args.resume_stage2_checkpoint
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.image_tar_url is not None:
        args.picked17k_image_tar_url = args.image_tar_url
    else:
        args.image_tar_url = args.picked17k_image_tar_url
    if args.num_frames % 8 != 1:
        parser.error("Stage3 v7-B currently follows Stage2 89->85 streaming shape, requiring num_frames % 8 == 1.")
    return args


@record
def main(argv=None):
    def _excepthook(exc_type, exc_value, exc_traceback):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[fatal rank={rank} local_rank={local_rank}] {exc_type.__name__}: {exc_value}", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _excepthook
    args = parse_stage3_args(argv)
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    }
    data_loader_config_cls = getattr(accelerate, "DataLoaderConfiguration", None)
    if data_loader_config_cls is not None:
        accelerator_kwargs["dataloader_config"] = data_loader_config_cls(
            dispatch_batches=False,
            split_batches=False,
            even_batches=False,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    v5.configure_deepspeed_runtime(accelerator, args)
    if accelerator.is_main_process:
        v5.dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)

    dataset = v6.FlashVSRStage2VideoOnlyDataset(
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
    model = FlashVSRStage3BTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lq_proj_checkpoint=args.lq_proj_checkpoint,
        resume_stage1_checkpoint=args.resume_stage1_checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        lq_proj_scale=args.lq_proj_scale,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        freeze_lq_proj_in=args.freeze_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        stage2_attention_mode=args.stage2_attention_mode,
        stage2_topk_ratio=args.stage2_topk_ratio,
        stage2_local_num=args.stage2_local_num,
        stage3_recon_num_latents=args.stage3_recon_num_latents,
        stage3_flow_weight=args.stage3_flow_weight,
        stage3_mse_weight=args.stage3_mse_weight,
        stage3_lpips_weight=args.stage3_lpips_weight,
        stage3_lpips_net=args.stage3_lpips_net,
        stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
        stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
        stage3_decoder_cpu_offload=args.stage3_decoder_cpu_offload,
        stage3_compute_z_pred=args.stage3_compute_z_pred,
        stage3_fake_fm_weight=args.stage3_fake_fm_weight,
        stage3_fake_update_ratio=args.stage3_fake_update_ratio,
        stage3_fake_checkpoint=args.stage3_fake_checkpoint,
        debug_dump_training_batch_dir=args.debug_dump_training_batch_dir,
        debug_dump_training_batch_limit=args.debug_dump_training_batch_limit,
        debug_dump_training_batch_fps=args.debug_dump_training_batch_fps,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    if accelerator.is_local_main_process:
        trainable_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
        print("Stage3 v7-B: one-step student + G_fake scaffold + Wan decoder reconstruction smoke path", flush=True)
        print(f"stage3_recon_num_latents={args.stage3_recon_num_latents}", flush=True)
        print(f"stage3_loss_weights flow={args.stage3_flow_weight} mse={args.stage3_mse_weight} lpips={args.stage3_lpips_weight}", flush=True)
        print(
            "stage3_fake "
            f"fm_weight={args.stage3_fake_fm_weight} "
            f"update_ratio={args.stage3_fake_update_ratio} "
            f"checkpoint={args.stage3_fake_checkpoint}",
            flush=True,
        )
        print(
            "stage3_first_frame_weights "
            f"pixel={args.stage3_first_frame_pixel_weight} lpips={args.stage3_first_frame_lpips_weight}",
            flush=True,
        )
        print(
            "stage3_decode "
            "context_mode=full_prefix "
            f"decoder_cpu_offload={args.stage3_decoder_cpu_offload}",
            flush=True,
        )
        print(f"stage3_compute_z_pred={args.stage3_compute_z_pred}", flush=True)
        if args.debug_dump_training_batch_dir and args.debug_dump_training_batch_limit > 0:
            print(
                "debug_dump_training_batch "
                f"dir={args.debug_dump_training_batch_dir} "
                f"limit={args.debug_dump_training_batch_limit} "
                f"fps={args.debug_dump_training_batch_fps}",
                flush=True,
            )
        print(f"Stage2 attention mode inherited for student: {args.stage2_attention_mode}", flush=True)
        print(f"Trainable parameter tensors: {len(trainable_params)}", flush=True)
        print(f"Trainable parameter count: {sum(numel for _, numel in trainable_params)}", flush=True)

    validation_callback = None
    if args.validation_num_samples > 0 and accelerator.is_main_process:
        print("Preparing fixed Stage3 v7-B validation samples...", flush=True)
        validation_samples = v5.collect_fixed_validation_samples(dataset, args.validation_num_samples)
        print(f"Prepared {len(validation_samples)} fixed Stage3 v7-B validation samples.", flush=True)
        validation_callback = FlashVSRStage3BValidationCallback(
            output_path=args.output_path,
            validation_samples=validation_samples,
            num_inference_steps=args.validation_num_inference_steps,
            fps=args.validation_fps,
            seed_base=(args.global_seed if args.global_seed is not None else 20260513),
            use_wandb=args.use_wandb,
        )
    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=v5.flashvsr_stage1_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
