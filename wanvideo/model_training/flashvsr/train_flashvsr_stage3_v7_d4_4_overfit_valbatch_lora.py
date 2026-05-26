import argparse
import os
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import accelerate
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate.utils import DeepSpeedPlugin, send_to_device
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.diffusion import ModelLogger
from diffsynth.diffusion.runner import (
    _DATALOADER_SUPPORTS_IN_ORDER,
    _PreBatchedIterableDataset,
    _first_item_collate,
    _init_data_worker_no_cuda,
    initialize_deepspeed_gradient_checkpointing,
    load_training_state,
    save_training_state,
)
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


def _stage3_timing_sync(device: torch.device):
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _stage3_timing_now(device: torch.device) -> float:
    _stage3_timing_sync(device)
    return time.perf_counter()


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
    forced_start = os.environ.get("FLASHVSR_STAGE3_FORCE_RECON_START")
    if forced_start not in (None, ""):
        start = max(0, min(int(forced_start), max_start))
        return start, start + recon_num
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
    # v7-C reads this immediately after the student forward to build DMD probes.
    # Keep the tensor live only for the current step; the runner clears it after
    # logging so it does not accidentally pin old graphs.
    pipe._stage3_last_z_pred = z_pred

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
        "fake_update_every_n_steps": int(stage3_fake_update_ratio),
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
        stage3_fake_update_ratio: int = 1,
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
        merged_inputs[0]["lq_latent_alignment"] = _stage3d31_teacher_lq_alignment_mode(self.pipe)
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

                with torch.inference_mode():
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
                            "validation_mode": "stage3_v7_d3_1_one_step_direct_decode",
                            "validation_mode_detail": "not_streaming_kvcache_validation",
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


def _stage3_val_from_train_batch_enabled() -> bool:
    return os.environ.get("FLASHVSR_STAGE3_VAL_FROM_TRAIN_BATCH", "0").strip().lower() in {"1", "true", "yes"}


def _stage3_overfit_cache_first_batch_enabled() -> bool:
    return os.environ.get("FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH", "0").strip().lower() in {"1", "true", "yes"}


def _stage3_fixed_lqgt_root() -> Optional[str]:
    root = os.environ.get("FLASHVSR_STAGE3_FIXED_LQGT_ROOT", "").strip()
    return root or None


def _stage3_load_fixed_lqgt_batch(root: str, process_index: int) -> Dict[str, Any]:
    root_path = Path(root)
    sample_paths = sorted(root_path.glob("sample_*.pt"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*.pt files found under fixed LQ/GT root: {root}")
    sample_path = sample_paths[int(process_index) % len(sample_paths)]
    payload = torch.load(sample_path, map_location="cpu")
    video = payload["video"]
    lq_video = payload["lq_video"]
    if not torch.is_tensor(video) or not torch.is_tensor(lq_video):
        raise TypeError(f"Fixed LQ/GT sample must contain tensor video/lq_video: {sample_path}")
    if video.ndim != 4 or lq_video.ndim != 4:
        raise ValueError(
            f"Fixed LQ/GT tensors must be [T,C,H,W], got video={tuple(video.shape)} lq={tuple(lq_video.shape)}"
        )
    sample_seed = int(payload.get("sample_seed", int(process_index)))
    sample_id = str(payload.get("sample_id", sample_path.stem))
    return {
        "video": video.to(dtype=torch.bfloat16).unsqueeze(0).contiguous(),
        "lq_video": lq_video.to(dtype=torch.bfloat16).unsqueeze(0).contiguous(),
        "sample_seed": torch.tensor([sample_seed], dtype=torch.long),
        "sample_id": [sample_id],
        "source_type": ["fixed_lqgt_overfit"],
    }


def _stage3_detach_validation_value(value, *, batch_size: int = 1, index: int = 0):
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim > 0 and tensor.shape[0] == batch_size:
            return tensor[index]
        return tensor
    if isinstance(value, dict):
        return {
            key: _stage3_detach_validation_value(item, batch_size=batch_size, index=index)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        if len(value) == batch_size:
            return _stage3_detach_validation_value(value[index], batch_size=1, index=0)
        return tuple(_stage3_detach_validation_value(item, batch_size=batch_size, index=index) for item in value)
    if isinstance(value, list):
        if len(value) == batch_size:
            return _stage3_detach_validation_value(value[index], batch_size=1, index=0)
        return [_stage3_detach_validation_value(item, batch_size=batch_size, index=index) for item in value]
    return value


def _stage3_validation_samples_from_train_batch(data: Dict[str, Any], num_samples: int) -> List[Dict[str, Any]]:
    """Use the already-loaded overfit batch as validation data without scanning the dataset again."""
    if num_samples <= 0:
        return []
    video = data.get("video") if isinstance(data, dict) else None
    batch_size = int(video.shape[0]) if torch.is_tensor(video) and video.ndim == 5 else 1
    count = min(int(num_samples), batch_size)
    return [
        _stage3_detach_validation_value(data, batch_size=batch_size, index=index)
        for index in range(count)
    ]


class Stage3CFakeScalarModel(torch.nn.Module):
    """C0 placeholder for validating a second optimizer/state path.

    This is intentionally not the final G_fake model. It lets v7-C prove the
    dedicated runner can own and save an independent fake optimizer before the
    expensive full-attention G_fake branch is attached in C3/C4.
    """

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, device: torch.device) -> torch.Tensor:
        target = torch.ones((), device=device, dtype=self.scale.dtype)
        return (self.scale.to(device) - target).pow(2)


def _stage3c_training_state_extra_path(output_path: str, step: int) -> str:
    return os.path.join(output_path, "training_state", f"step-{int(step)}", "flashvsr_stage3c_extra.pt")


def save_stage3c_extra_state(accelerator, output_path: str, step: int, fake_model, fake_optimizer, fake_scheduler, args):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return
    path = _stage3c_training_state_extra_path(output_path, step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fake_is_full_model = isinstance(fake_model, FlashVSRStage3BTrainingModule)
    if fake_is_full_model:
        fake_state = fake_model.export_trainable_state_dict(fake_model.state_dict())
        fake_state_note = "v7-C trainable G_fake state: trainable LoRA/lq_proj parameters only."
    else:
        fake_state = fake_model.state_dict()
        fake_state_note = "v7-C0 placeholder fake scalar state."
    payload = {
        "step": int(step),
        "fake_model": fake_state,
        "fake_model_is_full_stage3": bool(fake_is_full_model),
        "fake_optimizer": fake_optimizer.state_dict(),
        "fake_scheduler": fake_scheduler.state_dict() if fake_scheduler is not None else None,
        "stage3c_fake_skeleton_loss_weight": float(getattr(args, "stage3c_fake_skeleton_loss_weight", 0.0)),
        "stage3_fake_fm_weight": float(getattr(args, "stage3_fake_fm_weight", 0.0)),
        "stage3_fake_update_every_n_steps": int(_stage3c_fake_update_every_n_steps(args)),
        "stage3_fake_update_ratio": int(getattr(args, "stage3_fake_update_ratio", 1)),
        "stage3_dfake_gen_update_ratio": int(_stage3d4_dfake_gen_update_ratio(args)),
        "note": fake_state_note,
    }
    torch.save(payload, path)
    accelerator.wait_for_everyone()


def load_stage3c_extra_state_if_available(accelerator, state_dir: str, fake_model, fake_optimizer, fake_scheduler):
    path = os.path.join(state_dir, "flashvsr_stage3c_extra.pt")
    if not os.path.exists(path):
        if accelerator.is_main_process:
            print(f"[stage3c_resume] extra fake state not found, skip: {path}", flush=True)
        return
    payload = torch.load(path, map_location="cpu")
    fake_state = payload.get("fake_model", {})
    if isinstance(fake_model, FlashVSRStage3BTrainingModule):
        result = fake_model.load_state_dict(fake_state, strict=False)
        if accelerator.is_main_process:
            print(
                "[stage3c_resume] loaded trainable G_fake state "
                f"keys={len(fake_state)} missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}",
                flush=True,
            )
    else:
        fake_model.load_state_dict(fake_state, strict=True)
    if payload.get("fake_optimizer") is not None:
        fake_optimizer.load_state_dict(payload["fake_optimizer"])
    if fake_scheduler is not None and payload.get("fake_scheduler") is not None:
        fake_scheduler.load_state_dict(payload["fake_scheduler"])
    if accelerator.is_main_process:
        print(f"[stage3c_resume] loaded extra fake state: {path}", flush=True)


def _freeze_stage3c_probe_model(model: torch.nn.Module) -> int:
    """Freeze a probe teacher/guidance model and return its parameter count."""
    param_count = 0
    for param in model.parameters():
        param_count += int(param.numel())
        param.requires_grad = False
    model.eval()
    return param_count


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters() if param.requires_grad)


def _summarize_trainable_param_groups(model: torch.nn.Module, *, max_examples: int = 12) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": 0,
        "groups": {},
        "examples": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = int(param.numel())
        summary["total"] += numel
        if "lora_" in name:
            group = "lora"
        elif name.startswith("pipe.lq_proj_in."):
            group = "lq_proj_in"
        elif name.startswith("pipe.dit."):
            group = "dit_base_unexpected"
        else:
            group = name.split(".", 2)[0] if name else "other"
        summary["groups"][group] = int(summary["groups"].get(group, 0)) + numel
        if len(summary["examples"]) < max_examples:
            summary["examples"].append(f"{name}:{numel}")
    return summary


def _set_stage3c_fake_lq_proj_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """Toggle G_fake LQ projector training and return the affected parameter count."""
    affected = 0
    for name, param in model.named_parameters():
        if name.startswith("pipe.lq_proj_in."):
            param.requires_grad = bool(trainable)
            affected += int(param.numel())
    return affected


def _average_stage3c_fake_gradients(fake_model: torch.nn.Module) -> None:
    """Synchronize the standalone G_fake gradients without preparing it via Deepspeed.

    Accelerate/Deepspeed manages the student model. G_fake is intentionally kept
    outside that wrapper so the existing student path remains stable. Averaging
    fake gradients here keeps all ranks' standalone fake copies in sync.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    max_chunk_numel = max(1, int(os.environ.get("FLASHVSR_STAGE3_FAKE_GRAD_SYNC_CHUNK_NUMEL", "4194304")))
    for param in fake_model.parameters():
        if param.requires_grad and param.grad is not None:
            flat_grad = param.grad.detach().reshape(-1)
            for grad_chunk in flat_grad.split(max_chunk_numel):
                dist.all_reduce(grad_chunk, op=dist.ReduceOp.SUM)
                grad_chunk.div_(world_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stage3c_should_save_next_step(model_logger, save_steps, extra_save_steps) -> bool:
    next_step = int(model_logger.num_steps) + 1
    if save_steps is not None and int(save_steps) > 0 and next_step % int(save_steps) == 0:
        return True
    if extra_save_steps is not None:
        for step in extra_save_steps:
            if next_step == int(step):
                return True
    return False


def _stage3c_unique_modules(*modules):
    seen = set()
    unique = []
    for module in modules:
        if module is None:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        unique.append(module)
    return unique


def _stage3c_move_modules_to_device(modules, device):
    for module in _stage3c_unique_modules(*modules):
        if hasattr(module, "to"):
            module.to(device)


def _stage3c_move_optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _stage3d44_base_deepspeed_config_path() -> str:
    override_path = os.environ.get("FLASHVSR_STAGE3_DS_CONFIG", "").strip()
    if override_path:
        return override_path
    remote_path = "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json"
    if os.path.exists(remote_path):
        return remote_path
    local_fallback = Path(__file__).resolve().parent / "lora" / "history" / "deepspeed_zero2_flashvsr_nooffload.json"
    if local_fallback.exists():
        return str(local_fallback)
    return remote_path


def _stage3d44_load_deepspeed_config(args, *, micro_batch_size: int, fake: bool) -> Dict[str, Any]:
    """Build a concrete DeepSpeed config for one Accelerate DeepSpeed plugin."""
    plugin = getattr(getattr(args, "deepspeed_plugin", None), "deepspeed_config", None)
    config_path = None
    if fake:
        fake_config_path = os.environ.get("FLASHVSR_STAGE3_FAKE_DS_CONFIG", "").strip()
        if fake_config_path:
            config_path = fake_config_path
    if plugin is None:
        config_path = config_path or getattr(args, "deepspeed_config_file", None)
    if config_path is None:
        config_path = _stage3d44_base_deepspeed_config_path()
    if not os.path.exists(config_path):
        config_path = _stage3d44_base_deepspeed_config_path()
    with open(config_path, "r", encoding="utf-8") as handle:
        ds_config = json.load(handle)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    micro_batch_size = max(1, int(micro_batch_size))
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_size
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = micro_batch_size * grad_accum * max(1, world_size)
    ds_config["zero_force_ds_cpu_optimizer"] = False
    if fake:
        fake_offload = os.environ.get("FLASHVSR_STAGE3_FAKE_DS_OFFLOAD", "0") == "1"
        zero_config = ds_config.setdefault("zero_optimization", {})
        if not fake_offload:
            zero_config.pop("offload_optimizer", None)
            zero_config.pop("offload_param", None)
    return ds_config


def _stage3d44_build_deepspeed_plugins(args, *, micro_batch_size: int) -> Dict[str, DeepSpeedPlugin]:
    student_config = _stage3d44_load_deepspeed_config(args, micro_batch_size=micro_batch_size, fake=False)
    fake_config = _stage3d44_load_deepspeed_config(args, micro_batch_size=micro_batch_size, fake=True)
    return {
        "student": DeepSpeedPlugin(hf_ds_config=student_config),
        "fake": DeepSpeedPlugin(hf_ds_config=fake_config),
    }


def _stage3d44_select_deepspeed_plugin(accelerator, name: str) -> None:
    state = getattr(accelerator, "state", None)
    selector = getattr(state, "select_deepspeed_plugin", None)
    if callable(selector):
        selector(name)


def _stage3d44_active_deepspeed_config(accelerator) -> Dict[str, Any]:
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    config = getattr(plugin, "deepspeed_config", None)
    if isinstance(config, dict):
        return config
    hf_ds_config = getattr(plugin, "hf_ds_config", None)
    hf_config = getattr(hf_ds_config, "config", None)
    return hf_config if isinstance(hf_config, dict) else {}


def _stage3d44_ds_has_offload(ds_config: Dict[str, Any]) -> bool:
    zero_config = ds_config.get("zero_optimization", {}) if isinstance(ds_config, dict) else {}
    if not isinstance(zero_config, dict):
        return False
    return "offload_optimizer" in zero_config or "offload_param" in zero_config


def _stage3d43_fake_checkpoint_dir(output_path: str) -> str:
    return os.path.join(output_path, "stage3_fake_deepspeed")


def _stage3d43_fake_checkpoint_tag(step: int) -> str:
    return f"global_step{int(step)}"


def _stage3d43_save_fake_deepspeed_state(accelerator, output_path: str, step: int, fake_engine, args) -> None:
    if not hasattr(fake_engine, "save_checkpoint"):
        return
    client_state = {
        "step": int(step),
        "stage3_fake_fm_weight": float(getattr(args, "stage3_fake_fm_weight", 0.0)),
        "stage3_dfake_gen_update_ratio": int(_stage3d4_dfake_gen_update_ratio(args)),
        "note": "v7-D4.4 standalone DeepSpeed engine for trainable G_fake.",
    }
    save_dir = _stage3d43_fake_checkpoint_dir(output_path)
    fake_engine.save_checkpoint(save_dir, tag=_stage3d43_fake_checkpoint_tag(step), client_state=client_state)
    if accelerator.is_main_process:
        latest_path = os.path.join(save_dir, "latest")
        try:
            with open(latest_path, "w", encoding="utf-8") as handle:
                handle.write(_stage3d43_fake_checkpoint_tag(step))
        except OSError:
            pass
        print(f"[stage3d43_fake_save] saved G_fake DeepSpeed checkpoint: {save_dir}", flush=True)


def _stage3d43_load_fake_deepspeed_state_if_available(accelerator, state_dir: str, fake_engine) -> None:
    if not hasattr(fake_engine, "load_checkpoint"):
        return
    fake_state_dir = _stage3d43_fake_checkpoint_dir(state_dir)
    if not os.path.exists(fake_state_dir):
        if accelerator.is_main_process:
            print(f"[stage3d43_fake_resume] fake DeepSpeed state not found, skip: {fake_state_dir}", flush=True)
        return
    load_path, client_state = fake_engine.load_checkpoint(fake_state_dir)
    if accelerator.is_main_process:
        print(
            f"[stage3d43_fake_resume] loaded G_fake DeepSpeed checkpoint path={load_path} "
            f"client_state={client_state}",
            flush=True,
        )


def _stage3c_run_validation_with_aux_offload(
    accelerator,
    model,
    model_logger,
    checkpoint_path,
    real_model,
    fake_probe_model,
    fake_model,
    fake_optimizer,
):
    validation_callback = model_logger.validation_callback
    if validation_callback is None or not accelerator.is_main_process:
        return

    aux_modules = _stage3c_unique_modules(real_model, fake_probe_model, fake_model)
    if aux_modules:
        print(f"[stage3c_validation] offloading {len(aux_modules)} auxiliary module(s) to CPU before validation", flush=True)
    _stage3c_move_modules_to_device(aux_modules, torch.device("cpu"))
    _stage3c_move_optimizer_state_to_device(fake_optimizer, torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        validation_callback(
            accelerator=accelerator,
            model=accelerator.unwrap_model(model),
            checkpoint_path=checkpoint_path,
            step=model_logger.num_steps,
        )
    finally:
        _stage3c_move_modules_to_device(aux_modules, accelerator.device)
        _stage3c_move_optimizer_state_to_device(fake_optimizer, accelerator.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if aux_modules:
            print("[stage3c_validation] auxiliary module(s) restored after validation", flush=True)


def _maybe_run_stage3c_probe(probe_model, data, *, probe_every: int, global_step: int, dataset_load_from_cache: bool):
    if probe_model is None:
        return None
    probe_every = max(1, int(probe_every))
    if global_step % probe_every != 0:
        return None
    with torch.no_grad():
        if dataset_load_from_cache:
            probe_loss = probe_model({}, inputs=data)
        else:
            probe_loss = probe_model(data)
    return probe_loss.detach()


def _unwrap_stage3c_model(model):
    return getattr(model, "module", model)


def _get_stage3c_last_z_pred(model):
    module = _unwrap_stage3c_model(model)
    pipe = getattr(module, "pipe", None)
    return None if pipe is None else getattr(pipe, "_stage3_last_z_pred", None)


def _clear_stage3c_last_z_pred(model) -> None:
    module = _unwrap_stage3c_model(model)
    pipe = getattr(module, "pipe", None)
    if pipe is not None and hasattr(pipe, "_stage3_last_z_pred"):
        pipe._stage3_last_z_pred = None


def _stage3c_fake_update_every_n_steps(args) -> int:
    explicit = getattr(args, "stage3_fake_update_every_n_steps", None)
    if explicit is not None:
        return max(1, int(explicit))
    return max(1, int(getattr(args, "stage3_fake_update_ratio", 1)))


def _stage3d4_dfake_gen_update_ratio(args) -> int:
    return max(1, int(getattr(args, "stage3_dfake_gen_update_ratio", 1)))


def _stage3d4_is_generator_turn(args, runner_step: int) -> bool:
    return int(runner_step) % _stage3d4_dfake_gen_update_ratio(args) == 0


def _stage3d44_fake_lq_proj_update_every(args) -> int:
    return max(1, int(getattr(args, "stage3_fake_lq_proj_update_every_n_runner_steps", 1) or 1))


def _stage3d44_should_update_fake_lq_proj(args, runner_step: int) -> bool:
    if not bool(getattr(args, "stage3_fake_train_lq_proj_in", True)):
        return False
    return int(runner_step) % _stage3d44_fake_lq_proj_update_every(args) == 0


def _stage3d31_teacher_lq_alignment_mode(pipe) -> str:
    lq_proj = getattr(pipe, "lq_proj_in", None)
    temporal_mode = getattr(lq_proj, "temporal_mode", "streaming")
    if temporal_mode == "nonstreaming_aligned":
        return "trim_tail_to_match"
    return "exact"


def _stage3c_probe_predict_x0(
    probe_model,
    data,
    clean_latents: torch.Tensor,
    *,
    dataset_load_from_cache: bool,
    probe_name: str = "probe",
    dmd_point: dict | None = None,
    return_dmd_point: bool = False,
):
    """Run a frozen probe model on a shared clean latent and predict x0.

    This is the C4 logging-only approximation of the DMD score path. It keeps
    the probe no-grad and does not update G_real/G_fake. The student-generated
    `clean_latents` is detached before entering the probes.

    v7-D2 uses `dmd_point` to guarantee that G_real and G_fake are evaluated at
    the same noisy latent and timestep, matching the DMD2/OSEDiff score-difference
    semantics.
    """
    module = _unwrap_stage3c_model(probe_model)
    was_training = module.training
    module.eval()
    try:
        pipe = module.pipe
        debug = os.environ.get("FLASHVSR_STAGE3C_DMD_DEBUG", "0") == "1"
        if debug:
            print(
                f"[stage3c_dmd_debug] {probe_name} start "
                f"clean_latents={tuple(clean_latents.shape)} device={clean_latents.device} "
                f"shared_point={dmd_point is not None}",
                flush=True,
            )
        pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        if dataset_load_from_cache:
            inputs = data
        else:
            inputs = module.get_pipeline_inputs(data)
        if debug:
            print(f"[stage3c_dmd_debug] {probe_name} got_pipeline_inputs", flush=True)
        merged_inputs = module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
        if debug:
            print(f"[stage3c_dmd_debug] {probe_name} transferred_to_device device={pipe.device}", flush=True)
        for unit in pipe.units:
            if debug:
                print(f"[stage3c_dmd_debug] {probe_name} unit_start={unit.__class__.__name__}", flush=True)
            merged_inputs = pipe.unit_runner(unit, pipe, *merged_inputs)
            if debug:
                print(f"[stage3c_dmd_debug] {probe_name} unit_end={unit.__class__.__name__}", flush=True)
        merged = {}
        merged.update(merged_inputs[0])
        merged.update(merged_inputs[1])
        merged["lq_latent_alignment"] = _stage3d31_teacher_lq_alignment_mode(pipe)

        clean_latents = clean_latents.detach().to(device=pipe.device, dtype=pipe.torch_dtype)
        if dmd_point is None:
            max_timestep_boundary = int(merged.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
            min_timestep_boundary = int(merged.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
            timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
            noise = torch.randn_like(clean_latents)
            noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timestep)
            dmd_point = {
                "timestep": timestep.detach(),
                "timestep_id": timestep_id.detach(),
                "noise": noise.detach(),
                "noisy_latents": noisy_latents.detach(),
            }
        else:
            timestep = dmd_point["timestep"].to(dtype=pipe.torch_dtype, device=pipe.device)
            noisy_latents = dmd_point["noisy_latents"].to(dtype=pipe.torch_dtype, device=pipe.device)
        merged["input_latents"] = clean_latents
        merged["latents"] = noisy_latents
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        if debug:
            timestep_value = float(timestep.flatten()[0].detach().float().cpu().item())
            print(
                f"[stage3c_dmd_debug] {probe_name} model_fn_start "
                f"timestep={timestep_value:.6f} shared_point={dmd_point is not None} "
                f"lq_alignment={merged['lq_latent_alignment']}",
                flush=True,
            )
        noise_pred = pipe.model_fn(**models, **merged, timestep=timestep)
        if debug:
            print(f"[stage3c_dmd_debug] {probe_name} model_fn_end noise_pred={tuple(noise_pred.shape)}", flush=True)
        x0_pred = pipe.scheduler.step(noise_pred, timestep, noisy_latents, to_final=True)
        if debug:
            print(f"[stage3c_dmd_debug] {probe_name} done x0={tuple(x0_pred.shape)}", flush=True)
        result = x0_pred.detach()
        if return_dmd_point:
            return result, dmd_point
        return result
    finally:
        module.train(was_training)


def _stage3c_fake_fm_loss(
    fake_model,
    data,
    clean_latents: torch.Tensor,
    args,
    global_step: int,
    *,
    dataset_load_from_cache: bool,
):
    """Train G_fake on the current student fake latent distribution.

    This mirrors the DMD2 fake-score update at the flow-matching level:
    detach the student one-step output, add noise at a random timestep, and
    update only G_fake to predict the flow training target for that fake latent.
    """
    fake_weight = float(getattr(args, "stage3_fake_fm_weight", 0.0))
    module = _unwrap_stage3c_model(fake_model)
    if fake_weight == 0.0 or clean_latents is None or not isinstance(module, FlashVSRStage3BTrainingModule):
        return None

    module.train(True)
    pipe = module.pipe
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    inputs = data if dataset_load_from_cache else module.get_pipeline_inputs(data)
    merged_inputs = module.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        merged_inputs = pipe.unit_runner(unit, pipe, *merged_inputs)
    merged = {}
    merged.update(merged_inputs[0])
    merged.update(merged_inputs[1])
    merged["lq_latent_alignment"] = _stage3d31_teacher_lq_alignment_mode(pipe)

    fake_clean_latents = clean_latents.detach().to(device=pipe.device, dtype=pipe.torch_dtype)
    max_timestep_boundary = int(merged.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(merged.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(fake_clean_latents)
    noisy_latents = pipe.scheduler.add_noise(fake_clean_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(fake_clean_latents, noise, timestep)
    merged["input_latents"] = fake_clean_latents
    merged["latents"] = noisy_latents
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    fake_pred = pipe.model_fn(**models, **merged, timestep=timestep)
    fake_loss = F.mse_loss(fake_pred.float(), training_target.float())
    fake_loss = fake_loss * pipe.scheduler.training_weight(timestep)
    return fake_loss * fake_weight


def _maybe_run_stage3c_dmd_probe(real_model, fake_probe_model, data, clean_latents, args, global_step: int, *, dataset_load_from_cache: bool):
    if real_model is None or fake_probe_model is None or clean_latents is None:
        return None
    dmd_every = max(1, int(getattr(args, "stage3_dmd_probe_every", 1)))
    if global_step % dmd_every != 0:
        return None
    with torch.no_grad():
        real_x0, dmd_point = _stage3c_probe_predict_x0(
            real_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="real_probe",
            return_dmd_point=True,
        )
        fake_x0 = _stage3c_probe_predict_x0(
            fake_probe_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="fake_probe",
            dmd_point=dmd_point,
        )
        raw = real_x0.float() - fake_x0.float()
        norm = real_x0.float().abs().mean().clamp_min(1e-6)
        dmd_mag = raw.abs().mean() / norm
    return dmd_mag.detach()


def _maybe_run_stage3c_dmd_student_loss(
    real_model,
    fake_probe_model,
    data,
    clean_latents,
    args,
    global_step: int,
    *,
    dataset_load_from_cache: bool,
):
    """DMD2-style student loss using frozen real/fake score probes.

    This keeps G_real and G_fake fully no-grad and only sends the DMD gradient
    to the student one-step output `clean_latents`. It mirrors the DMD2 pattern:

        grad = ((z - real_x0) - (z - fake_x0)) / norm
        loss = 0.5 * mse(z, (z - grad).detach())

    C5 intentionally does not update G_fake yet; that is a separate runner step.
    """
    dmd_weight = float(getattr(args, "stage3_dmd_weight", 0.0))
    if dmd_weight == 0.0 or real_model is None or fake_probe_model is None or clean_latents is None:
        return None, None, None, None
    dmd_every = max(1, int(getattr(args, "stage3_dmd_probe_every", 1)))
    if global_step % dmd_every != 0:
        return None, None, None, None
    with torch.no_grad():
        real_x0, dmd_point = _stage3c_probe_predict_x0(
            real_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="real_dmd",
            return_dmd_point=True,
        )
        fake_x0 = _stage3c_probe_predict_x0(
            fake_probe_model,
            data,
            clean_latents,
            dataset_load_from_cache=dataset_load_from_cache,
            probe_name="fake_dmd",
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


def launch_stage3c_dual_optimizer_task(
    accelerator,
    fake_accelerator,
    dataset,
    model,
    fake_model,
    model_logger,
    args,
    real_model=None,
    fake_probe_model=None,
):
    """Dedicated v7-D4.4 runner with two Accelerate-managed DeepSpeed engines."""
    learning_rate = args.learning_rate
    fake_learning_rate = float(getattr(args, "stage3c_fake_learning_rate", learning_rate))
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.dataset_num_workers
    save_steps = args.save_steps
    num_epochs = args.num_epochs
    max_train_steps = getattr(args, "max_train_steps", None)
    log_loss_steps = getattr(args, "log_loss_steps", 1)
    extra_save_steps_raw = getattr(args, "extra_save_steps", "") or ""
    extra_save_steps = {int(step.strip()) for step in extra_save_steps_raw.split(",") if step.strip()}

    wandb_run = None
    if getattr(args, "use_wandb", False) and accelerator.is_main_process:
        try:
            import wandb

            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "flashvsr"),
                name=getattr(args, "wandb_name", None),
                entity=getattr(args, "wandb_entity", None),
                mode=getattr(args, "wandb_mode", "online"),
                config=vars(args),
            )
            # D4.x uses a DMD2-style runner where G_fake updates every runner
            # iteration while the student/generator updates once per dfake
            # window. Use runner_step as the W&B x-axis so fake-only updates do
            # not collapse onto the same student/global step.
            wandb.define_metric("train/runner_step")
            wandb.define_metric("train/*", step_metric="train/runner_step")
        except Exception as error:
            print(f"[wandb] init failed: {error}", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    fake_trainable_params = list(param for param in fake_model.parameters() if param.requires_grad)
    if not fake_trainable_params:
        raise RuntimeError("Stage3 v7-C fake_model has no trainable parameters.")
    fake_optimizer = torch.optim.AdamW(fake_trainable_params, lr=fake_learning_rate, weight_decay=0.0)
    fake_scheduler = torch.optim.lr_scheduler.ConstantLR(fake_optimizer)
    fake_lq_proj_params = _set_stage3c_fake_lq_proj_trainable(
        fake_model,
        bool(getattr(args, "stage3_fake_train_lq_proj_in", True)),
    )
    if real_model is not None:
        real_model.to(device=accelerator.device)
        real_model.eval()
    if fake_probe_model is not None:
        fake_probe_model.to(device=accelerator.device)
        fake_probe_model.eval()

    is_iterable_dataset = isinstance(dataset, torch.utils.data.IterableDataset)
    collate_fn = getattr(dataset, "custom_collate_fn", None) or _first_item_collate
    dataloader_dataset = dataset
    dataloader_batch_size = batch_size
    dataloader_collate_fn = collate_fn
    if is_iterable_dataset and batch_size > 1:
        dataloader_dataset = _PreBatchedIterableDataset(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_batch_size = 1
        dataloader_collate_fn = _first_item_collate
    dataloader_kwargs = {
        "batch_size": dataloader_batch_size,
        "shuffle": not is_iterable_dataset,
        "collate_fn": dataloader_collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(args, "dataloader_pin_memory", False)),
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, int(getattr(args, "dataloader_prefetch_factor", 2)))
        dataloader_kwargs["persistent_workers"] = bool(getattr(args, "dataloader_persistent_workers", False))
        multiprocessing_context = getattr(args, "dataloader_multiprocessing_context", None)
        if multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
        dataloader_kwargs["worker_init_fn"] = _init_data_worker_no_cuda
        if _DATALOADER_SUPPORTS_IN_ORDER:
            dataloader_kwargs["in_order"] = bool(getattr(args, "dataloader_in_order", True))
    dataloader = torch.utils.data.DataLoader(dataloader_dataset, **dataloader_kwargs)

    _stage3d44_select_deepspeed_plugin(accelerator, "student")
    model.to(device=accelerator.device)
    if is_iterable_dataset:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)
    fake_probe_shares_trainable_fake = fake_probe_model is fake_model
    _stage3d44_select_deepspeed_plugin(fake_accelerator, "fake")
    fake_model.to(device=fake_accelerator.device)
    fake_model, fake_optimizer, fake_scheduler = fake_accelerator.prepare(fake_model, fake_optimizer, fake_scheduler)
    initialize_deepspeed_gradient_checkpointing(fake_accelerator)
    fake_ds_config = _stage3d44_active_deepspeed_config(fake_accelerator)
    if fake_probe_shares_trainable_fake:
        fake_probe_model = fake_model

    resumed_epoch_id = 0
    global_step = 0
    runner_step = 0
    if getattr(args, "resume_training_state_dir", None):
        resume_state = load_training_state(accelerator, args.resume_training_state_dir, args=args)
        global_step = int(resume_state["step"])
        runner_step = global_step * _stage3d4_dfake_gen_update_ratio(args)
        resumed_epoch_id = int(resume_state["epoch_id"])
        model_logger.set_num_steps(global_step)
        _stage3d43_load_fake_deepspeed_state_if_available(fake_accelerator, args.resume_training_state_dir, fake_model)
        print(
            f"[stage3c_resume] loaded student/fake state student_step={global_step} "
            f"runner_step={runner_step} epoch_id={resumed_epoch_id}",
            flush=True,
        )

    validation_from_train_batch_done = (
        not _stage3_val_from_train_batch_enabled()
        or int(getattr(args, "validation_num_samples", 0) or 0) <= 0
        or model_logger.validation_callback is not None
    )
    overfit_cache_first_batch = _stage3_overfit_cache_first_batch_enabled()
    fixed_lqgt_root = _stage3_fixed_lqgt_root()
    overfit_cached_raw_data = None
    if accelerator.is_main_process:
        if fixed_lqgt_root:
            print(
                "[stage3_overfit] Using pre-generated fixed LQ/GT tensors "
                f"root={fixed_lqgt_root}; online degradation output will be ignored.",
                flush=True,
            )
        if overfit_cache_first_batch:
            print(
                "[stage3_overfit] Caching the first DataLoader batch per rank and reusing it for all runner steps.",
                flush=True,
            )
        print(
            "[stage3c_runner] D4.4 dual Accelerate DeepSpeed engine "
            f"student_lr={learning_rate} fake_lr={fake_learning_rate} "
            f"fake_skeleton_weight={args.stage3c_fake_skeleton_loss_weight} "
            f"fake_fm_weight={args.stage3_fake_fm_weight} "
            f"dfake_gen_update_ratio={_stage3d4_dfake_gen_update_ratio(args)} "
            "student_update_when=runner_step_mod_ratio_eq_0 "
            "fake_update_every_runner_step=1 "
            "fake_loss_uses_current_z_pred_detach=1 "
            "fake_backward=fake_accelerator_deepspeed "
            f"fake_update_every_n_steps_legacy={_stage3c_fake_update_every_n_steps(args)} "
            f"legacy_fake_update_ratio={args.stage3_fake_update_ratio} "
            f"fake_lq_proj_trainable={bool(getattr(args, 'stage3_fake_train_lq_proj_in', True))} "
            f"fake_lq_proj_update_every_runner_steps={_stage3d44_fake_lq_proj_update_every(args)} "
            f"fake_lq_proj_params={fake_lq_proj_params} "
            f"fake_trainable_params={_count_trainable_params(_unwrap_stage3c_model(fake_model))} "
            f"fake_trainable_groups={json.dumps(_summarize_trainable_param_groups(_unwrap_stage3c_model(fake_model)), sort_keys=True)} "
            f"fake_ds_zero_stage={fake_ds_config.get('zero_optimization', {}).get('stage', 'none')} "
            f"fake_ds_reduce_bucket={fake_ds_config.get('zero_optimization', {}).get('reduce_bucket_size', 'auto')} "
            f"fake_ds_allgather_bucket={fake_ds_config.get('zero_optimization', {}).get('allgather_bucket_size', 'auto')} "
            f"fake_ds_offload={int(_stage3d44_ds_has_offload(fake_ds_config))}",
            flush=True,
        )
        if real_model is not None:
            print(
                "[stage3c_runner] C2 frozen G_real probe enabled "
                f"checkpoint={args.stage3_real_checkpoint} "
                f"attention_mode={args.stage3_real_attention_mode} "
                f"probe_every={args.stage3_real_probe_every}",
                flush=True,
            )
        if fake_probe_model is not None:
            print(
                "[stage3c_runner] C3 frozen G_fake probe enabled "
                f"checkpoint={args.stage3_fake_probe_checkpoint} "
                f"attention_mode={args.stage3_fake_probe_attention_mode} "
                f"probe_every={args.stage3_fake_probe_every}",
                flush=True,
            )

    for epoch_id in range(resumed_epoch_id, num_epochs):
        progress_bar = tqdm(dataloader)
        data_iterator = iter(progress_bar)
        while True:
            timing_debug = os.environ.get("FLASHVSR_STAGE3_TIMING_DEBUG", "0") == "1"
            timing_last = _stage3_timing_now(accelerator.device) if timing_debug else None
            timing_parts = {}
            if fixed_lqgt_root:
                if overfit_cached_raw_data is None:
                    overfit_cached_raw_data = _stage3_load_fixed_lqgt_batch(fixed_lqgt_root, accelerator.process_index)
                    if accelerator.is_main_process:
                        print("[stage3_overfit] Loaded fixed LQ/GT batch for reuse.", flush=True)
                data = overfit_cached_raw_data
            elif overfit_cache_first_batch and overfit_cached_raw_data is not None:
                data = overfit_cached_raw_data
            else:
                try:
                    data = next(data_iterator)
                except StopIteration:
                    break
                if overfit_cache_first_batch and overfit_cached_raw_data is None:
                    overfit_cached_raw_data = data
                    if accelerator.is_main_process:
                        print("[stage3_overfit] Cached first training batch for reuse.", flush=True)
            raw_data_for_validation = data
            if is_iterable_dataset:
                data = send_to_device(data, accelerator.device)
            if not validation_from_train_batch_done:
                if accelerator.is_main_process:
                    validation_samples = _stage3_validation_samples_from_train_batch(
                        raw_data_for_validation,
                        int(getattr(args, "validation_num_samples", 0) or 0),
                    )
                    model_logger.validation_callback = FlashVSRStage3BValidationCallback(
                        output_path=args.output_path,
                        validation_samples=validation_samples,
                        num_inference_steps=args.validation_num_inference_steps,
                        fps=args.validation_fps,
                        seed_base=(args.global_seed if args.global_seed is not None else 20260513),
                        use_wandb=args.use_wandb,
                    )
                    print(
                        "[stage3_overfit] Prepared "
                        f"{len(validation_samples)} validation sample(s) from first training batch.",
                        flush=True,
                    )
                validation_from_train_batch_done = True
                accelerator.wait_for_everyone()
            if timing_debug:
                now = _stage3_timing_now(accelerator.device)
                timing_parts["data"] = now - timing_last
                timing_last = now

            current_runner_step = runner_step
            generator_turn = _stage3d4_is_generator_turn(args, current_runner_step)
            fake_lq_proj_turn = _stage3d44_should_update_fake_lq_proj(args, current_runner_step)
            if bool(getattr(args, "stage3_fake_train_lq_proj_in", True)):
                _set_stage3c_fake_lq_proj_trainable(
                    _unwrap_stage3c_model(fake_model),
                    fake_lq_proj_turn,
                )
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                try:
                    fake_optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    fake_optimizer.zero_grad()
                if hasattr(fake_model, "zero_grad"):
                    fake_model.zero_grad()
                if generator_turn:
                    if dataset.load_from_cache:
                        student_loss = model({}, inputs=data)
                    else:
                        student_loss = model(data)
                else:
                    with torch.no_grad():
                        if dataset.load_from_cache:
                            student_loss = model({}, inputs=data)
                        else:
                            student_loss = model(data)
                student_z_pred = _get_stage3c_last_z_pred(model)
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["student"] = now - timing_last
                    timing_last = now
                real_probe_loss = _maybe_run_stage3c_probe(
                    real_model,
                    data,
                    probe_every=getattr(args, "stage3_real_probe_every", 1),
                    global_step=global_step,
                    dataset_load_from_cache=bool(dataset.load_from_cache),
                )
                fake_probe_loss = _maybe_run_stage3c_probe(
                    fake_probe_model,
                    data,
                    probe_every=getattr(args, "stage3_fake_probe_every", 1),
                    global_step=global_step,
                    dataset_load_from_cache=bool(dataset.load_from_cache),
                )
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["probe"] = now - timing_last
                    timing_last = now
                dmd_student_loss, dmd_grad_norm, dmd_skipped, dmd_loss_clamped = _maybe_run_stage3c_dmd_student_loss(
                    real_model,
                    fake_probe_model,
                    data,
                    student_z_pred,
                    args,
                    global_step,
                    dataset_load_from_cache=bool(dataset.load_from_cache),
                ) if generator_turn else (None, None, None, None)
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["dmd"] = now - timing_last
                    timing_last = now
                dmd_probe_loss = None
                if generator_turn and dmd_student_loss is None:
                    dmd_probe_loss = _maybe_run_stage3c_dmd_probe(
                        real_model,
                        fake_probe_model,
                        data,
                        student_z_pred,
                        args,
                        global_step,
                        dataset_load_from_cache=bool(dataset.load_from_cache),
                    )
                    if timing_debug:
                        now = _stage3_timing_now(accelerator.device)
                        timing_parts["dmd_probe"] = now - timing_last
                        timing_last = now
                fake_did_update = True
                if isinstance(fake_model, Stage3CFakeScalarModel):
                    fake_loss = fake_model(accelerator.device) * float(args.stage3c_fake_skeleton_loss_weight)
                else:
                    fake_loss = _stage3c_fake_fm_loss(
                        fake_model,
                        data,
                        student_z_pred,
                        args,
                        global_step,
                        dataset_load_from_cache=bool(dataset.load_from_cache),
                    )
                    fake_did_update = fake_loss is not None
                    if fake_loss is None:
                        fake_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["fake"] = now - timing_last
                    timing_last = now
                if fake_did_update:
                    # Fake critic training must consume the current student
                    # sample, but not route gradients back into the student.
                    # _stage3c_fake_fm_loss detaches student_z_pred internally.
                    fake_accelerator.backward(fake_loss)
                    if timing_debug:
                        now = _stage3_timing_now(accelerator.device)
                        timing_parts["fake_backward_sync"] = now - timing_last
                        timing_last = now
                student_total_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                if generator_turn:
                    student_total_loss = student_loss
                    if dmd_student_loss is not None:
                        student_total_loss = student_total_loss + dmd_student_loss
                    # Keep the student/DeepSpeed backward graph free of fake
                    # critic parameters. This preserves DMD2-style optimizer
                    # ownership while avoiding the fragile two-turn runner.
                    accelerator.backward(student_total_loss)
                    if timing_debug:
                        now = _stage3_timing_now(accelerator.device)
                        timing_parts["student_backward"] = now - timing_last
                        timing_last = now
                total_loss = fake_loss.detach()
                if generator_turn:
                    total_loss = total_loss + student_total_loss.detach()
                _clear_stage3c_last_z_pred(model)
                if generator_turn:
                    optimizer.step()
                if fake_did_update:
                    fake_optimizer.step()
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["optim"] = now - timing_last
                    timing_last = now
                should_save_checkpoint = generator_turn and _stage3c_should_save_next_step(model_logger, save_steps, extra_save_steps)
                save_has_validation = should_save_checkpoint and int(getattr(args, "validation_num_samples", 0) or 0) > 0
                should_save_with_validation = model_logger.validation_callback is not None and save_has_validation
                validation_callback = None
                if should_save_with_validation:
                    # C6 keeps trainable G_fake plus frozen probes resident during training.
                    # Save first, then run validation manually after offloading auxiliaries.
                    validation_callback = model_logger.validation_callback
                    model_logger.validation_callback = None
                saved_checkpoint_path = None
                if generator_turn:
                    saved_checkpoint_path = model_logger.on_step_end(
                        accelerator,
                        model,
                        save_steps,
                        extra_save_steps=extra_save_steps,
                        loss=total_loss,
                    )
                    global_step = model_logger.num_steps
                if saved_checkpoint_path is not None:
                    save_training_state(
                        accelerator=accelerator,
                        output_path=model_logger.output_path,
                        step=model_logger.num_steps,
                        epoch_id=epoch_id,
                        args=args,
                    )
                    _stage3d43_save_fake_deepspeed_state(
                        fake_accelerator,
                        model_logger.output_path,
                        model_logger.num_steps,
                        fake_model,
                        args,
                    )
                    if validation_callback is not None:
                        model_logger.validation_callback = validation_callback
                        _stage3c_run_validation_with_aux_offload(
                            accelerator,
                            model,
                            model_logger,
                            saved_checkpoint_path,
                            real_model,
                            fake_probe_model,
                            fake_model,
                            None,
                        )
                    if save_has_validation:
                        accelerator.wait_for_everyone()
                if should_save_with_validation and model_logger.validation_callback is None:
                    model_logger.validation_callback = validation_callback
                if generator_turn:
                    scheduler.step()
                if fake_did_update:
                    fake_scheduler.step()
                if timing_debug:
                    now = _stage3_timing_now(accelerator.device)
                    timing_parts["save_sched"] = now - timing_last
                    timing_last = now
                runner_step = current_runner_step + 1

                no_gather_log = os.environ.get("FLASHVSR_STAGE3C_NO_GATHER_LOG", "0") == "1"
                if no_gather_log:
                    gathered_student = student_loss.detach().float().mean()
                    gathered_total = total_loss.detach().float().mean()
                else:
                    gathered_student = accelerator.gather(student_loss.detach()).mean()
                    gathered_total = accelerator.gather(total_loss.detach()).mean()
                if accelerator.is_main_process:
                    stage3_loss_parts = {}
                    stage3_pipe = getattr(_unwrap_stage3c_model(model), "pipe", None)
                    if stage3_pipe is not None:
                        stage3_loss_parts = dict(getattr(stage3_pipe, "_stage3_last_losses", {}) or {})
                    student_value = float(gathered_student.item())
                    total_value = float(gathered_total.item())
                    fake_value = float(fake_loss.detach().cpu().item())
                    fake_param = float(_unwrap_stage3c_model(fake_model).scale.detach().cpu().item()) if isinstance(_unwrap_stage3c_model(fake_model), Stage3CFakeScalarModel) else 0.0
                    real_probe_value = None
                    if real_probe_loss is not None:
                        real_probe_value = float(real_probe_loss.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(real_probe_loss).mean().item())
                    fake_probe_value = None
                    if fake_probe_loss is not None:
                        fake_probe_value = float(fake_probe_loss.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(fake_probe_loss).mean().item())
                    dmd_probe_value = None
                    if dmd_probe_loss is not None:
                        dmd_probe_value = float(dmd_probe_loss.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(dmd_probe_loss).mean().item())
                    dmd_student_value = None
                    if dmd_student_loss is not None:
                        dmd_student_value = float(dmd_student_loss.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(dmd_student_loss.detach()).mean().item())
                    dmd_grad_value = None
                    if dmd_grad_norm is not None:
                        dmd_grad_value = float(dmd_grad_norm.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(dmd_grad_norm.detach()).mean().item())
                    dmd_skip_value = None
                    if dmd_skipped is not None:
                        dmd_skip_value = float(dmd_skipped.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(dmd_skipped.detach()).mean().item())
                    dmd_loss_clamp_value = None
                    if dmd_loss_clamped is not None:
                        dmd_loss_clamp_value = float(dmd_loss_clamped.detach().float().mean().item()) if no_gather_log else float(accelerator.gather(dmd_loss_clamped.detach()).mean().item())
                    progress_bar.set_postfix(loss=f"{total_value:.6f}", step=global_step, runner=current_runner_step)
                    if log_loss_steps and current_runner_step % log_loss_steps == 0:
                        real_probe_text = "" if real_probe_value is None else f" real_probe={real_probe_value:.6f}"
                        fake_probe_text = "" if fake_probe_value is None else f" fake_probe={fake_probe_value:.6f}"
                        dmd_probe_text = "" if dmd_probe_value is None else f" dmd_probe={dmd_probe_value:.6f}"
                        dmd_student_text = "" if dmd_student_value is None else f" dmd_student={dmd_student_value:.6f}"
                        dmd_grad_text = "" if dmd_grad_value is None else f" dmd_grad={dmd_grad_value:.6f}"
                        dmd_skip_text = "" if dmd_skip_value is None else f" dmd_skip={dmd_skip_value:.0f}"
                        dmd_loss_clamp_text = "" if dmd_loss_clamp_value is None else f" dmd_loss_clamp={dmd_loss_clamp_value:.0f}"
                        print(
                            "[stage3c_train] "
                            f"epoch={epoch_id} step={global_step} runner_step={current_runner_step} "
                            f"generator_update={int(generator_turn)} dfake_gen_update_ratio={_stage3d4_dfake_gen_update_ratio(args)} "
                            f"loss={total_value:.6f} student={student_value:.6f} "
                            f"fake_loss={fake_value:.8f} fake_update={int(fake_did_update)} "
                            f"fake_lq_proj_update={int(fake_lq_proj_turn)} fake_scale={fake_param:.6f}"
                            f"{real_probe_text}{fake_probe_text}{dmd_probe_text}"
                            f"{dmd_student_text}{dmd_grad_text}{dmd_skip_text}{dmd_loss_clamp_text}",
                            flush=True,
                        )
                        if timing_debug:
                            timing_text = " ".join(f"{key}={value:.3f}s" for key, value in timing_parts.items())
                            print(f"[stage3_timing] step={global_step} {timing_text}", flush=True)
                    if wandb_run is not None:
                        log_payload = {
                            "train/loss": total_value,
                            "train/fake_fm_loss": fake_value,
                            "train/generator_did_update": int(generator_turn),
                            "train/runner_step": current_runner_step,
                            "train/fake_lq_proj_did_update": int(fake_lq_proj_turn),
                        }
                        if stage3_loss_parts:
                            recon_flow = float(stage3_loss_parts.get("loss_flow", 0.0))
                            recon_mse = float(stage3_loss_parts.get("loss_mse", 0.0))
                            recon_lpips = float(stage3_loss_parts.get("loss_lpips", 0.0))
                            log_payload.update(
                                {
                                    "train/flow_loss": recon_flow,
                                    "train/mse_loss": recon_mse,
                                    "train/lpips_loss": recon_lpips,
                                }
                            )
                        if dmd_student_value is not None:
                            log_payload["train/dmd_student_loss"] = dmd_student_value
                        wandb_run.log(log_payload, step=current_runner_step)
                if max_train_steps is not None and global_step >= max_train_steps:
                    break
        if max_train_steps is not None and global_step >= max_train_steps:
            break

    saved_checkpoint_path = model_logger.on_training_end(accelerator, model, save_steps, extra_save_steps=extra_save_steps)
    if saved_checkpoint_path is not None:
        save_training_state(
            accelerator=accelerator,
            output_path=model_logger.output_path,
            step=model_logger.num_steps,
            epoch_id=epoch_id if num_epochs > 0 else 0,
            args=args,
        )
        _stage3d43_save_fake_deepspeed_state(
            fake_accelerator,
            model_logger.output_path,
            model_logger.num_steps,
            fake_model,
            args,
        )
    if wandb_run is not None:
        wandb_run.finish()


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
        help="Trainable G_fake initialization checkpoint. For final Stage3 this should be a G_real/Stage1 copy.",
    )
    parser.add_argument(
        "--stage3_fake_attention_mode",
        type=str,
        default="dense_full",
        choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"),
        help="Attention mode for trainable G_fake. Use dense_full for the Stage1-copy DMD2 path.",
    )
    parser.add_argument(
        "--stage3_fake_lq_proj_temporal_mode",
        type=str,
        default="nonstreaming_aligned",
        choices=("streaming", "nonstreaming", "nonstreaming_aligned"),
        help="v7-D3.1: LQ projector temporal mode for trainable G_fake. Use nonstreaming_aligned for Stage1 v5.3.5 teacher-copy semantics.",
    )
    parser.add_argument(
        "--stage3_fake_train_lq_proj_in",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether trainable G_fake also updates lq_proj_in. "
            "For D4.4 long runs this can be disabled to keep G_fake as a LoRA-only "
            "critic and avoid a 288M-parameter fake-side ZeRO all-reduce every runner step."
        ),
    )
    parser.add_argument(
        "--stage3_fake_lq_proj_update_every_n_runner_steps",
        type=int,
        default=1,
        help=(
            "Keep G_fake lq_proj_in trainable, but only enable its gradients every N runner steps. "
            "LoRA still updates every fake step; this reduces the 288M fake-projector ZeRO sync pressure "
            "without permanently freezing the projector."
        ),
    )
    parser.add_argument(
        "--stage3_fake_fm_weight",
        type=float,
        default=0.0,
        help="Reserved for Stage3 v7-C full DMD. C0 keeps this 0 and only validates optimizer/state wiring.",
    )
    parser.add_argument(
        "--stage3_fake_update_every_n_steps",
        type=int,
        default=None,
        help=(
            "v7-D2 explicit fake critic schedule: update G_fake once every N student steps. "
            "This is not DMD2's fake-updates-per-generator ratio."
        ),
    )
    parser.add_argument(
        "--stage3_fake_update_ratio",
        type=int,
        default=1,
        help=(
            "Deprecated legacy alias for stage3_fake_update_every_n_steps. "
            "Larger values mean less frequent fake updates in this runner."
        ),
    )
    parser.add_argument(
        "--stage3_dfake_gen_update_ratio",
        type=int,
        default=1,
        help=(
            "v7-D4 DMD2-style dfake/gen ratio. G_fake updates every runner iteration; "
            "the student/generator updates only when runner_step % ratio == 0. "
            "Larger values mean more fake critic updates per student update."
        ),
    )
    parser.add_argument("--stage3c_fake_learning_rate", type=float, default=None)
    parser.add_argument(
        "--stage3c_fake_skeleton_loss_weight",
        type=float,
        default=1e-6,
        help="C0-only tiny placeholder loss for proving the second optimizer/state path. Set 0 to disable fake updates.",
    )
    parser.add_argument(
        "--stage3_real_checkpoint",
        type=str,
        default=None,
        help="Optional frozen G_real checkpoint for v7-C2 probe. This is no-grad and not part of optimizer.",
    )
    parser.add_argument(
        "--stage3_real_attention_mode",
        type=str,
        default="dense_full",
        choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"),
        help="Attention mode for frozen G_real probe. Use dense_full for Stage1/full-attention teacher probe.",
    )
    parser.add_argument(
        "--stage3_real_lq_proj_temporal_mode",
        type=str,
        default="nonstreaming_aligned",
        choices=("streaming", "nonstreaming", "nonstreaming_aligned"),
        help="v7-D3.1: LQ projector temporal mode for frozen G_real. Use nonstreaming_aligned for Stage1 v5.3.5 teacher semantics.",
    )
    parser.add_argument(
        "--stage3_real_probe_every",
        type=int,
        default=1,
        help="Run frozen G_real no-grad probe every N student steps when --stage3_real_checkpoint is set.",
    )
    parser.add_argument(
        "--stage3_fake_probe_checkpoint",
        type=str,
        default=None,
        help="Optional frozen G_fake probe checkpoint for v7-C3. This is logging-only and not optimized.",
    )
    parser.add_argument(
        "--stage3_fake_probe_attention_mode",
        type=str,
        default="block_sparse_chunk_causal",
        choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"),
        help="Attention mode for frozen G_fake probe.",
    )
    parser.add_argument(
        "--stage3_fake_probe_every",
        type=int,
        default=1,
        help="Run frozen G_fake no-grad probe every N student steps when --stage3_fake_probe_checkpoint is set.",
    )
    parser.add_argument(
        "--stage3_dmd_probe_every",
        type=int,
        default=1,
        help="Run logging-only DMD direction probe every N steps when both frozen G_real and G_fake probes are set.",
    )
    parser.add_argument(
        "--stage3_dmd_weight",
        type=float,
        default=0.0,
        help="Weight for C5 DMD2-style student loss. 0 keeps C4 as logging-only probe.",
    )
    parser.add_argument(
        "--stage3_dmd_grad_norm_max",
        type=float,
        default=0.0,
        help="If >0, treat DMD gradients with mean abs norm above this value as spikes.",
    )
    parser.add_argument(
        "--stage3_dmd_loss_max",
        type=float,
        default=0.0,
        help="If >0, rescale the DMD gradient when the unweighted DMD student loss exceeds this value.",
    )
    parser.add_argument(
        "--stage3_dmd_spike_policy",
        type=str,
        default="none",
        choices=("none", "skip", "clamp"),
        help="How to handle DMD gradient spikes: keep, skip this DMD term, or clamp gradient magnitude.",
    )
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
        parser.error("Stage3 v7-C currently follows Stage2 89->85 streaming shape, requiring num_frames % 8 == 1.")
    if args.stage3c_fake_learning_rate is None:
        args.stage3c_fake_learning_rate = args.learning_rate
    if (float(getattr(args, "stage3_fake_fm_weight", 0.0)) > 0.0 or float(getattr(args, "stage3_dmd_weight", 0.0)) > 0.0) and not getattr(
        args, "stage3_fake_checkpoint", None
    ):
        parser.error(
            "v7-D3.1 requires --stage3_fake_checkpoint when stage3_fake_fm_weight or stage3_dmd_weight is enabled; "
            "otherwise it would fall back to the scalar fake placeholder."
        )
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
    deepspeed_plugins = _stage3d44_build_deepspeed_plugins(args, micro_batch_size=max(1, int(args.batch_size)))
    accelerator = accelerate.Accelerator(
        **accelerator_kwargs,
        deepspeed_plugins=deepspeed_plugins,
    )
    _stage3d44_select_deepspeed_plugin(accelerator, "student")
    fake_accelerator = accelerate.Accelerator()
    _stage3d44_select_deepspeed_plugin(fake_accelerator, "fake")
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
        # Fake FM is handled by the dedicated v7-C runner and its second
        # optimizer. The student forward must keep the v7-B guard disabled.
        stage3_fake_fm_weight=0.0,
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
        print("Stage3 v7-C0: dual optimizer runner skeleton + v7-B one-step reconstruction path", flush=True)
        print(f"stage3_recon_num_latents={args.stage3_recon_num_latents}", flush=True)
        print(f"stage3_loss_weights flow={args.stage3_flow_weight} mse={args.stage3_mse_weight} lpips={args.stage3_lpips_weight}", flush=True)
        print(
            "stage3c_fake "
            f"fm_weight={args.stage3_fake_fm_weight} "
            f"dfake_gen_update_ratio={_stage3d4_dfake_gen_update_ratio(args)} "
            f"fake_update_every_n_steps_legacy={_stage3c_fake_update_every_n_steps(args)} "
            f"legacy_update_ratio={args.stage3_fake_update_ratio} "
            f"checkpoint={args.stage3_fake_checkpoint} "
            f"attention_mode={args.stage3_fake_attention_mode} "
            f"fake_lr={args.stage3c_fake_learning_rate} "
            f"skeleton_weight={args.stage3c_fake_skeleton_loss_weight}",
            flush=True,
        )
        print(
            "stage3c_real "
            f"checkpoint={args.stage3_real_checkpoint} "
            f"attention_mode={args.stage3_real_attention_mode} "
            f"probe_every={args.stage3_real_probe_every}",
            flush=True,
        )
        print(
            "stage3c_fake_probe "
            f"checkpoint={args.stage3_fake_probe_checkpoint} "
            f"attention_mode={args.stage3_fake_probe_attention_mode} "
            f"probe_every={args.stage3_fake_probe_every}",
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
        if _stage3_val_from_train_batch_enabled():
            print(
                "[stage3_overfit] Deferring validation sample creation to first training batch.",
                flush=True,
            )
        else:
            print("Preparing fixed Stage3 v7-C validation samples...", flush=True)
            validation_samples = v5.collect_fixed_validation_samples(dataset, args.validation_num_samples)
            print(f"Prepared {len(validation_samples)} fixed Stage3 v7-C validation samples.", flush=True)
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
    real_model = None
    if args.stage3_real_checkpoint is not None:
        real_model = FlashVSRStage3BTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            prompt_tensor_path=args.prompt_tensor_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=None,
            lq_proj_checkpoint=None,
            resume_stage1_checkpoint=args.stage3_real_checkpoint,
            lq_proj_layer_num=args.lq_proj_layer_num,
            lq_proj_scale=args.lq_proj_scale,
            zero_init_lq_proj_in=False,
            freeze_lq_proj_in=True,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            stage2_attention_mode=args.stage3_real_attention_mode,
            stage2_topk_ratio=args.stage2_topk_ratio,
            stage2_local_num=args.stage2_local_num,
            lq_proj_temporal_mode=args.stage3_real_lq_proj_temporal_mode,
            stage3_recon_num_latents=args.stage3_recon_num_latents,
            stage3_flow_weight=1.0,
            stage3_mse_weight=0.0,
            stage3_lpips_weight=0.0,
            stage3_lpips_net=args.stage3_lpips_net,
            stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
            stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
            stage3_decoder_cpu_offload=args.stage3_decoder_cpu_offload,
            stage3_compute_z_pred=False,
            stage3_fake_fm_weight=0.0,
            stage3_fake_update_ratio=args.stage3_fake_update_ratio,
            stage3_fake_checkpoint=None,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        )
        real_param_count = _freeze_stage3c_probe_model(real_model)
        if accelerator.is_local_main_process:
            print(
                "Stage3 v7-C2 frozen G_real probe loaded "
                f"params={real_param_count} checkpoint={args.stage3_real_checkpoint} "
                f"attention_mode={args.stage3_real_attention_mode} "
                f"lq_proj_temporal_mode={args.stage3_real_lq_proj_temporal_mode}",
                flush=True,
            )
    if args.stage3_fake_checkpoint is not None:
        fake_model = FlashVSRStage3BTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            prompt_tensor_path=args.prompt_tensor_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=None,
            lq_proj_checkpoint=None,
            resume_stage1_checkpoint=args.stage3_fake_checkpoint,
            lq_proj_layer_num=args.lq_proj_layer_num,
            lq_proj_scale=args.lq_proj_scale,
            zero_init_lq_proj_in=False,
            freeze_lq_proj_in=not bool(args.stage3_fake_train_lq_proj_in),
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            stage2_attention_mode=args.stage3_fake_attention_mode,
            stage2_topk_ratio=args.stage2_topk_ratio,
            stage2_local_num=args.stage2_local_num,
            lq_proj_temporal_mode=args.stage3_fake_lq_proj_temporal_mode,
            stage3_recon_num_latents=args.stage3_recon_num_latents,
            stage3_flow_weight=1.0,
            stage3_mse_weight=0.0,
            stage3_lpips_weight=0.0,
            stage3_lpips_net=args.stage3_lpips_net,
            stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
            stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
            stage3_decoder_cpu_offload=args.stage3_decoder_cpu_offload,
            stage3_compute_z_pred=False,
            stage3_fake_fm_weight=0.0,
            stage3_fake_update_ratio=args.stage3_fake_update_ratio,
            stage3_fake_checkpoint=None,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        )
        fake_lq_proj_param_count = _set_stage3c_fake_lq_proj_trainable(
            fake_model,
            bool(args.stage3_fake_train_lq_proj_in),
        )
        if accelerator.is_local_main_process:
            print(
                "Stage3 v7-C6 trainable G_fake loaded "
                f"trainable_params={_count_trainable_params(fake_model)} "
                f"checkpoint={args.stage3_fake_checkpoint} "
                f"attention_mode={args.stage3_fake_attention_mode} "
                f"lq_proj_temporal_mode={args.stage3_fake_lq_proj_temporal_mode} "
                f"train_lq_proj_in={bool(args.stage3_fake_train_lq_proj_in)} "
                f"lq_proj_param_count={fake_lq_proj_param_count}",
                flush=True,
            )
    else:
        fake_model = Stage3CFakeScalarModel()
    fake_probe_model = None
    if args.stage3_fake_probe_checkpoint is not None:
        fake_probe_model = FlashVSRStage3BTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            prompt_tensor_path=args.prompt_tensor_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=None,
            lq_proj_checkpoint=None,
            resume_stage1_checkpoint=args.stage3_fake_probe_checkpoint,
            lq_proj_layer_num=args.lq_proj_layer_num,
            lq_proj_scale=args.lq_proj_scale,
            zero_init_lq_proj_in=False,
            freeze_lq_proj_in=True,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            stage2_attention_mode=args.stage3_fake_probe_attention_mode,
            stage2_topk_ratio=args.stage2_topk_ratio,
            stage2_local_num=args.stage2_local_num,
            lq_proj_temporal_mode=args.stage3_fake_lq_proj_temporal_mode,
            stage3_recon_num_latents=args.stage3_recon_num_latents,
            stage3_flow_weight=1.0,
            stage3_mse_weight=0.0,
            stage3_lpips_weight=0.0,
            stage3_lpips_net=args.stage3_lpips_net,
            stage3_first_frame_pixel_weight=args.stage3_first_frame_pixel_weight,
            stage3_first_frame_lpips_weight=args.stage3_first_frame_lpips_weight,
            stage3_decoder_cpu_offload=args.stage3_decoder_cpu_offload,
            stage3_compute_z_pred=False,
            stage3_fake_fm_weight=0.0,
            stage3_fake_update_ratio=args.stage3_fake_update_ratio,
            stage3_fake_checkpoint=None,
            fp8_models=args.fp8_models,
            offload_models=args.offload_models,
            device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        )
        fake_probe_param_count = _freeze_stage3c_probe_model(fake_probe_model)
        if accelerator.is_local_main_process:
            print(
                "Stage3 v7-C3 frozen G_fake probe loaded "
                f"params={fake_probe_param_count} checkpoint={args.stage3_fake_probe_checkpoint} "
                f"attention_mode={args.stage3_fake_probe_attention_mode} "
                f"lq_proj_temporal_mode={args.stage3_fake_lq_proj_temporal_mode}",
                flush=True,
            )
    elif isinstance(fake_model, FlashVSRStage3BTrainingModule):
        fake_probe_model = fake_model
        if accelerator.is_local_main_process:
            print(
                "Stage3 v7-C6 uses trainable G_fake also as no-grad DMD fake probe.",
                flush=True,
            )
    launch_stage3c_dual_optimizer_task(
        accelerator,
        fake_accelerator,
        dataset,
        model,
        fake_model,
        model_logger,
        args=args,
        real_model=real_model,
        fake_probe_model=fake_probe_model,
    )


if __name__ == "__main__":
    main()
