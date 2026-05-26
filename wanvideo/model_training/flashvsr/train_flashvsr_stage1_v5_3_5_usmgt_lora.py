import os
import sys
import traceback
import warnings
import argparse
import json
import random
from copy import deepcopy
from typing import List, Dict, Any, Optional, Sequence
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from torch.distributed.elastic.multiprocessing.errors import record

from diffsynth.core import UnifiedDataset, ModelConfig, gradient_checkpoint_forward
from diffsynth.core.data.operators import LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.core.loader.file import load_state_dict
from diffsynth.diffusion import *
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_dit_joint_v5 import (
    _build_segment_token_lengths,
    _patchify_tokens_with_segments,
    build_joint_wan_from_existing_v5,
)
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ShapeChecker,
)
from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.joint_batching_v5 import collate_image_video_joint_v5
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v3 import FlashVSRTarStreamingDatasetV3
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v5 import FlashVSRTarStreamingDatasetV5
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53_usmgt import FlashVSRTarStreamingDatasetV53USMGT

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_T = 2
_FLASHVSR_BLOCK_BRANCH_REPORTED = False
_TENSOR_DEBUG_REPORTED = set()


def _serialize_sample_seed(sample_seed: Any) -> Any:
    if torch.is_tensor(sample_seed):
        if sample_seed.numel() == 1:
            return int(sample_seed.item())
        return [int(item) for item in sample_seed.detach().cpu().reshape(-1).tolist()]
    if sample_seed is None:
        return -1
    return int(sample_seed)


def _resolve_per_sample_token_lengths(
    dit: WanModel,
    x: torch.Tensor,
    f: int,
    h: int,
    w: int,
    sequence_lengths: Optional[torch.Tensor] = None,
    segment_lengths: Optional[Sequence[Sequence[int]]] = None,
):
    if segment_lengths is None:
        return None
    if sequence_lengths is None:
        raise ValueError("segment_lengths requires sequence_lengths for packed joint attention")
    def _to_latent_frames(frame_count: int) -> int:
        frame_count = max(1, int(frame_count))
        return ((frame_count - 1) // 4) + 1

    raw_sequence_lengths = [int(length) for length in sequence_lengths.detach().cpu().tolist()]
    latent_segment_lengths = [
        [_to_latent_frames(int(length)) for length in one_sample if int(length) > 0]
        for one_sample in segment_lengths
    ]
    for sample_index, one_sample in enumerate(segment_lengths):
        one_sample_lengths = [int(length) for length in one_sample if int(length) > 0]
        if sum(one_sample_lengths) != raw_sequence_lengths[sample_index]:
            raise ValueError(
                f"segment_lengths={one_sample_lengths} do not sum to raw sequence_length={raw_sequence_lengths[sample_index]}"
            )
    sequence_lengths_list = [sum(one_sample) for one_sample in latent_segment_lengths]
    return _build_segment_token_lengths(
        batch_size=int(x.shape[0]),
        frames_after_patchify=int(f),
        h_tokens=int(h),
        w_tokens=int(w),
        sequence_lengths=sequence_lengths_list,
        segment_lengths=latent_segment_lengths,
    )


def _resolve_raw_segment_lengths(
    sequence_lengths: Optional[torch.Tensor],
    segment_lengths: Optional[Sequence[Sequence[int]]],
) -> Optional[List[List[int]]]:
    if segment_lengths is None:
        return None
    if sequence_lengths is None:
        raise ValueError("segment_lengths requires sequence_lengths")
    resolved: List[List[int]] = []
    for sample_index, one_sample in enumerate(segment_lengths):
        sample_lengths = [int(length) for length in one_sample if int(length) > 0]
        if sum(sample_lengths) != int(sequence_lengths[sample_index]):
            raise ValueError(
                f"segment_lengths={sample_lengths} do not sum to sequence_length={int(sequence_lengths[sample_index])}"
            )
        resolved.append(sample_lengths)
    return resolved


def _resolve_latent_segment_lengths(
    sequence_lengths: Optional[torch.Tensor],
    segment_lengths: Optional[Sequence[Sequence[int]]],
) -> Optional[List[List[int]]]:
    raw_segment_lengths = _resolve_raw_segment_lengths(sequence_lengths, segment_lengths)
    if raw_segment_lengths is None:
        return None
    return [[((int(length) - 1) // 4) + 1 for length in one_sample] for one_sample in raw_segment_lengths]


def _repeat_frames_for_lq_segment(segment: torch.Tensor, min_frames: int = 4) -> torch.Tensor:
    if segment.shape[2] >= min_frames:
        return segment
    repeat_times = (min_frames + segment.shape[2] - 1) // segment.shape[2]
    return segment.repeat(1, 1, repeat_times, 1, 1)[:, :, :min_frames]


def _encode_video_segments_with_vae(
    pipe,
    input_video: torch.Tensor,
    *,
    raw_segment_lengths: Optional[Sequence[Sequence[int]]],
    tiled: bool,
    tile_size,
    tile_stride,
    framewise_decoding: bool,
):
    if raw_segment_lengths is None:
        if framewise_decoding:
            return pipe.vae.encode_framewise(input_video, device=pipe.device)
        return pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

    encoded_samples: List[torch.Tensor] = []
    for sample_index, one_sample_lengths in enumerate(raw_segment_lengths):
        sample = input_video[sample_index : sample_index + 1]
        offset = 0
        sample_latents: List[torch.Tensor] = []
        for seg_len in one_sample_lengths:
            segment = sample[:, :, offset : offset + seg_len]
            offset += seg_len
            if framewise_decoding:
                encoded = pipe.vae.encode_framewise(segment, device=pipe.device)
            else:
                encoded = pipe.vae.encode(
                    segment,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
            sample_latents.append(encoded)
        encoded_samples.append(torch.cat(sample_latents, dim=2))
    return torch.cat(encoded_samples, dim=0)


def _encode_lq_segments_with_projection(
    pipe,
    lq_video: torch.Tensor,
    *,
    raw_segment_lengths: Optional[Sequence[Sequence[int]]],
):
    if raw_segment_lengths is None:
        return pipe.lq_proj_in(lq_video)

    batch_outputs: Optional[List[List[torch.Tensor]]] = None
    for sample_index, one_sample_lengths in enumerate(raw_segment_lengths):
        sample = lq_video[sample_index : sample_index + 1]
        offset = 0
        per_segment_outputs: List[List[torch.Tensor]] = []
        for seg_len in one_sample_lengths:
            segment = sample[:, :, offset : offset + seg_len]
            offset += seg_len
            # FlashVSRLQProjIn's first streaming clip only warms up its caches.
            # Single-frame image segments therefore need at least 5 repeated
            # frames so the second clip can emit one latent-time token.
            segment = _repeat_frames_for_lq_segment(segment, min_frames=5)
            encoded = pipe.lq_proj_in(segment)
            if encoded is None:
                raise ValueError(
                    f"lq_proj_in returned None for segmented input with seg_len={seg_len}; "
                    "segment-aware v5 expects at least one output token per segment."
                )
            per_segment_outputs.append(encoded)

        if batch_outputs is None:
            batch_outputs = [[] for _ in range(len(per_segment_outputs[0]))]
        for layer_index in range(len(batch_outputs)):
            batch_outputs[layer_index].append(
                torch.cat([segment_outputs[layer_index] for segment_outputs in per_segment_outputs], dim=1)
            )

    if batch_outputs is None:
        return None
    return [torch.cat(one_layer, dim=0) for one_layer in batch_outputs]


def _build_v5_sample_loss_weights(data: Dict[str, Any]) -> Optional[torch.Tensor]:
    sample_kinds = data.get("sample_kind")
    if sample_kinds is None:
        return None
    # V5 uses first-frame fairness, not sample-count fairness.
    #
    # Important consequence:
    # - A 17f video contributes roughly 5 latent-time positions.
    # - A grouped-image sample also contains 5 independent 1-frame segments.
    # - The per-sample mean MSE already averages over those 5 image segments, so
    #   each image naturally contributes ~1/5 of the grouped sample loss.
    #
    # Therefore grouped-image samples should NOT be divided by K again here.
    # An extra 1/K would underweight every image by another factor of K.
    return torch.ones(len(sample_kinds), dtype=torch.float32)


def _latent_length_from_raw_frames(num_frames: int) -> int:
    num_frames = max(1, int(num_frames))
    return ((num_frames - 1) // 4) + 1


def _concat_lq_latent_layers(
    first_layers: Optional[List[torch.Tensor]],
    second_layers: Optional[List[torch.Tensor]],
) -> Optional[List[torch.Tensor]]:
    if first_layers is None:
        return second_layers
    if second_layers is None:
        return first_layers
    if len(first_layers) != len(second_layers):
        raise ValueError(f"lq latent layer count mismatch: {len(first_layers)} vs {len(second_layers)}")
    return [torch.cat([lhs, rhs], dim=1) for lhs, rhs in zip(first_layers, second_layers)]


def _lq_latent_length_from_raw_frames(raw_frames: int) -> int:
    # LR Proj-In intentionally has no token for the uncompressed first latent.
    return max(0, (int(raw_frames) - 1) // 4)


def _align_lq_latents_to_dit_tokens(
    lq_latents: Sequence[torch.Tensor],
    expected_tokens: int,
    tokens_per_frame: int,
    raw_segment_lengths: Optional[Sequence[Sequence[int]]] = None,
) -> List[torch.Tensor]:
    if all(layer_latents.shape[1] == expected_tokens for layer_latents in lq_latents):
        return list(lq_latents)

    def align_global(layer_latents: torch.Tensor) -> torch.Tensor:
        current_tokens = layer_latents.shape[1]
        if current_tokens == expected_tokens:
            return layer_latents
        if current_tokens < expected_tokens:
            pad_tokens = expected_tokens - current_tokens
            if pad_tokens % tokens_per_frame != 0:
                raise ValueError(
                    f"Cannot align lq_latents with x tokens: x={expected_tokens}, lq={current_tokens}, "
                    f"tokens_per_frame={tokens_per_frame}"
                )
            padding = torch.zeros(
                layer_latents.shape[0],
                pad_tokens,
                layer_latents.shape[2],
                device=layer_latents.device,
                dtype=layer_latents.dtype,
            )
            return torch.cat([padding, layer_latents], dim=1)
        trim_tokens = current_tokens - expected_tokens
        if trim_tokens % tokens_per_frame != 0:
            raise ValueError(
                f"Cannot trim lq_latents to x tokens: x={expected_tokens}, lq={current_tokens}, "
                f"tokens_per_frame={tokens_per_frame}"
            )
        return layer_latents[:, trim_tokens:, :]

    if raw_segment_lengths is None:
        return [align_global(layer_latents) for layer_latents in lq_latents]

    aligned_layers: List[torch.Tensor] = []
    for layer_latents in lq_latents:
        per_sample: List[torch.Tensor] = []
        for sample_index, one_sample_raw_lengths in enumerate(raw_segment_lengths):
            sample_tokens = layer_latents[sample_index : sample_index + 1]
            offset = 0
            aligned_segments: List[torch.Tensor] = []
            for raw_frames in one_sample_raw_lengths:
                dit_frames = _latent_length_from_raw_frames(int(raw_frames))
                lq_frames = _lq_latent_length_from_raw_frames(int(raw_frames))
                dit_tokens = dit_frames * tokens_per_frame
                lq_tokens = lq_frames * tokens_per_frame
                segment = sample_tokens[:, offset : offset + lq_tokens]
                offset += lq_tokens
                if segment.shape[1] > dit_tokens:
                    segment = segment[:, segment.shape[1] - dit_tokens :]
                elif segment.shape[1] < dit_tokens:
                    pad_tokens = dit_tokens - segment.shape[1]
                    padding = torch.zeros(
                        sample_tokens.shape[0],
                        pad_tokens,
                        sample_tokens.shape[2],
                        device=sample_tokens.device,
                        dtype=sample_tokens.dtype,
                    )
                    segment = torch.cat([padding, segment], dim=1)
                aligned_segments.append(segment)
            if offset != sample_tokens.shape[1]:
                raise ValueError(
                    f"LQ segment alignment consumed {offset} tokens, but layer has {sample_tokens.shape[1]} tokens. "
                    f"raw_segment_lengths={one_sample_raw_lengths}"
                )
            per_sample.append(torch.cat(aligned_segments, dim=1))
        aligned = torch.cat(per_sample, dim=0)
        if aligned.shape[1] != expected_tokens:
            raise ValueError(f"Aligned LQ tokens mismatch: expected={expected_tokens}, got={aligned.shape[1]}")
        aligned_layers.append(aligned)
    return aligned_layers


def FlowMatchSFTLossV5(pipe: BasePipeline, loss_sample_weights: Optional[torch.Tensor] = None, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    per_element = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
    per_sample = per_element.reshape(per_element.shape[0], -1).mean(dim=1)

    if loss_sample_weights is not None:
        weights = loss_sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
        if weights.shape[0] != per_sample.shape[0]:
            raise ValueError(
                f"loss_sample_weights batch mismatch: weights={tuple(weights.shape)} per_sample={tuple(per_sample.shape)}"
            )
        loss = (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)
    else:
        loss = per_sample.mean()
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTLossV53(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    latent_segment_lengths = _resolve_latent_segment_lengths(
        inputs.get("sequence_lengths"),
        inputs.get("segment_lengths"),
    )
    if latent_segment_lengths is None:
        return FlowMatchSFTLossV5(pipe, **inputs)

    per_element = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction="none")
    per_sample_losses = []
    for sample_index, one_sample_lengths in enumerate(latent_segment_lengths):
        segment_losses = []
        offset = 0
        for seg_len in one_sample_lengths:
            seg_len = int(seg_len)
            if seg_len <= 0:
                continue
            segment_loss = per_element[sample_index : sample_index + 1, :, offset : offset + seg_len].mean()
            segment_losses.append(segment_loss)
            offset += seg_len
        if not segment_losses:
            raise ValueError(f"Sample {sample_index} has no valid latent segments for v5.3 loss")
        per_sample_losses.append(torch.stack(segment_losses).mean())

    loss = torch.stack(per_sample_losses).mean()
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def _append_flashvsr_debug(filename: str, message: str) -> None:
    debug_dir = os.environ.get("FLASHVSR_DEBUG_DIR")
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, filename), "a", encoding="utf-8") as file:
        file.write(message + "\n")


def _flashvsr_train_debug_enabled() -> bool:
    return os.environ.get("FLASHVSR_TRAIN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _tensor_debug_dir(pipe=None) -> Optional[str]:
    if pipe is not None:
        path = getattr(pipe, "debug_tensor_dump_dir", None)
        if path:
            return path
    return os.environ.get("FLASHVSR_TENSOR_DEBUG_DIR")


def _tensor_to_display_frames(video_tensor: torch.Tensor) -> List[Image.Image]:
    if video_tensor.ndim == 5:
        video_tensor = video_tensor[0]
    if video_tensor.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] or [B,T,C,H,W], got {tuple(video_tensor.shape)}")
    tensor = video_tensor.detach().cpu().float()
    if tensor.shape[1] not in (1, 3):
        raise ValueError(f"Expected channel dimension 1 or 3, got {tuple(tensor.shape)}")
    if tensor.min() < 0:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in tensor:
        if frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _dump_tensor_preview_once(
    key: str,
    tensor: Optional[torch.Tensor] = None,
    pipe=None,
    extra: Optional[Dict[str, Any]] = None,
    fps: int = 8,
) -> None:
    if key in _TENSOR_DEBUG_REPORTED:
        return
    debug_dir = _tensor_debug_dir(pipe)
    if not debug_dir:
        return
    print(f"[tensor_dump] begin key={key} dir={debug_dir}", flush=True)
    os.makedirs(debug_dir, exist_ok=True)
    payload: Dict[str, Any] = {}
    if tensor is not None:
        payload.update(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "min": float(tensor.detach().float().min().item()),
                "max": float(tensor.detach().float().max().item()),
                "mean": float(tensor.detach().float().mean().item()),
                "std": float(tensor.detach().float().std().item()),
            }
        )
    if extra:
        payload.update(extra)
    try:
        with open(os.path.join(debug_dir, f"{key}.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        if tensor is not None:
            torch.save(tensor.detach().cpu(), os.path.join(debug_dir, f"{key}.pt"))
            frames = _tensor_to_display_frames(tensor)
            save_video(
                frames,
                os.path.join(debug_dir, f"{key}.mp4"),
                fps=fps,
                quality=5,
                ffmpeg_params=["-pix_fmt", "yuv420p"],
            )
        print(f"[tensor_dump] done key={key}", flush=True)
    except Exception as error:
        print(f"[tensor_dump] error key={key} error={error}", flush=True)
        with open(os.path.join(debug_dir, f"{key}.error.txt"), "w", encoding="utf-8") as file:
            file.write(str(error))
    _TENSOR_DEBUG_REPORTED.add(key)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=dim) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x):
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


class FlashVSRLQProjIn(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num=1, zero_init_output=True, temporal_mode: str = "streaming"):
        super().__init__()
        if temporal_mode not in ("streaming", "nonstreaming", "nonstreaming_aligned"):
            raise ValueError(f"Unsupported lq_proj temporal_mode={temporal_mode!r}.")
        self.temporal_mode = temporal_mode
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])
        if zero_init_output:
            self.zero_init_output_projection()
        self.clear_cache()

    def zero_init_output_projection(self):
        for layer in self.linear_layers:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, video):
        if self.temporal_mode in ("nonstreaming", "nonstreaming_aligned"):
            return self.forward_nonstreaming(video)
        self.clear_cache()
        t = video.shape[2]
        iterations = 1 + (t - 1) // 4
        first_frame = video[:, :, :1].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        outputs = None
        for clip_idx in range(iterations):
            cur = self.stream_forward(video[:, :, clip_idx * 4 : (clip_idx + 1) * 4])
            if cur is None:
                continue
            if outputs is None:
                outputs = cur
            else:
                for layer_idx in range(len(outputs)):
                    outputs[layer_idx] = torch.cat([outputs[layer_idx], cur[layer_idx]], dim=1)
        return outputs

    def forward_nonstreaming(self, video):
        self.clear_cache()
        first_frame = video[:, :, :1].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)
        x = self.pixel_shuffle(video)
        x = self.conv1(x, None)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x, None)
        x = self.norm2(x)
        x = self.act2(x)
        # The legacy nonstreaming mode mimics FlashVSR's streaming projector by
        # dropping the warm-up output: 89 raw frames -> 22 LQ latent frames.
        # The aligned mode keeps it so Stage-1 non-streaming SR matches WAN VAE:
        # 89 raw frames -> 23 LQ latent frames, 17 -> 5, image pseudo-video 5 -> 2.
        if self.temporal_mode == "nonstreaming" and x.shape[2] > 0:
            x = x[:, :, 1:]
        x = rearrange(x, "b c f h w -> b (f h w) c")
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](x))
        return outputs

    def clear_cache(self):
        self.cache = {"conv1": None, "conv2": None}
        self.clip_idx = 0

    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:].clone()
            self.cache["conv2"] = cache2_x
            self.clip_idx += 1
            return None
        x = self.pixel_shuffle(video_clip)
        cache1_x = x[:, :, -CACHE_T:].clone()
        x = self.conv1(x, self.cache["conv1"])
        self.cache["conv1"] = cache1_x
        x = self.norm1(x)
        x = self.act1(x)
        cache2_x = x[:, :, -CACHE_T:].clone()
        x = self.conv2(x, self.cache["conv2"])
        self.cache["conv2"] = cache2_x
        x = self.norm2(x)
        x = self.act2(x)
        x = rearrange(x, "b c f h w -> b (f h w) c")
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](x))
        self.clip_idx += 1
        return outputs


def _build_release_style_lq_latents(lq_proj_in: FlashVSRLQProjIn, lq_video: torch.Tensor):
    lq_proj_in.clear_cache()
    first_frame = lq_video[:, :, :1].repeat(1, 1, 3, 1, 1)
    lq_video = torch.cat([first_frame, lq_video], dim=2)
    total_frames = int(lq_video.shape[2])
    outputs = None
    for start in range(0, total_frames, 4):
        cur = lq_proj_in.stream_forward(lq_video[:, :, start : start + 4])
        if cur is None:
            continue
        if outputs is None:
            outputs = cur
        else:
            for layer_idx in range(len(outputs)):
                outputs[layer_idx] = torch.cat([outputs[layer_idx], cur[layer_idx]], dim=1)
    return outputs


class FlashVSRUnit_FixedPrompt(PipelineUnit):
    def __init__(self):
        super().__init__(output_params=("context",))

    def process(self, pipe):
        if pipe.fixed_prompt_tensor is None:
            if pipe.prompt_tensor_path is None:
                raise ValueError("prompt_tensor_path is required for FlashVSR Stage 1 training.")
            pipe.fixed_prompt_tensor = torch.load(pipe.prompt_tensor_path, map_location="cpu")
        context = pipe.fixed_prompt_tensor.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"context": context}


class WanFixedPromptEmbeddedUnit(PipelineUnit):
    def __init__(self):
        super().__init__(output_params=("embedded_context",))

    def process(self, pipe):
        if pipe.fixed_prompt_tensor is None:
            if pipe.prompt_tensor_path is None:
                raise ValueError("prompt_tensor_path is required for fixed-prompt validation.")
            pipe.fixed_prompt_tensor = torch.load(pipe.prompt_tensor_path, map_location="cpu")
        raw_context = pipe.fixed_prompt_tensor.to(device=pipe.device, dtype=pipe.torch_dtype)
        embedded_context = pipe.dit.text_embedding(raw_context)
        return {"embedded_context": embedded_context}


class WanVideoUnit_InputVideoEmbedderV5(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
                "framewise_decoding",
                "sequence_lengths",
                "segment_lengths",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
        framewise_decoding,
        sequence_lengths=None,
        segment_lengths=None,
    ):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        raw_segment_lengths = _resolve_raw_segment_lengths(sequence_lengths, segment_lengths)
        input_latents = _encode_video_segments_with_vae(
            pipe,
            input_video,
            raw_segment_lengths=raw_segment_lengths,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            framewise_decoding=framewise_decoding,
        )
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents}


class WanTextPromptLQPipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        tokenizer_config=None,
        lq_proj_layer_num=None,
        lq_proj_temporal_mode="streaming",
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=tokenizer_config,
        )
        pipe.__class__ = WanTextPromptLQPipeline
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            FlashVSRUnit_LQVideoEmbedder(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage1_model_fn
        pipe.compilable_models = ["dit"]
        pipe.lq_proj_scale = 1.0
        effective_lq_proj_layers = 1 if lq_proj_layer_num is None else int(lq_proj_layer_num)
        pipe.lq_proj_in = FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=effective_lq_proj_layers,
            zero_init_output=False,
            temporal_mode=lq_proj_temporal_mode,
        ).to(device=device, dtype=torch_dtype)
        return pipe

    @torch.no_grad()
    def infer_from_lq_text(
        self,
        prompt: str,
        negative_prompt: str,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_shared = {
            "input_video": None,
            "lq_video": lq_video,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": framewise_decoding,
            "vace_reference_image": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "lq_proj_scale": self.lq_proj_scale,
        }
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        if "latents" not in inputs_shared:
            inputs_shared["latents"] = inputs_shared["noise"]

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )

        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(
                inputs_shared["latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class FlashVSRUnit_LQVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("lq_video", "height", "width", "sequence_lengths", "segment_lengths"),
            output_params=("lq_latents",),
            onload_model_names=("lq_proj_in",),
        )

    def process(self, pipe, lq_video, height, width, sequence_lengths=None, segment_lengths=None):
        if lq_video is None:
            return {}
        if torch.is_tensor(lq_video):
            raw_lq_for_dump = lq_video.permute(0, 2, 1, 3, 4).contiguous()
            _dump_tensor_preview_once("01_input_lq_tensor", raw_lq_for_dump, pipe=pipe)
        if torch.is_tensor(lq_video):
            lq_video = pipe.preprocess_video(lq_video)
        else:
            resized = [frame.resize((width, height)) if frame.size != (width, height) else frame for frame in lq_video]
            lq_video = pipe.preprocess_video(resized)
        _dump_tensor_preview_once(
            "02_preprocessed_lq_tensor",
            lq_video.permute(0, 2, 1, 3, 4).contiguous(),
            pipe=pipe,
        )
        lq_input = lq_video.to(device=pipe.device, dtype=pipe.torch_dtype)
        raw_segment_lengths = _resolve_raw_segment_lengths(sequence_lengths, segment_lengths)
        lq_latents = _encode_lq_segments_with_projection(
            pipe,
            lq_input,
            raw_segment_lengths=raw_segment_lengths,
        )
        if lq_latents is not None:
            _dump_tensor_preview_once(
                "03_lq_proj_latents",
                None,
                pipe=pipe,
                extra={
                    "num_layers": len(lq_latents),
                    "layer_shapes": [list(layer.shape) for layer in lq_latents[: min(4, len(lq_latents))]],
                    "layer0_dtype": str(lq_latents[0].dtype),
                    "layer0_min": float(lq_latents[0].detach().float().min().item()),
                    "layer0_max": float(lq_latents[0].detach().float().max().item()),
                    "layer0_mean": float(lq_latents[0].detach().float().mean().item()),
                },
            )
        return {"lq_latents": lq_latents}


class WanVideoUnit_ShapeCheckerV53(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "video_num_frames", "image_num_frames"),
            output_params=("height", "width", "video_num_frames", "image_num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, video_num_frames, image_num_frames):
        video_h, video_w, video_num_frames = pipe.check_resize_height_width(height, width, int(video_num_frames))
        image_h, image_w, image_num_frames = pipe.check_resize_height_width(height, width, int(image_num_frames))
        if (video_h, video_w) != (image_h, image_w):
            raise ValueError(
                f"v5.3 branch-aware shape checker produced mismatched sizes: "
                f"video={(video_h, video_w)} image={(image_h, image_w)}"
            )
        return {
            "height": video_h,
            "width": video_w,
            "video_num_frames": int(video_num_frames),
            "image_num_frames": int(image_num_frames),
        }


class WanVideoUnit_NoiseInitializerV53(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "video_num_frames", "image_num_frames", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: WanVideoPipeline, height, width, video_num_frames, image_num_frames, seed, rand_device):
        length = _latent_length_from_raw_frames(video_num_frames) + _latent_length_from_raw_frames(image_num_frames)
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedderV53(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "image_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "framewise_decoding",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        input_video,
        image_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        framewise_decoding,
    ):
        if input_video is None or image_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        image_video = pipe.preprocess_video(image_video)
        if framewise_decoding:
            input_latents_video = pipe.vae.encode_framewise(input_video, device=pipe.device)
            input_latents_image = pipe.vae.encode_framewise(image_video, device=pipe.device)
        else:
            input_latents_video = pipe.vae.encode(
                input_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents_image = pipe.vae.encode(
                image_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = torch.cat([input_latents_video, input_latents_image], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents}


class FlashVSRUnit_LQVideoEmbedderV53(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("lq_video", "image_lq_video", "height", "width"),
            output_params=("lq_latents",),
            onload_model_names=("lq_proj_in",),
        )

    def process(self, pipe, lq_video, image_lq_video, height, width):
        if lq_video is None or image_lq_video is None:
            return {}
        if torch.is_tensor(lq_video):
            lq_video = pipe.preprocess_video(lq_video)
        else:
            resized = [frame.resize((width, height)) if frame.size != (width, height) else frame for frame in lq_video]
            lq_video = pipe.preprocess_video(resized)
        if torch.is_tensor(image_lq_video):
            image_lq_video = pipe.preprocess_video(image_lq_video)
        else:
            resized = [frame.resize((width, height)) if frame.size != (width, height) else frame for frame in image_lq_video]
            image_lq_video = pipe.preprocess_video(resized)
        video_latents = pipe.lq_proj_in(lq_video.to(device=pipe.device, dtype=pipe.torch_dtype))
        image_latents = pipe.lq_proj_in(image_lq_video.to(device=pipe.device, dtype=pipe.torch_dtype))
        lq_latents = _concat_lq_latent_layers(video_latents, image_latents)
        return {"lq_latents": lq_latents}


def flashvsr_stage1_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    sequence_lengths: Optional[torch.Tensor] = None,
    segment_lengths: Optional[Sequence[Sequence[int]]] = None,
    **kwargs,
):
    global _FLASHVSR_BLOCK_BRANCH_REPORTED
    batch_size = latents.shape[0]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(
            f"Context batch size mismatch: context={tuple(context.shape)}, latents={tuple(latents.shape)}"
        )
    context = dit.text_embedding(context)

    latent_segment_lengths = _resolve_latent_segment_lengths(sequence_lengths, segment_lengths)
    x, (f, h, w) = _patchify_tokens_with_segments(
        dit,
        latents,
        latent_segment_lengths=latent_segment_lengths,
    )
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)
    per_sample_token_lengths = _resolve_per_sample_token_lengths(
        dit=dit,
        x=x,
        f=f,
        h=h,
        w=w,
        sequence_lengths=sequence_lengths,
        segment_lengths=segment_lengths,
    )

    if lq_latents is not None:
        tokens_per_frame = h * w
        expected_tokens = x.shape[1]
        raw_segment_lengths = _resolve_raw_segment_lengths(sequence_lengths, segment_lengths)
        lq_latents = _align_lq_latents_to_dit_tokens(
            lq_latents,
            expected_tokens=expected_tokens,
            tokens_per_frame=tokens_per_frame,
            raw_segment_lengths=raw_segment_lengths,
        )
        _dump_tensor_preview_once(
            "04_model_token_alignment",
            None,
            extra={
                "x_shape": list(x.shape),
                "grid": {"f": int(f), "h": int(h), "w": int(w)},
                "expected_tokens": int(expected_tokens),
                "tokens_per_frame": int(tokens_per_frame),
                "raw_segment_lengths": raw_segment_lengths,
                "aligned_lq_shape": list(lq_latents[0].shape) if lq_latents else None,
            },
        )

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + (lq_latents[block_id] * lq_proj_scale)
        if not _FLASHVSR_BLOCK_BRANCH_REPORTED:
            _FLASHVSR_BLOCK_BRANCH_REPORTED = True
            rank = os.environ.get("RANK", "?")
            local_rank = os.environ.get("LOCAL_RANK", "?")
            branch = "gradient_checkpoint" if dit.training else "direct_block"
            message = (
                f"[flashvsr_block] rank={rank} local_rank={local_rank} "
                f"branch={branch} dit_training={dit.training} "
                f"use_gradient_checkpointing={use_gradient_checkpointing} "
                f"use_gradient_checkpointing_offload={use_gradient_checkpointing_offload}"
            )
            if _flashvsr_train_debug_enabled():
                print(message, flush=True)
            _append_flashvsr_debug("flashvsr_block_branches.log", message)
        if dit.training:
            if per_sample_token_lengths is not None:
                def block_forward(
                    hidden_states,
                    block=block,
                    context=context,
                    t_mod=t_mod,
                    freqs=freqs,
                    per_sample_token_lengths=per_sample_token_lengths,
                ):
                    return block(
                        hidden_states,
                        context,
                        t_mod,
                        freqs,
                        per_sample_token_lengths=per_sample_token_lengths,
                    )
            else:
                def block_forward(hidden_states, block=block, context=context, t_mod=t_mod, freqs=freqs):
                    return block(hidden_states, context, t_mod, freqs)
            x = gradient_checkpoint_forward(
                block_forward,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
            )
        else:
            if per_sample_token_lengths is not None:
                x = block(x, context, t_mod, freqs, per_sample_token_lengths=per_sample_token_lengths)
            else:
                x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


def flashvsr_stage1_fixed_prompt_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    embedded_context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    sequence_lengths: Optional[torch.Tensor] = None,
    segment_lengths: Optional[Sequence[Sequence[int]]] = None,
    **kwargs,
):
    batch_size = latents.shape[0]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = embedded_context
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.shape[0] == 1 and batch_size > 1:
        context = context.expand(batch_size, -1, -1)
    elif context.shape[0] != batch_size:
        raise ValueError(
            f"Embedded context batch size mismatch: context={tuple(context.shape)}, latents={tuple(latents.shape)}"
        )

    latent_segment_lengths = _resolve_latent_segment_lengths(sequence_lengths, segment_lengths)
    x, (f, h, w) = _patchify_tokens_with_segments(
        dit,
        latents,
        latent_segment_lengths=latent_segment_lengths,
    )
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)
    per_sample_token_lengths = _resolve_per_sample_token_lengths(
        dit=dit,
        x=x,
        f=f,
        h=h,
        w=w,
        sequence_lengths=sequence_lengths,
        segment_lengths=segment_lengths,
    )

    if lq_latents is not None:
        tokens_per_frame = h * w
        expected_tokens = x.shape[1]
        raw_segment_lengths = _resolve_raw_segment_lengths(sequence_lengths, segment_lengths)
        lq_latents = _align_lq_latents_to_dit_tokens(
            lq_latents,
            expected_tokens=expected_tokens,
            tokens_per_frame=tokens_per_frame,
            raw_segment_lengths=raw_segment_lengths,
        )

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + (lq_latents[block_id] * lq_proj_scale)
        if dit.training:
            if per_sample_token_lengths is not None:
                def block_forward(
                    hidden_states,
                    block=block,
                    context=context,
                    t_mod=t_mod,
                    freqs=freqs,
                    per_sample_token_lengths=per_sample_token_lengths,
                ):
                    return block(
                        hidden_states,
                        context,
                        t_mod,
                        freqs,
                        per_sample_token_lengths=per_sample_token_lengths,
                    )
            else:
                def block_forward(hidden_states, block=block, context=context, t_mod=t_mod, freqs=freqs):
                    return block(hidden_states, context, t_mod, freqs)
            x = gradient_checkpoint_forward(
                block_forward,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x,
            )
        else:
            if per_sample_token_lengths is not None:
                x = block(x, context, t_mod, freqs, per_sample_token_lengths=per_sample_token_lengths)
            else:
                x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))


class FlashVSRStage1Pipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        prompt_tensor_path=None,
        lq_proj_checkpoint=None,
        lq_proj_layer_num=None,
        zero_init_lq_proj_in=True,
        lq_proj_temporal_mode="streaming",
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=None,
        )
        pipe.__class__ = FlashVSRStage1Pipeline
        pipe.prompt_tensor_path = prompt_tensor_path
        pipe.fixed_prompt_tensor = None
        pipe.units = [
            WanVideoUnit_ShapeCheckerV53(),
            WanVideoUnit_NoiseInitializerV53(),
            FlashVSRUnit_FixedPrompt(),
            WanVideoUnit_InputVideoEmbedderV53(),
            FlashVSRUnit_LQVideoEmbedderV53(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage1_model_fn
        pipe.compilable_models = ["dit"]
        pipe.debug_tensor_dump_dir = None
        pipe.lq_proj_scale = 1.0
        effective_lq_proj_layers = 1 if lq_proj_layer_num is None else int(lq_proj_layer_num)
        pipe.lq_proj_in = FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=effective_lq_proj_layers,
            zero_init_output=zero_init_lq_proj_in and lq_proj_checkpoint is None,
            temporal_mode=lq_proj_temporal_mode,
        ).to(device=device, dtype=torch_dtype)
        if lq_proj_checkpoint is not None:
            state_dict = torch.load(lq_proj_checkpoint, map_location="cpu")
            pipe.lq_proj_in.load_state_dict(state_dict, strict=True)
        return pipe

    @torch.no_grad()
    def infer_from_lq(
        self,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        num_inference_steps: int = 10,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
        progress_bar_cmd=tqdm,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_shared = {
            "input_video": None,
            "lq_video": lq_video,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": framewise_decoding,
            "vace_reference_image": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "lq_proj_scale": self.lq_proj_scale,
        }
        inputs_posi = {}
        inputs_nega = {}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps, disable=True)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanFixedPromptFlashVSRStage1Pipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        prompt_tensor_path=None,
        lq_proj_layer_num=None,
        lq_proj_temporal_mode="streaming",
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=None,
        )
        pipe.__class__ = WanFixedPromptFlashVSRStage1Pipeline
        pipe.prompt_tensor_path = prompt_tensor_path
        pipe.fixed_prompt_tensor = None
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanFixedPromptEmbeddedUnit(),
            FlashVSRUnit_LQVideoEmbedder(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage1_fixed_prompt_model_fn
        pipe.compilable_models = ["dit"]
        pipe.lq_proj_scale = 1.0
        effective_lq_proj_layers = 1 if lq_proj_layer_num is None else int(lq_proj_layer_num)
        pipe.lq_proj_in = FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=effective_lq_proj_layers,
            zero_init_output=False,
            temporal_mode=lq_proj_temporal_mode,
        ).to(device=device, dtype=torch_dtype)
        return pipe

    @torch.no_grad()
    def infer_from_lq(
        self,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_shared = {
            "input_video": None,
            "lq_video": lq_video,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": framewise_decoding,
            "vace_reference_image": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "lq_proj_scale": self.lq_proj_scale,
        }
        inputs_posi = {}
        inputs_nega = {}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        if "latents" not in inputs_shared:
            inputs_shared["latents"] = inputs_shared["noise"]

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )

        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(
                inputs_shared["latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def flashvsr_stage1_export(state_dict):
    converted = {}
    for name, param in state_dict.items():
        if name.startswith("pipe.dit."):
            converted[name[len("pipe.dit.") :]] = param
        elif name.startswith("pipe.lq_proj_in."):
            converted["lq_proj_in." + name[len("pipe.lq_proj_in.") :]] = param
        else:
            converted[name] = param
    return converted


def flashvsr_stage1_split_exported_state(state_dict):
    exported_state = flashvsr_stage1_export(state_dict)
    lq_proj_state = {}
    lora_state = {}
    other_state = {}
    for key, value in exported_state.items():
        if key.startswith("lq_proj_in."):
            lq_proj_state[key[len("lq_proj_in.") :]] = value
        elif "lora_" in key:
            lora_state[key] = value
        else:
            other_state[key] = value
    return lq_proj_state, lora_state, other_state


class FlashVSRStage1TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        prompt_tensor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=384,
        lora_checkpoint=None,
        lq_proj_checkpoint=None,
        resume_stage1_checkpoint=None,
        lq_proj_layer_num=None,
        lq_proj_scale: float = 1.0,
        lq_proj_temporal_mode: str = "streaming",
        zero_init_lq_proj_in=True,
        freeze_lq_proj_in: bool = False,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        image_video_joint_packed: bool = False,
        debug_tensor_dump_dir=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is required for practical memory use. Forcing it on.")
            use_gradient_checkpointing = True

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        self.pipe = FlashVSRStage1Pipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            prompt_tensor_path=prompt_tensor_path,
            lq_proj_checkpoint=lq_proj_checkpoint,
            lq_proj_layer_num=lq_proj_layer_num,
            zero_init_lq_proj_in=zero_init_lq_proj_in,
            lq_proj_temporal_mode=lq_proj_temporal_mode,
        )
        if image_video_joint_packed:
            self.pipe.dit = build_joint_wan_from_existing_v5(self.pipe.dit)
        self.pipe.lq_proj_scale = float(lq_proj_scale)
        self.pipe.debug_tensor_dump_dir = debug_tensor_dump_dir
        self.pipe = self.split_pipeline_units("sft", self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            task="sft",
        )
        if resume_stage1_checkpoint is not None:
            if lora_base_model is None:
                raise ValueError("resume_stage1_checkpoint requires lora_base_model to be enabled for v2 warm-start.")
            self._load_stage1_resume_checkpoint(
                checkpoint_path=resume_stage1_checkpoint,
                lora_base_model=lora_base_model,
            )
        if freeze_lq_proj_in:
            for param in self.pipe.lq_proj_in.parameters():
                param.requires_grad = False
            self.pipe.lq_proj_in.eval()
            print("freeze_lq_proj_in=True: lq_proj_in parameters frozen.", flush=True)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.image_video_joint_packed = image_video_joint_packed

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        export_names = self.trainable_param_names()
        export_names.update(
            name for name, _ in self.named_parameters() if name.startswith("pipe.lq_proj_in.")
        )
        state_dict = {name: param for name, param in state_dict.items() if name in export_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict

    def _load_stage1_resume_checkpoint(self, checkpoint_path: str, lora_base_model: str) -> None:
        state_dict = load_state_dict(checkpoint_path, device="cpu")
        lq_proj_state, lora_state, _ = flashvsr_stage1_split_exported_state(state_dict)
        if not lq_proj_state and not lora_state:
            raise ValueError(
                f"resume_stage1_checkpoint={checkpoint_path} does not contain FlashVSR stage1 lq_proj_in or LoRA weights."
            )
        if lq_proj_state:
            load_result = self.pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
            print(
                f"Stage1 resume loaded lq_proj_in from {checkpoint_path}, "
                f"keys={len(lq_proj_state)}, missing={len(missing)}, unexpected={len(unexpected)}",
                flush=True,
            )
            if unexpected:
                print(f"Unexpected lq_proj_in keys: {unexpected}", flush=True)
        if lora_state:
            lora_model = getattr(self.pipe, lora_base_model)
            mapped_lora_state = self.mapping_lora_state_dict(lora_state)
            load_result = lora_model.load_state_dict(mapped_lora_state, strict=False)
            print(
                f"Stage1 resume loaded LoRA from {checkpoint_path}, "
                f"keys={len(mapped_lora_state)}, missing={len(load_result[0])}, unexpected={len(load_result[1])}",
                flush=True,
            )
            if len(load_result[1]) > 0:
                print(f"Warning, resume LoRA key mismatch! Unexpected keys: {load_result[1]}", flush=True)

    def _build_branch_inputs(self, video, lq_video, data):
        if not torch.is_tensor(video) or not torch.is_tensor(data["image_video"]):
            raise ValueError("v5.3 branch-aware inputs expect tensor-collated video/image branches.")
        if video.ndim != 5 or data["image_video"].ndim != 5:
            raise ValueError(
                f"Unsupported v5.3 branch tensor shapes: video={tuple(video.shape)} image={tuple(data['image_video'].shape)}"
            )
        height = int(video.shape[-2])
        width = int(video.shape[-1])
        video_num_frames = int(video.shape[1])
        image_num_frames = int(data["image_video"].shape[1])
        inputs_shared = {
            "input_video": video,
            "image_video": data["image_video"],
            "lq_video": lq_video,
            "image_lq_video": data["image_lq_video"],
            "height": height,
            "width": width,
            "video_num_frames": video_num_frames,
            "image_num_frames": image_num_frames,
            "cfg_scale": 1.0,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "framewise_decoding": False,
            "vace_reference_image": None,
            "seed": 0,
            "lq_proj_scale": self.pipe.lq_proj_scale,
        }
        if "sequence_lengths" in data:
            inputs_shared["sequence_lengths"] = data["sequence_lengths"]
        if "segment_lengths" in data:
            inputs_shared["segment_lengths"] = data["segment_lengths"]
        return inputs_shared, {}, {}

    def get_pipeline_inputs(self, data):
        if torch.is_tensor(data["video"]):
            _dump_tensor_preview_once("00_input_hr_tensor", data["video"], pipe=self.pipe)
        if not torch.is_tensor(data["video"]) or not torch.is_tensor(data["image_video"]):
            raise ValueError("v5.3 expects tensor-collated video and image branches.")
        if data["video"].ndim != 5 or data["image_video"].ndim != 5:
            raise ValueError(
                f"v5.3 expects [B,T,C,H,W] tensors, got video={tuple(data['video'].shape)} image={tuple(data['image_video'].shape)}"
            )
        if data["lq_video"].ndim != 5 or data["image_lq_video"].ndim != 5:
            raise ValueError(
                f"v5.3 expects [B,T,C,H,W] LQ tensors, got lq_video={tuple(data['lq_video'].shape)} image_lq={tuple(data['image_lq_video'].shape)}"
            )
        if data["video"].shape[0] != data["image_video"].shape[0]:
            raise ValueError(
                f"v5.3 batch mismatch, video={tuple(data['video'].shape)} image={tuple(data['image_video'].shape)}"
            )
        if data["video"].shape[2:] != data["image_video"].shape[2:]:
            raise ValueError(
                f"v5.3 branch channel/spatial mismatch, video={tuple(data['video'].shape)} image={tuple(data['image_video'].shape)}"
            )
        if data["lq_video"].shape[2:] != data["image_lq_video"].shape[2:]:
            raise ValueError(
                f"v5.3 LQ branch channel/spatial mismatch, lq_video={tuple(data['lq_video'].shape)} image_lq={tuple(data['image_lq_video'].shape)}"
            )

        video_frames = int(data["video"].shape[1])
        image_frames = int(data["image_video"].shape[1])
        batch_size = int(data["video"].shape[0])

        branch_data = dict(data)
        branch_data["sequence_lengths"] = torch.full((batch_size,), video_frames + image_frames, dtype=torch.long)
        branch_data["segment_lengths"] = [[video_frames, image_frames] for _ in range(batch_size)]
        return self._build_branch_inputs(branch_data["video"], branch_data["lq_video"], branch_data)

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        merged_inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            merged_inputs = self.pipe.unit_runner(unit, self.pipe, *merged_inputs)

        return FlowMatchSFTLossV53(self.pipe, **merged_inputs[0], **merged_inputs[1])


def flashvsr_parser():
    parser = argparse.ArgumentParser(description="FlashVSR Stage 1 LoRA training scaffold.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
    parser.add_argument("--prompt_tensor_path", type=str, default=None, help="Path to fixed prompt tensor.")
    parser.add_argument("--lq_proj_checkpoint", type=str, default=None, help="Optional path to initialize lq_proj_in.")
    parser.add_argument("--resume_stage1_checkpoint", type=str, default=None, help="Warm-start from a mixed stage1 step checkpoint containing both lq_proj_in and LoRA weights.")
    parser.add_argument("--lq_proj_layer_num", type=int, default=1, help="Number of linear projection heads in lq_proj_in. Defaults to 1.")
    parser.add_argument("--lq_proj_scale", type=float, default=1.0, help="Fixed multiplicative scale applied to lq_proj_in latents before adding to x.")
    parser.add_argument(
        "--lq_proj_temporal_mode",
        type=str,
        default="streaming",
        choices=("streaming", "nonstreaming", "nonstreaming_aligned"),
        help=(
            "LR projector temporal path. 'streaming' matches FlashVSR chunk/cache behavior; "
            "'nonstreaming' runs full 3D convs and drops the warm-up output; "
            "'nonstreaming_aligned' keeps the warm-up output so LQ tokens match WAN VAE latent time."
        ),
    )
    parser.add_argument("--zero_init_lq_proj_in", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True, help="Zero-initialize lq_proj_in output projection so step-0 keeps base-model behavior.")
    parser.add_argument("--freeze_lq_proj_in", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=False, help="Freeze lq_proj_in parameters during training.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--dataset_mode", type=str, default="unified", choices=("unified", "streaming", "parquet_v2", "tar_v3", "tar_v5", "tar_v53"), help="Dataset backend.")
    parser.add_argument("--internal_url", type=str, default=None, help="Video root/prefix for streaming mode.")
    parser.add_argument("--metadata_url", type=str, default=None, help="Optional parquet root/path for metadata-driven streaming mode.")
    parser.add_argument("--metadata_source", type=str, default="auto", choices=("auto", "storymotion", "takano"), help="Parquet adapter type.")
    parser.add_argument("--max_parquet_records", type=int, default=None, help="Optional limit for loaded parquet rows.")
    parser.add_argument("--min_overall_score", type=float, default=None, help="Optional storymotion quality filter.")
    parser.add_argument("--require_qwen35_parse_success", default=False, action="store_true", help="Keep only rows with parsed qwen output.")
    parser.add_argument("--image_internal_url", type=str, default=None, help="Optional image root/prefix for pseudo-video mixing.")
    parser.add_argument("--image_dataset_prob", type=float, default=0.0, help="Probability of drawing pseudo-video image samples in streaming mode.")
    parser.add_argument("--image_metadata_url", type=str, default=None, help="Image metadata parquet root/path for parquet_v2 mode.")
    parser.add_argument("--image_as_single_frame", default=True, action="store_true", help="Treat image data as f=1 samples in parquet_v2 mode.")
    parser.add_argument("--takano_dataset_prob", type=float, default=None, help="Explicit sampling probability for Takano in parquet_v2 mode.")
    parser.add_argument("--yubari_dataset_prob", type=float, default=None, help="Explicit sampling probability for Yubari in parquet_v2 mode.")
    parser.add_argument("--yubari_video_tar_url", type=str, default=None, help="Yubari video root for parquet_v2 mode.")
    parser.add_argument("--takano_video_tar_url", type=str, default=None, help="Takano video root for tar_v5/tar_v53 mode.")
    parser.add_argument("--image_tar_url", type=str, default=None, help="Preferred image source path for tar_v3/tar_v5/tar_v53 mode. Supports tar roots or txt manifests of image paths.")
    parser.add_argument("--picked17k_image_tar_url", type=str, default=None, help="Legacy alias of --image_tar_url. Kept for backward compatibility.")
    parser.add_argument("--image_branch_num_frames", type=int, default=None, help="Optional raw frame count for the v5.3 image pseudo-video branch. Defaults to latent-time-equivalent length from --num_frames.")
    parser.add_argument("--picked17k_dataset_prob", type=float, default=0.1, help="Sampling probability of picked image source in tar_v3/tar_v5 mode.")
    parser.add_argument("--yubari_video_prob", type=float, default=None, help="Relative sampling probability of Yubari inside the video branch for tar_v5/tar_v53 mode.")
    parser.add_argument("--takano_video_prob", type=float, default=None, help="Relative sampling probability of Takano inside the video branch for tar_v5/tar_v53 mode.")
    parser.add_argument("--yubari_shard_start", type=int, default=None, help="Optional Yubari shard range start.")
    parser.add_argument("--yubari_shard_end", type=int, default=None, help="Optional Yubari shard range end.")
    parser.add_argument("--max_yubari_records", type=int, default=None, help="Optional limit for Yubari loaded records.")
    parser.add_argument("--media_cache_dir", type=str, default=None, help="Optional local cache dir for downloaded Takano/image media files in parquet_v2 mode.")
    parser.add_argument("--parquet_cache_dir", type=str, default=None, help="Optional local cache dir for downloaded parquet shards in parquet_v2 mode.")
    parser.add_argument("--parquet_prewarm_files_per_source", type=int, default=8, help="When parquet cache is enabled, let each node local_rank=0 prewarm this many parquet shards per source before the rest of the ranks proceed.")
    parser.add_argument("--image_video_joint_packed", default=False, action="store_true", help="Enable packed varlen joint attention for mixed image/video batches.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride for streaming-mode video sampling.")
    parser.add_argument("--max_source_frames", type=int, default=160, help="Maximum decoded source frames per raw video sample.")
    parser.add_argument("--enable_degradation", default=False, action="store_true", help="Enable online HR->LQ degradation in streaming mode.")
    parser.add_argument("--degradation_config_path", type=str, default=None, help="Path to RealESRGAN/RealBasicVSR-style degradation config.")
    parser.add_argument("--gt_sharpen", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True, help="USMGT-only: sharpen GT before both VAE target and LQ degradation.")
    parser.add_argument("--gt_sharpen_backend", type=str, default="opencv", choices=("opencv", "torch"), help="USMGT-only: OpenCV CPU or torch backend.")
    parser.add_argument("--gt_sharpen_device", type=str, default=None, help="USMGT-only: device for torch GT sharpening, e.g. auto/cuda/cuda:N/cpu.")
    parser.add_argument("--degradation_device", type=str, default=None, help="USMGT-only: online degradation device, e.g. auto/cuda/cuda:N/cpu.")
    parser.add_argument("--degradation_seed", type=int, default=None, help="Optional seed for deterministic clip degradation.")
    parser.add_argument("--hq_prefix_frames", type=int, default=0, help="Keep the first N control frames as HQ before degradation replacement.")
    parser.add_argument("--control_dropout_prob", type=float, default=0.0, help="Probability of replacing control video with zeros.")
    parser.add_argument("--shuffle_buffer", type=int, default=100, help="Shuffle buffer size for TAR streaming.")
    parser.add_argument("--global_seed", type=int, default=None, help="Global seed for dataset order, clip sampling and degradation.")
    parser.add_argument("--validation_num_samples", type=int, default=0, help="Number of fixed training samples used for online validation.")
    parser.add_argument("--validation_num_inference_steps", type=int, default=10, help="Inference steps for online validation videos.")
    parser.add_argument("--validation_fps", type=int, default=8, help="FPS for saved validation videos.")
    parser.add_argument("--validation_prompt_file", type=str, default=None, help="Optional text prompt file for pure Wan-text validation baseline.")
    parser.add_argument("--validation_negative_prompt", type=str, default="", help="Negative prompt used by Wan-text validation baseline.")
    parser.add_argument("--validation_cfg_scale", type=float, default=1.0, help="CFG scale used for validation.")
    parser.add_argument("--validation_use_wan_text_baseline", default=False, action="store_true", help="Use pure Wan text-to-video validation instead of infer_from_lq.")
    parser.add_argument("--debug_tensor_dump_dir", type=str, default=None, help="Optional directory to dump one batch of HR/LQ tensors and alignment stats.")
    return parser


def _flatten_flashvsr_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    ordered_sections = [
        "data",
        "model",
        "train",
        "lora",
        "validation",
        "output",
        "wandb",
        "runtime",
    ]
    for key, value in config_data.items():
        if key not in ordered_sections and not isinstance(value, dict):
            merged[key] = value
    for section in ordered_sections:
        value = config_data.get(section)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def parse_flashvsr_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = flashvsr_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**_flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if args.image_tar_url is not None:
        if args.picked17k_image_tar_url is not None and args.picked17k_image_tar_url != args.image_tar_url:
            parser.error("--image_tar_url conflicts with --picked17k_image_tar_url. Please set only one or make them identical.")
        args.picked17k_image_tar_url = args.image_tar_url
    else:
        args.image_tar_url = args.picked17k_image_tar_url
    if args.resume_stage1_checkpoint is not None and args.lora_checkpoint is not None:
        parser.error("--resume_stage1_checkpoint cannot be combined with --lora_checkpoint.")
    if args.resume_training_state_dir is not None and (
        args.resume_stage1_checkpoint is not None
        or args.lora_checkpoint is not None
        or args.lq_proj_checkpoint is not None
    ):
        parser.error("--resume_training_state_dir cannot be combined with --resume_stage1_checkpoint, --lora_checkpoint, or --lq_proj_checkpoint.")
    return args


def dump_resolved_args(args) -> None:
    os.makedirs(args.output_path, exist_ok=True)
    payload = dict(sorted(vars(args).items()))
    payload["_runtime"] = {
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
    }
    with open(os.path.join(args.output_path, "resolved_args.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_path, "resolved_args.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def configure_deepspeed_runtime(accelerator, args) -> None:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return

    micro_batch_size = int(getattr(args, "batch_size", 1))
    grad_accum_steps = int(getattr(args, "gradient_accumulation_steps", 1))
    world_size = max(int(getattr(accelerator.state, "num_processes", 1)), 1)
    train_batch_size = micro_batch_size * grad_accum_steps * world_size

    configs = []
    if hasattr(plugin, "deepspeed_config") and isinstance(plugin.deepspeed_config, dict):
        configs.append(plugin.deepspeed_config)
    hf_ds_config = getattr(plugin, "hf_ds_config", None)
    hf_ds_config_dict = getattr(hf_ds_config, "config", None)
    if isinstance(hf_ds_config_dict, dict):
        configs.append(hf_ds_config_dict)

    for config in configs:
        config["train_micro_batch_size_per_gpu"] = micro_batch_size
        config["gradient_accumulation_steps"] = grad_accum_steps
        config["train_batch_size"] = train_batch_size

    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        f"[deepspeed_runtime_config] rank={rank} local_rank={local_rank} "
        f"train_micro_batch_size_per_gpu={micro_batch_size} "
        f"gradient_accumulation_steps={grad_accum_steps} "
        f"train_batch_size={train_batch_size}",
        flush=True,
    )


def _tensor_video_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    if video.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(video.shape)}")
    video = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in video:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def collect_fixed_validation_samples(dataset, num_samples: int) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []
    if isinstance(dataset, FlashVSRStreamingDataset):
        rng = random.Random(dataset.global_seed if dataset.global_seed is not None else 20260407)
        samples: List[Dict[str, Any]] = []
        if hasattr(dataset, "validation_video_iterator"):
            video_iterator = dataset.validation_video_iterator(rng=rng)
        elif (
            dataset.parquet_records
            or dataset.video_tar_urls
            or dataset.video_file_urls
            or dataset.video_manifest_urls
        ):
            video_iterator = dataset._video_iterator(rng=rng)
        else:
            video_iterator = None
        while video_iterator is not None and len(samples) < num_samples:
            processed = next(video_iterator)
            if processed is None:
                continue
            cached: Dict[str, Any] = {}
            for key, value in processed.items():
                if torch.is_tensor(value):
                    cached[key] = value.detach().cpu().clone()
                else:
                    cached[key] = deepcopy(value)
            samples.append(cached)
            if len(samples) >= num_samples:
                return samples
    iterator = iter(dataset)
    samples: List[Dict[str, Any]] = []
    restart_count = 0
    max_restarts = max(num_samples * 32, 64)
    while len(samples) < num_samples:
        try:
            sample = next(iterator)
        except StopIteration:
            restart_count += 1
            if restart_count > max_restarts:
                raise RuntimeError(
                    f"Failed to collect {num_samples} fixed validation samples after "
                    f"{max_restarts} iterator restarts from dataset={type(dataset).__name__}."
                )
            iterator = iter(dataset)
            continue
        cached: Dict[str, Any] = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                cached[key] = value.detach().cpu().clone()
            else:
                cached[key] = deepcopy(value)
        samples.append(cached)
    return samples


class FlashVSRValidationCallback:
    def __init__(
        self,
        output_path: str,
        validation_samples: List[Dict[str, Any]],
        num_inference_steps: int,
        fps: int,
        seed_base: int = 20260407,
        use_wandb: bool = False,
        validation_prompt: Optional[str] = None,
        validation_negative_prompt: str = "",
        validation_cfg_scale: float = 1.0,
        validation_use_wan_text_baseline: bool = False,
        validation_model_configs: Optional[List[ModelConfig]] = None,
        validation_tokenizer_config: Optional[ModelConfig] = None,
        validation_prompt_tensor_path: Optional[str] = None,
        validation_lq_proj_layer_num: Optional[int] = None,
        validation_lq_proj_temporal_mode: str = "streaming",
    ):
        self.output_path = output_path
        self.validation_samples = validation_samples
        self.num_inference_steps = num_inference_steps
        self.fps = fps
        self.seed_base = seed_base
        self.use_wandb = use_wandb
        self.validation_prompt = validation_prompt
        self.validation_negative_prompt = validation_negative_prompt
        self.validation_cfg_scale = validation_cfg_scale
        self.validation_use_wan_text_baseline = validation_use_wan_text_baseline
        self.validation_model_configs = validation_model_configs or []
        self.validation_tokenizer_config = validation_tokenizer_config
        self.validation_prompt_tensor_path = validation_prompt_tensor_path
        self.validation_lq_proj_layer_num = validation_lq_proj_layer_num
        self.validation_lq_proj_temporal_mode = validation_lq_proj_temporal_mode
    def _get_v2_validation_pipe(self, device, torch_dtype):
        return WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            prompt_tensor_path=self.validation_prompt_tensor_path,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
            lq_proj_temporal_mode=self.validation_lq_proj_temporal_mode,
        )

    def _get_wan_text_baseline_pipe(self, device, torch_dtype):
        return WanTextPromptLQPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            tokenizer_config=self.validation_tokenizer_config,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
            lq_proj_temporal_mode=self.validation_lq_proj_temporal_mode,
        )

    def __call__(self, accelerator, model, checkpoint_path: str, step: int):
        if not self.validation_samples:
            return
        validation_dir = os.path.join(self.output_path, "validation", f"step-{step}")
        os.makedirs(validation_dir, exist_ok=True)

        inference_model = model
        pipe = inference_model.pipe
        scheduler_state = {
            "timesteps": pipe.scheduler.timesteps.clone() if hasattr(pipe.scheduler, "timesteps") and pipe.scheduler.timesteps is not None else None,
            "training": getattr(pipe.scheduler, "training", None),
        }
        training_mode = inference_model.training
        inference_model.eval()
        try:
            for sample_index, sample in enumerate(self.validation_samples):
                sample_dir = os.path.join(validation_dir, f"sample_{sample_index:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                hr_tensor = sample["video"]
                lq_tensor = sample["lq_video"]
                hr_frames = _tensor_video_to_pil_frames(hr_tensor)
                lq_frames = _tensor_video_to_pil_frames(lq_tensor)
                save_video(hr_frames, os.path.join(sample_dir, "hr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                save_video(lq_frames, os.path.join(sample_dir, "lq.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                if self.validation_use_wan_text_baseline:
                    if not self.validation_prompt:
                        raise ValueError("validation_prompt must be set when validation_use_wan_text_baseline is enabled.")
                    baseline_pipe = self._get_wan_text_baseline_pipe(device=pipe.device, torch_dtype=pipe.torch_dtype)
                    baseline_pipe.lq_proj_scale = pipe.lq_proj_scale
                    exported_state = flashvsr_stage1_export(model.state_dict())
                    lq_proj_state = {}
                    lora_state = {}
                    for key, value in exported_state.items():
                        if key.startswith("lq_proj_in."):
                            lq_proj_state[key[len("lq_proj_in."):]] = value.detach().cpu()
                        elif "lora_" in key:
                            lora_state[key] = value.detach().cpu()
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
                    baseline_pipe.clear_lora(verbose=0)
                    if lora_state:
                        baseline_pipe.load_lora(
                            baseline_pipe.dit,
                            state_dict=lora_state,
                            verbose=0,
                        )
                    sr_frames = baseline_pipe.infer_from_lq_text(
                        prompt=self.validation_prompt,
                        negative_prompt=self.validation_negative_prompt,
                        lq_video=lq_tensor.unsqueeze(0),
                        height=int(hr_tensor.shape[-2]),
                        width=int(hr_tensor.shape[-1]),
                        num_frames=int(hr_tensor.shape[0]),
                        seed=self.seed_base + sample_index,
                        rand_device="cpu",
                        cfg_scale=self.validation_cfg_scale,
                        num_inference_steps=self.num_inference_steps,
                        tiled=True,
                        output_type="quantized",
                    )
                else:
                    baseline_pipe = self._get_v2_validation_pipe(device=pipe.device, torch_dtype=pipe.torch_dtype)
                    baseline_pipe.lq_proj_scale = pipe.lq_proj_scale
                    exported_state = flashvsr_stage1_export(model.state_dict())
                    lq_proj_state = {}
                    lora_state = {}
                    for key, value in exported_state.items():
                        if key.startswith("lq_proj_in."):
                            lq_proj_state[key[len("lq_proj_in."):]] = value.detach().cpu()
                        elif "lora_" in key:
                            lora_state[key] = value.detach().cpu()
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
                    baseline_pipe.clear_lora(verbose=0)
                    if lora_state:
                        baseline_pipe.load_lora(
                            baseline_pipe.dit,
                            state_dict=lora_state,
                            verbose=0,
                        )
                    sr_frames = baseline_pipe.infer_from_lq(
                        lq_video=lq_tensor.unsqueeze(0),
                        height=int(hr_tensor.shape[-2]),
                        width=int(hr_tensor.shape[-1]),
                        num_frames=int(hr_tensor.shape[0]),
                        seed=self.seed_base + sample_index,
                        rand_device="cpu",
                        num_inference_steps=self.num_inference_steps,
                        tiled=True,
                        output_type="quantized",
                    )
                save_video(sr_frames, os.path.join(sample_dir, "sr.mp4"), fps=self.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
                with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "checkpoint_path": checkpoint_path,
                            "step": step,
                            "sample_index": sample_index,
                            "validation_mode": "wan_text_baseline" if self.validation_use_wan_text_baseline else "v2_wan_fixed_prompt_projection",
                            "validation_cfg_scale": self.validation_cfg_scale,
                            "sample_seed": _serialize_sample_seed(sample.get("sample_seed")),
                        },
                        file,
                        ensure_ascii=False,
                        indent=2,
                    )
                if self.use_wandb and sample_index == 0:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(
                                {
                                    "validation/step": step,
                                    "validation/hr_video": wandb.Video(os.path.join(sample_dir, "hr.mp4"), fps=self.fps, format="mp4"),
                                    "validation/lq_video": wandb.Video(os.path.join(sample_dir, "lq.mp4"), fps=self.fps, format="mp4"),
                                    "validation/sr_video": wandb.Video(os.path.join(sample_dir, "sr.mp4"), fps=self.fps, format="mp4"),
                                },
                                step=step,
                            )
                    except Exception as error:
                        print(f"[wandb] validation log failed: {error}", flush=True)
        finally:
            inference_model.train(training_mode)
            if scheduler_state["training"]:
                pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
            else:
                if scheduler_state["timesteps"] is not None:
                    pipe.scheduler.timesteps = scheduler_state["timesteps"]
                if scheduler_state["training"] is not None:
                    pipe.scheduler.training = scheduler_state["training"]


@record
def main():
    def _flashvsr_excepthook(exc_type, exc_value, exc_traceback):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(
            f"[fatal rank={rank} local_rank={local_rank}] "
            f"{getattr(exc_type, '__name__', str(exc_type))}: {exc_value}",
            flush=True,
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _flashvsr_excepthook
    args = parse_flashvsr_args()
    if args.parquet_cache_dir:
        os.environ["FLASHVSR_PARQUET_CACHE_DIR"] = args.parquet_cache_dir
    if args.debug_tensor_dump_dir:
        os.environ["FLASHVSR_TENSOR_DEBUG_DIR"] = args.debug_tensor_dump_dir
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
    configure_deepspeed_runtime(accelerator, args)

    if accelerator.is_main_process:
        dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)
    if args.dataset_mode == "streaming":
        dataset = FlashVSRStreamingDataset(
            internal_url=args.internal_url,
            metadata_url=args.metadata_url,
            metadata_source=args.metadata_source,
            max_parquet_records=args.max_parquet_records,
            min_overall_score=args.min_overall_score,
            require_qwen35_parse_success=args.require_qwen35_parse_success,
            image_internal_url=args.image_internal_url,
            image_dataset_prob=args.image_dataset_prob,
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
            image_branch_num_frames=args.image_branch_num_frames,
        )
    elif args.dataset_mode == "parquet_v2":
        dataset = FlashVSRParquetTarDatasetV2(
            metadata_url=args.metadata_url,
            metadata_source=args.metadata_source,
            image_metadata_url=args.image_metadata_url,
            image_internal_url=args.image_internal_url,
            image_dataset_prob=args.image_dataset_prob,
            takano_dataset_prob=args.takano_dataset_prob,
            yubari_dataset_prob=args.yubari_dataset_prob,
            image_as_single_frame=args.image_as_single_frame,
            yubari_video_tar_url=args.yubari_video_tar_url,
            yubari_shard_start=args.yubari_shard_start,
            yubari_shard_end=args.yubari_shard_end,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            stride=args.stride,
            max_source_frames=args.max_source_frames,
            enable_degradation=args.enable_degradation,
            degradation_config_path=args.degradation_config_path,
            global_seed=args.global_seed,
            output_tensors=True,
            max_parquet_records=args.max_parquet_records,
            max_yubari_records=args.max_yubari_records,
            media_cache_dir=args.media_cache_dir,
            parquet_prewarm_files_per_source=args.parquet_prewarm_files_per_source,
        )
        if args.image_video_joint_packed:
            dataset.custom_collate_fn = collate_image_video_joint_v5
        else:
            dataset.custom_collate_fn = FlashVSRParquetTarDatasetV2.tensor_collate_fn
    elif args.dataset_mode == "tar_v3":
        if args.image_dataset_prob > 0:
            raise ValueError("tar_v3 uses picked17k_dataset_prob instead of image_dataset_prob")
        if args.picked17k_dataset_prob > 0 and not args.image_video_joint_packed:
            raise ValueError("tar_v3 with picked17k single-frame images requires --image_video_joint_packed")
        dataset = FlashVSRTarStreamingDatasetV3(
            yubari_video_tar_url=args.yubari_video_tar_url,
            picked17k_image_tar_url=args.image_tar_url,
            picked17k_dataset_prob=args.picked17k_dataset_prob,
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
            image_as_single_frame=args.image_as_single_frame,
            output_tensors=True,
        )
    elif args.dataset_mode == "tar_v5":
        if args.image_dataset_prob > 0:
            raise ValueError("tar_v5 uses picked17k_dataset_prob instead of image_dataset_prob")
        if args.picked17k_dataset_prob > 0 and not args.image_video_joint_packed:
            raise ValueError("tar_v5 grouped-image training requires --image_video_joint_packed")
        dataset = FlashVSRTarStreamingDatasetV5(
            yubari_video_tar_url=args.yubari_video_tar_url,
            picked17k_image_tar_url=args.image_tar_url,
            picked17k_dataset_prob=args.picked17k_dataset_prob,
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
    elif args.dataset_mode == "tar_v53":
        dataset = FlashVSRTarStreamingDatasetV53USMGT(
            yubari_video_tar_url=args.yubari_video_tar_url,
            takano_video_tar_url=args.takano_video_tar_url,
            image_tar_root_url=args.image_tar_url,
            yubari_video_prob=args.yubari_video_prob,
            takano_video_prob=args.takano_video_prob,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            stride=args.stride,
            max_source_frames=args.max_source_frames,
            enable_degradation=args.enable_degradation,
            degradation_config_path=args.degradation_config_path,
            gt_sharpen=args.gt_sharpen,
            gt_sharpen_backend=args.gt_sharpen_backend,
            gt_sharpen_device=args.gt_sharpen_device,
            degradation_device=args.degradation_device,
            degradation_seed=args.degradation_seed,
            hq_prefix_frames=args.hq_prefix_frames,
            control_dropout_prob=args.control_dropout_prob,
            shuffle_buffer=args.shuffle_buffer,
            global_seed=args.global_seed,
            output_tensors=True,
            image_branch_num_frames=args.image_branch_num_frames,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
            special_operator_map={
                "video": ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16)),
                "lq_video": ToAbsolutePath(args.dataset_base_path)
                >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16)),
            },
        )
    model = FlashVSRStage1TrainingModule(
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
        lq_proj_temporal_mode=args.lq_proj_temporal_mode,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        freeze_lq_proj_in=args.freeze_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        image_video_joint_packed=args.image_video_joint_packed,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    if accelerator.is_local_main_process:
        trainable_named_params = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
        trainable_param_count = sum(numel for _, numel in trainable_named_params)
        preview_names = [name for name, _ in trainable_named_params[:80]]
        print(f"Trainable parameter tensors: {len(trainable_named_params)}")
        print(f"Trainable parameter count: {trainable_param_count}")
        print("Trainable parameter preview:")
        for name in preview_names:
            print(f"  - {name}")
        if len(trainable_named_params) > len(preview_names):
            print(f"  ... and {len(trainable_named_params) - len(preview_names)} more")
    validation_callback = None
    if args.validation_num_samples > 0 and accelerator.is_main_process:
        print("Preparing fixed validation samples...", flush=True)
        validation_samples = collect_fixed_validation_samples(dataset, args.validation_num_samples)
        print(f"Prepared {len(validation_samples)} fixed validation samples.", flush=True)
        validation_prompt = None
        if args.validation_prompt_file:
            with open(args.validation_prompt_file, "r", encoding="utf-8") as file:
                validation_prompt = file.read().strip()
        model_paths = json.loads(args.model_paths) if args.model_paths is not None else []
        if not model_paths:
            raise ValueError("V2 validation requires model_paths to locate the base Wan model.")
        base_model_dir = str(Path(model_paths[0]).resolve().parent)
        if args.validation_use_wan_text_baseline:
            validation_model_configs = [
                ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(Path(base_model_dir) / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
            ]
            validation_tokenizer_config = ModelConfig(path=str(Path(base_model_dir) / "google/umt5-xxl"))
        else:
            validation_model_configs = [
                ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
            ]
            validation_tokenizer_config = None
        validation_callback = FlashVSRValidationCallback(
            output_path=args.output_path,
            validation_samples=validation_samples,
            num_inference_steps=args.validation_num_inference_steps,
            fps=args.validation_fps,
            seed_base=(args.global_seed if args.global_seed is not None else 20260407),
            use_wandb=args.use_wandb,
            validation_prompt=validation_prompt,
            validation_negative_prompt=args.validation_negative_prompt,
            validation_cfg_scale=args.validation_cfg_scale,
            validation_use_wan_text_baseline=args.validation_use_wan_text_baseline,
            validation_model_configs=validation_model_configs,
            validation_tokenizer_config=validation_tokenizer_config,
            validation_prompt_tensor_path=args.prompt_tensor_path,
            validation_lq_proj_layer_num=args.lq_proj_layer_num,
            validation_lq_proj_temporal_mode=args.lq_proj_temporal_mode,
        )
    accelerator.wait_for_everyone()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=flashvsr_stage1_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
