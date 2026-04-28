import os
import sys
import traceback
import warnings
import argparse
import json
import random
from copy import deepcopy
from typing import List, Dict, Any, Optional
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
from diffsynth.diffusion import *
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ShapeChecker,
)
from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CACHE_T = 2
_FLASHVSR_BLOCK_BRANCH_REPORTED = False
_TENSOR_DEBUG_REPORTED = set()


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
    def __init__(self, in_dim, out_dim, layer_num=1, zero_init_output=True):
        super().__init__()
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


class WanTextPromptLQPipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        tokenizer_config=None,
        lq_proj_layer_num=None,
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
            input_params=("lq_video", "height", "width"),
            output_params=("lq_latents",),
            onload_model_names=("lq_proj_in",),
        )

    def process(self, pipe, lq_video, height, width):
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
        lq_latents = pipe.lq_proj_in(lq_input)
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


def flashvsr_stage1_model_fn(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    lq_latents=None,
    lq_proj_scale: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
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

    patchified = dit.patchify(latents)
    if isinstance(patchified, tuple):
        x, (f, h, w) = patchified
    else:
        x = patchified
        if x.ndim == 5:
            _, _, f, h, w = x.shape
            x = rearrange(x, "b c f h w -> b (f h w) c")
        elif x.ndim == 3:
            f = latents.shape[2] // dit.patch_size[0]
            h = latents.shape[3] // dit.patch_size[1]
            w = latents.shape[4] // dit.patch_size[2]
        else:
            raise ValueError(f"Unsupported patchify output shape: {tuple(x.shape)}")
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if lq_latents is not None:
        tokens_per_frame = h * w
        expected_tokens = x.shape[1]
        aligned_lq_latents = []
        for layer_latents in lq_latents:
            current_tokens = layer_latents.shape[1]
            if current_tokens == expected_tokens:
                aligned_lq_latents.append(layer_latents)
                continue
            if current_tokens < expected_tokens:
                pad_tokens = expected_tokens - current_tokens
                if pad_tokens % tokens_per_frame != 0:
                    raise ValueError(
                        f"Cannot align lq_latents with x tokens: x={expected_tokens}, lq={current_tokens}, h={h}, w={w}"
                    )
                padding = torch.zeros(
                    layer_latents.shape[0],
                    pad_tokens,
                    layer_latents.shape[2],
                    device=layer_latents.device,
                    dtype=layer_latents.dtype,
                )
                aligned_lq_latents.append(torch.cat([padding, layer_latents], dim=1))
                continue
            trim_tokens = current_tokens - expected_tokens
            if trim_tokens % tokens_per_frame != 0:
                raise ValueError(
                    f"Cannot trim lq_latents to x tokens: x={expected_tokens}, lq={current_tokens}, h={h}, w={w}"
                )
            aligned_lq_latents.append(layer_latents[:, trim_tokens:, :])
        lq_latents = aligned_lq_latents
        _dump_tensor_preview_once(
            "04_model_token_alignment",
            None,
            extra={
                "x_shape": list(x.shape),
                "grid": {"f": int(f), "h": int(h), "w": int(w)},
                "expected_tokens": int(expected_tokens),
                "tokens_per_frame": int(tokens_per_frame),
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
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs,
            )
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

    patchified = dit.patchify(latents)
    if isinstance(patchified, tuple):
        x, (f, h, w) = patchified
    else:
        x = patchified
        if x.ndim == 5:
            _, _, f, h, w = x.shape
            x = rearrange(x, "b c f h w -> b (f h w) c")
        elif x.ndim == 3:
            f = latents.shape[2] // dit.patch_size[0]
            h = latents.shape[3] // dit.patch_size[1]
            w = latents.shape[4] // dit.patch_size[2]
        else:
            raise ValueError(f"Unsupported patchify output shape: {tuple(x.shape)}")
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    if lq_latents is not None:
        tokens_per_frame = h * w
        expected_tokens = x.shape[1]
        aligned_lq_latents = []
        for layer_latents in lq_latents:
            current_tokens = layer_latents.shape[1]
            if current_tokens == expected_tokens:
                aligned_lq_latents.append(layer_latents)
                continue
            if current_tokens < expected_tokens:
                pad_tokens = expected_tokens - current_tokens
                if pad_tokens % tokens_per_frame != 0:
                    raise ValueError(
                        f"Cannot align lq_latents with x tokens: x={expected_tokens}, lq={current_tokens}, h={h}, w={w}"
                    )
                padding = torch.zeros(
                    layer_latents.shape[0],
                    pad_tokens,
                    layer_latents.shape[2],
                    device=layer_latents.device,
                    dtype=layer_latents.dtype,
                )
                aligned_lq_latents.append(torch.cat([padding, layer_latents], dim=1))
                continue
            trim_tokens = current_tokens - expected_tokens
            if trim_tokens % tokens_per_frame != 0:
                raise ValueError(
                    f"Cannot trim lq_latents to x tokens: x={expected_tokens}, lq={current_tokens}, h={h}, w={w}"
                )
            aligned_lq_latents.append(layer_latents[:, trim_tokens:, :])
        lq_latents = aligned_lq_latents

    for block_id, block in enumerate(dit.blocks):
        if lq_latents is not None and block_id < len(lq_latents):
            x = x + (lq_latents[block_id] * lq_proj_scale)
        if dit.training:
            x = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                x, context, t_mod, freqs,
            )
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
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            FlashVSRUnit_FixedPrompt(),
            WanVideoUnit_InputVideoEmbedder(),
            FlashVSRUnit_LQVideoEmbedder(),
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


def flashvsr_stage1_v3_export(state_dict):
    converted = {}
    for name, param in state_dict.items():
        if name.startswith("pipe.dit."):
            converted[name[len("pipe.dit.") :]] = param
        elif name.startswith("pipe.lq_proj_in."):
            converted["lq_proj_in." + name[len("pipe.lq_proj_in.") :]] = param
        else:
            converted[name] = param
    return converted


def split_v3_exported_state(exported_state):
    dit_state = {}
    lq_proj_state = {}
    other_state = {}
    for key, value in exported_state.items():
        if key.startswith("lq_proj_in."):
            lq_proj_state[key[len("lq_proj_in."):]] = value.detach().cpu()
        elif "lora_" in key:
            # v3 should not produce LoRA weights, but keep this branch explicit so
            # unexpected LoRA tensors do not silently leak into full-finetune validation.
            continue
        elif key.startswith("blocks.") or key.startswith("patch_embedding.") or key.startswith("time_embedding.") or key.startswith("time_projection.") or key.startswith("text_embedding.") or key.startswith("head.") or key.startswith("rope_embedder.") or key.startswith("img_emb.") or key.startswith("modulation.") or key.startswith("head_mod.") or key.startswith("before_proj.") or key.startswith("scale_shift_table") or key.startswith("pos_embed") or key.startswith("time_embed.") or key.startswith("time_mlp.") or key.startswith("norm_out.") or key.startswith("proj_out."):
            dit_state[key] = value.detach().cpu()
        else:
            other_state[key] = value.detach().cpu()
    if not dit_state:
        for key, value in exported_state.items():
            if not key.startswith("lq_proj_in.") and "lora_" not in key:
                dit_state[key] = value.detach().cpu()
    return dit_state, lq_proj_state, other_state


class FlashVSRStage1FullFinetuneTrainingModule(DiffusionTrainingModule):
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
        lq_proj_layer_num=None,
        lq_proj_scale: float = 1.0,
        zero_init_lq_proj_in=True,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        debug_tensor_dump_dir=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
    ):
        super().__init__()
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
        )
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
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

    def get_pipeline_inputs(self, data):
        if torch.is_tensor(data["video"]):
            video = data["video"]
            if video.ndim == 5:
                height = int(video.shape[-2])
                width = int(video.shape[-1])
                num_frames = int(video.shape[1])
            else:
                raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")
            _dump_tensor_preview_once("00_input_hr_tensor", video, pipe=self.pipe)
        else:
            video = data["video"]
            height = data["video"][0].size[1]
            width = data["video"][0].size[0]
            num_frames = len(data["video"])
        inputs_shared = {
            "input_video": video,
            "lq_video": data["lq_video"],
            "height": height,
            "width": width,
            "num_frames": num_frames,
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
        return inputs_shared, {}, {}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = FlowMatchSFTLoss(self.pipe, **inputs[0], **inputs[1])
        return loss


def flashvsr_v3_parser():
    parser = argparse.ArgumentParser(description="FlashVSR Stage 1 full-finetune scaffold.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    for action in parser._actions:
        if action.dest == "dataset_base_path":
            action.required = False
    parser.add_argument("--prompt_tensor_path", type=str, default=None, help="Path to fixed prompt tensor.")
    parser.add_argument("--lq_proj_checkpoint", type=str, default=None, help="Optional path to initialize lq_proj_in.")
    parser.add_argument("--lq_proj_layer_num", type=int, default=1, help="Number of linear projection heads in lq_proj_in. Defaults to 1.")
    parser.add_argument("--lq_proj_scale", type=float, default=1.0, help="Fixed multiplicative scale applied to lq_proj_in latents before adding to x.")
    parser.add_argument("--zero_init_lq_proj_in", type=lambda x: str(x).lower() in ("1", "true", "yes", "y"), default=True, help="Zero-initialize lq_proj_in output projection so step-0 keeps base-model behavior.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    parser.add_argument("--dataset_mode", type=str, default="unified", choices=("unified", "streaming"), help="Dataset backend.")
    parser.add_argument("--internal_url", type=str, default=None, help="Video root/prefix for streaming mode.")
    parser.add_argument("--metadata_url", type=str, default=None, help="Optional parquet root/path for metadata-driven streaming mode.")
    parser.add_argument("--metadata_source", type=str, default="auto", choices=("auto", "storymotion", "takano"), help="Parquet adapter type.")
    parser.add_argument("--max_parquet_records", type=int, default=None, help="Optional limit for loaded parquet rows.")
    parser.add_argument("--min_overall_score", type=float, default=None, help="Optional storymotion quality filter.")
    parser.add_argument("--require_qwen35_parse_success", default=False, action="store_true", help="Keep only rows with parsed qwen output.")
    parser.add_argument("--image_internal_url", type=str, default=None, help="Optional image root/prefix for pseudo-video mixing.")
    parser.add_argument("--image_dataset_prob", type=float, default=0.0, help="Probability of drawing pseudo-video image samples in streaming mode.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride for streaming-mode video sampling.")
    parser.add_argument("--max_source_frames", type=int, default=160, help="Maximum decoded source frames per raw video sample.")
    parser.add_argument("--enable_degradation", default=False, action="store_true", help="Enable online HR->LQ degradation in streaming mode.")
    parser.add_argument("--degradation_config_path", type=str, default=None, help="Path to RealESRGAN/RealBasicVSR-style degradation config.")
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


def parse_flashvsr_v3_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = flashvsr_v3_parser()
    if pre_args.config is not None:
        with open(pre_args.config, "r", encoding="utf-8") as file:
            config_data = yaml.safe_load(file) or {}
        parser.set_defaults(**_flatten_flashvsr_config(config_data))
    args = parser.parse_args(argv)
    if args.prompt_tensor_path is None:
        parser.error("--prompt_tensor_path is required, either from CLI or YAML config.")
    if isinstance(args.lora_base_model, str) and args.lora_base_model.strip().lower() in ("", "none", "null"):
        args.lora_base_model = None
    if isinstance(args.lora_checkpoint, str) and args.lora_checkpoint.strip().lower() in ("", "none", "null"):
        args.lora_checkpoint = None
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
        fast_video_urls = list(dataset.video_file_urls[: max(num_samples * 4, num_samples)])
        for url in fast_video_urls:
            processed = dataset._process_video_bytes(dataset._open_binary(url), sample_id=os.path.basename(url), rng=rng)
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
    while len(samples) < num_samples:
        sample = next(iterator)
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
    def _get_v2_validation_pipe(self, device, torch_dtype):
        return WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            prompt_tensor_path=self.validation_prompt_tensor_path,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
        )

    def _get_wan_text_baseline_pipe(self, device, torch_dtype):
        return WanTextPromptLQPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=self.validation_model_configs,
            tokenizer_config=self.validation_tokenizer_config,
            lq_proj_layer_num=self.validation_lq_proj_layer_num,
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
                    exported_state = flashvsr_stage1_v3_export(model.state_dict())
                    dit_state, lq_proj_state, _ = split_v3_exported_state(exported_state)
                    baseline_pipe.dit.load_state_dict(dit_state, strict=False)
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
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
                    exported_state = flashvsr_stage1_v3_export(model.state_dict())
                    dit_state, lq_proj_state, _ = split_v3_exported_state(exported_state)
                    baseline_pipe.dit.load_state_dict(dit_state, strict=False)
                    baseline_pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
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
                            "sample_seed": int(sample.get("sample_seed", torch.tensor(-1)).item() if torch.is_tensor(sample.get("sample_seed")) else sample.get("sample_seed", -1)),
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
    args = parse_flashvsr_v3_args()
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
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")

    def stage_log(message: str) -> None:
        print(f"[stage rank={rank} local_rank={local_rank}] {message}", flush=True)

    if getattr(accelerator, "dataloader_config", None) is not None:
        stage_log(
            "accelerator dataloader_config "
            f"dispatch_batches={getattr(accelerator.dataloader_config, 'dispatch_batches', None)} "
            f"split_batches={getattr(accelerator.dataloader_config, 'split_batches', None)} "
            f"even_batches={getattr(accelerator.dataloader_config, 'even_batches', None)}"
        )

    if accelerator.is_main_process:
        dump_resolved_args(args)
        print(f"Resolved args saved under: {args.output_path}", flush=True)
    stage_log("about to build dataset")
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
    stage_log("dataset constructed")
    stage_log("about to build training module")
    model = FlashVSRStage1FullFinetuneTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        prompt_tensor_path=args.prompt_tensor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lq_proj_checkpoint=args.lq_proj_checkpoint,
        lq_proj_layer_num=args.lq_proj_layer_num,
        zero_init_lq_proj_in=args.zero_init_lq_proj_in,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    stage_log("training module constructed")
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
        )
    stage_log("validation callback ready")
    accelerator.wait_for_everyone()
    stage_log("pre-training barrier finished")
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=None,
        state_dict_converter=flashvsr_stage1_v3_export,
        validation_callback=validation_callback,
    )
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
