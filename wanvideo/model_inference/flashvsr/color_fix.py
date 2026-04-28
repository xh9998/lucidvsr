from __future__ import annotations

from typing import Iterable, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    n, c = feat.shape[:2]
    var = feat.view(n, c, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(n, c, 1, 1)
    mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean, std


def _adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = _calc_mean_std(style_feat)
    content_mean, content_std = _calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _make_gaussian3x3_kernel(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(vals, dtype=dtype, device=device)


def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    _, c, _, _ = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    pad = radius
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, dilation=radius, groups=c)


def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    high = torch.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2**i
        blurred = _wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low


def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    c_high, _ = _wavelet_decompose(content, levels=levels)
    _, s_low = _wavelet_decompose(style, levels=levels)
    return c_high + s_low


def _frames_to_tensor(frames: Sequence) -> torch.Tensor:
    arrays = []
    for frame in frames:
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")
            array = np.asarray(frame, dtype=np.uint8)
        else:
            array = np.asarray(frame, dtype=np.uint8)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
        arrays.append(array)
    if not arrays:
        raise ValueError("No frames to color-fix.")
    stacked = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
    return tensor


def _tensor_to_uint8_frames(tensor: torch.Tensor) -> list[np.ndarray]:
    tensor = tensor.clamp(-1.0, 1.0)
    tensor = ((tensor + 1.0) * 127.5).round().byte()
    return [frame.permute(1, 2, 0).cpu().numpy() for frame in tensor]


def apply_color_fix(
    sr_frames: Sequence,
    lq_frames: Sequence,
    *,
    method: Literal["wavelet", "adain"] = "adain",
    levels: int = 5,
) -> list[np.ndarray]:
    if len(sr_frames) == 0:
        return []
    if len(sr_frames) != len(lq_frames):
        raise ValueError(f"Frame count mismatch: sr={len(sr_frames)} lq={len(lq_frames)}")

    sr = _frames_to_tensor(sr_frames)
    lq = _frames_to_tensor(lq_frames)
    if sr.shape != lq.shape:
        raise ValueError(f"Shape mismatch for color-fix: sr={tuple(sr.shape)} lq={tuple(lq.shape)}")

    if method == "wavelet":
        fixed = _wavelet_reconstruct(sr, lq, levels=levels)
    elif method == "adain":
        fixed = _adain(sr, lq)
    else:
        raise ValueError(f"Unknown color-fix method: {method}")
    return _tensor_to_uint8_frames(fixed)
