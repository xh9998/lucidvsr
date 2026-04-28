"""Small subset of apex.normalization used by SeedVR inference.

This avoids compiling NVIDIA Apex on test machines where SeedVR only needs
LayerNorm/RMSNorm modules for inference.
"""

from __future__ import annotations

import torch
from torch import nn


class FusedLayerNorm(nn.LayerNorm):
    pass


class FusedRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(variance.to(dtype=x.dtype) + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(dtype=y.dtype, device=y.device)
        return y

