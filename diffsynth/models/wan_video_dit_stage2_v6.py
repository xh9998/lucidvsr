import math
import types
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from .wan_video_dit import rope_apply

try:
    from block_sparse_attn import block_sparse_attn_func
except ModuleNotFoundError:  # Local Mac may not have the remote FlashVSR kernel.
    block_sparse_attn_func = None


def _time_causal_mask(
    f: int,
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Token mask that allows all spatial tokens from current and past latent times."""
    time_index = torch.arange(f, device=device).repeat_interleave(h * w)
    allowed = time_index.view(-1, 1) >= time_index.view(1, -1)
    mask = torch.zeros((f * h * w, f * h * w), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def dense_time_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    grid: Tuple[int, int, int],
) -> torch.Tensor:
    """Correct dense fallback for Stage 2 causal attention.

    This is intentionally time-causal, not sequence-causal: tokens in the same
    latent frame can see each other, while future latent frames are masked.
    It is a correctness baseline for smoke/numeric checks and is not the final
    high-resolution training kernel.
    """
    f, h, w = grid
    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    mask = _time_causal_mask(f, h, w, device=q.device, dtype=torch.float32)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return rearrange(out, "b n s d -> b s (n d)")


def _block_time_mask(
    *,
    batch_size: int,
    num_heads: int,
    f_blocks: int,
    spatial_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Block mask for all past/current temporal blocks.

    Shape follows FlashVSR's block_sparse_attn kernel expectation:
    (B, heads, q_blocks, kv_blocks).
    """
    query_time = torch.arange(f_blocks, device=device).repeat_interleave(spatial_blocks)
    key_time = torch.arange(f_blocks, device=device).repeat_interleave(spatial_blocks)
    mask = key_time.view(1, -1) <= query_time.view(-1, 1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1).contiguous()


def _partition_3d_blocks(x: torch.Tensor, win: Tuple[int, int, int]) -> torch.Tensor:
    b, f, h, w, c = x.shape
    wf, wh, ww = win
    x = x.view(b, f // wf, wf, h // wh, wh, w // ww, ww, c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wf * wh * ww, c)


def _reverse_3d_blocks(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]) -> torch.Tensor:
    f, h, w = orig
    wf, wh, ww = win
    nf, nh, nw = f // wf, h // wh, w // ww
    b = windows.shape[0] // (nf * nh * nw)
    x = windows.view(b, nf, nh, nw, wf, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, f, h, w, -1)


def block_streaming_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    grid: Tuple[int, int, int],
    pre_cache_k: Optional[torch.Tensor] = None,
    pre_cache_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashVSR-style streaming block causal attention.

    Unlike the earlier whole-sequence fallback, this expects the caller to feed
    even latent-time chunks, matching the official inference contract:
    first chunk `f=6`, later chunks `f=2`. Previous chunks enter through
    `pre_cache_k/v`, so no odd-length global latent sequence needs padding.
    """
    if block_sparse_attn_func is None:
        raise RuntimeError("block_sparse_attn is unavailable; use dense_time_causal mode for local smoke.")
    b, s, d = q.shape
    f, h, w = grid
    if s != f * h * w:
        raise ValueError(f"Token/grid mismatch: tokens={s}, grid={(f, h, w)}")
    if h % 8 != 0 or w % 8 != 0:
        raise ValueError(f"Stage2 block sparse requires h/w token grid divisible by 8, got h={h}, w={w}")
    if f % 2 != 0:
        raise ValueError(
            f"Official-style Stage2 block sparse expects even chunk latent-time, got f={f}. "
            "Use chunks like first f=6 and later f=2 instead of padding a whole odd sequence."
        )

    win = (2, 8, 8)
    q_grid = q.view(b, f, h, w, d)
    k_grid = k.view(b, f, h, w, d)
    v_grid = v.view(b, f, h, w, d)
    q_w = _partition_3d_blocks(q_grid, win)
    k_w = _partition_3d_blocks(k_grid, win)
    v_w = _partition_3d_blocks(v_grid, win)
    if pre_cache_k is not None and pre_cache_v is not None:
        k_w = torch.cat([pre_cache_k, k_w], dim=0)
        v_w = torch.cat([pre_cache_v, v_w], dim=0)

    block_size = q_w.shape[1]
    q_blocks = q_w.shape[0] // b
    kv_blocks = k_w.shape[0] // b
    if block_size != 128:
        raise ValueError(f"Unexpected Stage2 block size {block_size}; expected 128.")
    reorder_q = rearrange(q_w, "(b nb) bs d -> b (nb bs) d", b=b, nb=q_blocks, bs=block_size)
    reorder_k = rearrange(k_w, "(b nb) bs d -> b (nb bs) d", b=b, nb=kv_blocks, bs=block_size)
    reorder_v = rearrange(v_w, "(b nb) bs d -> b (nb bs) d", b=b, nb=kv_blocks, bs=block_size)

    q_kernel = rearrange(reorder_q, "b s (n hd) -> (b s) n hd", n=num_heads)
    k_kernel = rearrange(reorder_k, "b s (n hd) -> (b s) n hd", n=num_heads)
    v_kernel = rearrange(reorder_v, "b s (n hd) -> (b s) n hd", n=num_heads)
    cu_q = torch.arange(0, (b + 1) * reorder_q.shape[1], reorder_q.shape[1], device=q.device, dtype=torch.int32)
    cu_k = torch.arange(0, (b + 1) * reorder_k.shape[1], reorder_k.shape[1], device=q.device, dtype=torch.int32)
    head_mask_type = torch.ones(num_heads, device=q.device, dtype=torch.int32)
    spatial_blocks = (h // win[1]) * (w // win[2])
    query_f_blocks = f // win[0]
    cache_f_blocks = 0 if pre_cache_k is None else pre_cache_k.shape[0] // (b * spatial_blocks)
    query_time = cache_f_blocks + torch.arange(query_f_blocks, device=q.device).repeat_interleave(spatial_blocks)
    key_time = torch.arange(cache_f_blocks + query_f_blocks, device=q.device).repeat_interleave(spatial_blocks)
    block_mask = (key_time.view(1, -1) <= query_time.view(-1, 1))
    block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(b, num_heads, -1, -1).contiguous()

    # Keep `_block_time_mask` available for tests, but build the streaming mask
    # explicitly above so cached KV blocks are treated as past time.
    spatial_blocks = (h // win[1]) * (w // win[2])
    out = block_sparse_attn_func(
        q_kernel,
        k_kernel,
        v_kernel,
        cu_q,
        cu_k,
        head_mask_type,
        None,
        block_mask,
        reorder_q.shape[1],
        reorder_k.shape[1],
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = rearrange(out.unsqueeze(0), "b s n hd -> b s (n hd)", n=num_heads)
    out_w = rearrange(out, "b (nb bs) d -> (b nb) bs d", nb=q_blocks, bs=block_size)
    out_grid = _reverse_3d_blocks(out_w, win, (f, h, w)).view(b, f * h * w, d)
    return out_grid, k_w.detach(), v_w.detach()


def stage2_self_attention_forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    grid = getattr(self, "_flashvsr_stage2_grid", None)
    mode = getattr(self, "_flashvsr_stage2_attention_mode", "dense_time_causal")
    if grid is None:
        raise RuntimeError("Stage2 causal attention requires _flashvsr_stage2_grid to be set before block forward.")
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    if mode == "dense_time_causal":
        out = dense_time_causal_attention(q, k, v, num_heads=self.num_heads, grid=grid)
    elif mode == "block_streaming_causal":
        out, cache_k, cache_v = block_streaming_causal_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            grid=grid,
            pre_cache_k=getattr(self, "_flashvsr_stage2_cache_k", None),
            pre_cache_v=getattr(self, "_flashvsr_stage2_cache_v", None),
        )
        self._flashvsr_stage2_cache_k = cache_k
        self._flashvsr_stage2_cache_v = cache_v
    else:
        raise ValueError(f"Unsupported Stage2 attention mode: {mode}")
    return self.o(out)


def enable_stage2_causal_attention(dit: torch.nn.Module, *, mode: str = "dense_time_causal") -> torch.nn.Module:
    """Patch a WanModel instance to use Stage2 causal self-attention.

    The patch is instance-local and state-dict compatible: it reuses the
    existing q/k/v/o/RMSNorm parameters and only replaces the forward method.
    """
    for block in dit.blocks:
        self_attn = block.self_attn
        if not hasattr(self_attn, "_flashvsr_stage2_original_forward"):
            self_attn._flashvsr_stage2_original_forward = self_attn.forward
        self_attn._flashvsr_stage2_attention_mode = mode
        self_attn.forward = types.MethodType(stage2_self_attention_forward, self_attn)
    dit.flashvsr_stage2_attention_mode = mode
    return dit


def set_stage2_grid(dit: torch.nn.Module, grid: Tuple[int, int, int]) -> None:
    for block in dit.blocks:
        block.self_attn._flashvsr_stage2_grid = tuple(int(v) for v in grid)


def clear_stage2_caches(dit: torch.nn.Module) -> None:
    for block in dit.blocks:
        block.self_attn._flashvsr_stage2_cache_k = None
        block.self_attn._flashvsr_stage2_cache_v = None
