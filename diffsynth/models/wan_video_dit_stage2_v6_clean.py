import math
import random
import types
from typing import Optional, Tuple

import torch
from einops import rearrange

from .wan_video_dit import flash_attention, rope_apply

try:
    from block_sparse_attn import block_sparse_attn_func
except ModuleNotFoundError:  # Local development machines may not have the CUDA extension.
    block_sparse_attn_func = None


STAGE2_BLOCK_WINDOW = (2, 8, 8)


def _partition_3d_blocks(x: torch.Tensor, win: Tuple[int, int, int]) -> torch.Tensor:
    batch, frames, height, width, channels = x.shape
    win_f, win_h, win_w = win
    if frames % win_f != 0 or height % win_h != 0 or width % win_w != 0:
        raise ValueError(
            f"Stage2 block-sparse requires grid divisible by {win}, got grid={(frames, height, width)}"
        )
    x = x.view(batch, frames // win_f, win_f, height // win_h, win_h, width // win_w, win_w, channels)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(batch, -1, win_f * win_h * win_w, channels)


def _reverse_3d_blocks(windows: torch.Tensor, win: Tuple[int, int, int], grid: Tuple[int, int, int]) -> torch.Tensor:
    frames, height, width = grid
    win_f, win_h, win_w = win
    batch = windows.shape[0]
    windows = windows.view(batch, frames // win_f, height // win_h, width // win_w, win_f, win_h, win_w, -1)
    windows = windows.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return windows.view(batch, frames, height, width, -1)


def build_stage2_chunk_block_mask(
    *,
    batch_size: int,
    num_heads: int,
    latent_frames: int,
    height_blocks: int,
    width_blocks: int,
    device: torch.device,
    local_num: Optional[int] = None,
) -> torch.Tensor:
    """Build author/FlashVSR-style chunk causal mask at block granularity.

    One chunk equals two latent-time positions, matching FlashVSR's `(2, 8, 8)`
    sparse block. This follows the official FlashVSR `generate_causal_block_mask`
    semantics: causal + local history window, then explicitly unmask the first
    three chunks into a full-attention start window.
    """
    if latent_frames % 2 != 0:
        raise ValueError(f"Stage2 latent_frames must be even after dropping first GT latent, got {latent_frames}")
    chunks = latent_frames // 2
    spatial_blocks = int(height_blocks) * int(width_blocks)
    if local_num is None:
        local_random = random.random()
        if local_random < 0.3:
            local_num = chunks - 3
        elif local_random < 0.4:
            local_num = chunks - 4
        elif local_random < 0.5:
            local_num = chunks - 2
        else:
            local_num = chunks
    local_num = max(1, min(int(local_num), chunks))

    query = torch.arange(chunks, device=device).view(-1, 1)
    key = torch.arange(chunks, device=device).view(1, -1)
    chunk_allowed = (key <= query) & (key >= query - local_num + 1)
    if chunks >= 2:
        chunk_allowed[0, 1] = True
    if chunks >= 3:
        chunk_allowed[:2, 2] = True

    allowed = chunk_allowed.repeat_interleave(spatial_blocks, dim=0).repeat_interleave(spatial_blocks, dim=1)
    return allowed.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1).contiguous()


def _select_topk_blocks(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    *,
    num_heads: int,
    allowed_mask: torch.Tensor,
    topk_ratio: float,
    spatial_blocks: int,
) -> torch.Tensor:
    """Select sparse block pairs inside the chunk-causal allowed region.

    FlashVSR's draft block mask selects top-k pairs per temporal chunk after
    pooling each `(2,8,8)` block. Keep that grouping for training as well, but
    do not add the spatial-local inference mask here.
    """
    batch, q_block_count, block_size, dim = q_blocks.shape
    kv_block_count = k_blocks.shape[1]
    head_dim = dim // num_heads
    if q_block_count % int(spatial_blocks) != 0:
        raise ValueError(f"q_block_count={q_block_count} is not divisible by spatial_blocks={spatial_blocks}")
    chunks = q_block_count // int(spatial_blocks)
    q_pool = q_blocks.mean(dim=2).view(batch, q_block_count, num_heads, head_dim).permute(0, 2, 1, 3)
    k_pool = k_blocks.mean(dim=2).view(batch, kv_block_count, num_heads, head_dim).permute(0, 2, 1, 3)
    scores = torch.einsum("bhqd,bhkd->bhqk", q_pool.float(), k_pool.float()) / math.sqrt(head_dim)
    scores = scores.masked_fill(~allowed_mask, -torch.inf)

    attn_map = torch.softmax(scores, dim=-1)
    flat = attn_map.view(batch, num_heads, chunks, spatial_blocks, kv_block_count).reshape(
        batch, num_heads, chunks, spatial_blocks * kv_block_count
    )
    topk = max(1, int((spatial_blocks * spatial_blocks) * float(topk_ratio)) - 1)
    apply_topk = min(flat.shape[-1] - 1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=-1, largest=True).values[..., -1:]
    selected_flat = flat > thresholds
    selected = selected_flat.view(batch, num_heads, chunks, spatial_blocks, kv_block_count).reshape(
        batch, num_heads, q_block_count, kv_block_count
    )
    selected &= allowed_mask

    # Guarantee every query block has at least one legal key block even when
    # top-k runs into all -inf scores after tail-drop masking.
    no_selection = ~selected.any(dim=-1, keepdim=True)
    if no_selection.any():
        first_legal = allowed_mask.float().argmax(dim=-1, keepdim=True)
        selected.scatter_(-1, first_legal, no_selection)
        selected &= allowed_mask
    return selected.contiguous()


def block_sparse_chunk_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    grid: Tuple[int, int, int],
    topk_ratio: float = 2.0,
    local_num: Optional[int] = None,
) -> torch.Tensor:
    """Stage2 training attention: chunk-causal mask + `(2,8,8)` block sparse."""
    if block_sparse_attn_func is None:
        raise RuntimeError("block_sparse_attn is unavailable; Stage2 block-sparse training cannot run.")

    batch, tokens, dim = q.shape
    frames, height, width = [int(item) for item in grid]
    if tokens != frames * height * width:
        raise ValueError(f"Token/grid mismatch: tokens={tokens}, grid={(frames, height, width)}")
    win = STAGE2_BLOCK_WINDOW
    win_f, win_h, win_w = win
    if frames % win_f != 0 or height % win_h != 0 or width % win_w != 0:
        raise ValueError(f"Stage2 grid must be divisible by {win}, got {(frames, height, width)}")

    q_blocks = _partition_3d_blocks(q.view(batch, frames, height, width, dim), win)
    k_blocks = _partition_3d_blocks(k.view(batch, frames, height, width, dim), win)
    v_blocks = _partition_3d_blocks(v.view(batch, frames, height, width, dim), win)
    if q_blocks.shape[2] != 128:
        raise ValueError(f"Unexpected block token count {q_blocks.shape[2]}; expected 128.")

    allowed = build_stage2_chunk_block_mask(
        batch_size=batch,
        num_heads=num_heads,
        latent_frames=frames,
        height_blocks=height // win_h,
        width_blocks=width // win_w,
        device=q.device,
        local_num=local_num,
    )
    block_mask = _select_topk_blocks(
        q_blocks,
        k_blocks,
        num_heads=num_heads,
        allowed_mask=allowed,
        topk_ratio=topk_ratio,
        spatial_blocks=(height // win_h) * (width // win_w),
    )

    q_reordered = rearrange(q_blocks, "b nb bs d -> b (nb bs) d")
    k_reordered = rearrange(k_blocks, "b nb bs d -> b (nb bs) d")
    v_reordered = rearrange(v_blocks, "b nb bs d -> b (nb bs) d")
    q_kernel = rearrange(q_reordered, "b s (n hd) -> (b s) n hd", n=num_heads)
    k_kernel = rearrange(k_reordered, "b s (n hd) -> (b s) n hd", n=num_heads)
    v_kernel = rearrange(v_reordered, "b s (n hd) -> (b s) n hd", n=num_heads)

    q_len = int(q_reordered.shape[1])
    k_len = int(k_reordered.shape[1])
    cu_q = torch.arange(0, (batch + 1) * q_len, q_len, device=q.device, dtype=torch.int32)
    cu_k = torch.arange(0, (batch + 1) * k_len, k_len, device=q.device, dtype=torch.int32)
    head_mask_type = torch.ones(num_heads, device=q.device, dtype=torch.int32)
    out = block_sparse_attn_func(
        q_kernel,
        k_kernel,
        v_kernel,
        cu_q,
        cu_k,
        head_mask_type,
        None,
        block_mask,
        q_len,
        k_len,
        0.0,
        deterministic=False,
        softmax_scale=None,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = rearrange(out.unsqueeze(0), "b s n hd -> b s (n hd)", n=num_heads)
    out_blocks = rearrange(out, "b (nb bs) d -> b nb bs d", nb=q_blocks.shape[1], bs=128)
    out_grid = _reverse_3d_blocks(out_blocks, win, (frames, height, width))
    return out_grid.view(batch, tokens, dim)


def stage2_self_attention_forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    grid = getattr(self, "_flashvsr_stage2_grid", None)
    if grid is None:
        raise RuntimeError("Stage2 attention requires _flashvsr_stage2_grid to be set before block forward.")
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)
    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    mode = getattr(self, "_flashvsr_stage2_attention_mode", "block_sparse_chunk_causal")
    if mode == "block_sparse_chunk_causal":
        out = block_sparse_chunk_causal_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            grid=grid,
            topk_ratio=getattr(self, "_flashvsr_stage2_topk_ratio", 2.0),
            local_num=getattr(self, "_flashvsr_stage2_local_num", None),
        )
    elif mode == "dense_full":
        out = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
    else:
        raise ValueError(f"Unsupported Stage2 attention mode: {mode}")
    return self.o(out)


def enable_stage2_causal_attention(
    dit: torch.nn.Module,
    *,
    mode: str = "block_sparse_chunk_causal",
    topk_ratio: float = 2.0,
    local_num: Optional[int] = None,
) -> torch.nn.Module:
    """Patch a WanModel instance in-place without changing its state dict."""
    for block in dit.blocks:
        self_attn = block.self_attn
        if not hasattr(self_attn, "_flashvsr_stage2_original_forward"):
            self_attn._flashvsr_stage2_original_forward = self_attn.forward
        self_attn._flashvsr_stage2_attention_mode = mode
        self_attn._flashvsr_stage2_topk_ratio = float(topk_ratio)
        self_attn._flashvsr_stage2_local_num = None if local_num is None else int(local_num)
        self_attn.forward = types.MethodType(stage2_self_attention_forward, self_attn)
    dit.flashvsr_stage2_attention_mode = mode
    dit.flashvsr_stage2_topk_ratio = float(topk_ratio)
    dit.flashvsr_stage2_local_num = None if local_num is None else int(local_num)
    return dit


def set_stage2_grid(dit: torch.nn.Module, grid: Tuple[int, int, int]) -> None:
    grid = tuple(int(v) for v in grid)
    for block in dit.blocks:
        block.self_attn._flashvsr_stage2_grid = grid
