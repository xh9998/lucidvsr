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


@torch.no_grad()
def build_official_spatial_local_mask(
    block_h: int,
    block_w: int,
    local_range: int = 9,
    *,
    device: torch.device,
) -> torch.Tensor:
    """FlashVSR official `build_local_block_mask_shifted_vec_normal_slide`.

    This is a spatial block mask over one latent chunk. It deliberately does
    not clamp at image borders, matching the official "normal_slide" variant.
    """
    rows = torch.arange(block_h, device=device)
    cols = torch.arange(block_w, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    row_all = yy.reshape(-1)
    col_all = xx.reshape(-1)
    half = int(local_range) // 2
    start_row = row_all - half
    end_row = start_row + int(local_range) - 1
    start_col = col_all - half
    end_col = start_col + int(local_range) - 1
    in_row = (row_all[None, :] >= start_row[:, None]) & (row_all[None, :] <= end_row[:, None])
    in_col = (col_all[None, :] >= start_col[:, None]) & (col_all[None, :] <= end_col[:, None])
    return in_row & in_col


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


def _select_topk_blocks_official(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    *,
    num_heads: int,
    allowed_mask: torch.Tensor,
    topk_ratio: float,
    chunks: int,
    spatial_blocks: int,
) -> torch.Tensor:
    """Official FlashVSR-style draft block selection.

    The older v6 implementation selected top-k keys independently for every
    query block. FlashVSR instead pools Q/K per `(2,8,8)` block, computes a
    coarse block attention map, then selects top-k block pairs per temporal
    chunk across all spatial query blocks. This function mirrors that grouping
    while still applying the full-sequence causal/local allowed mask.
    """
    batch, q_block_count, block_size, dim = q_blocks.shape
    kv_block_count = k_blocks.shape[1]
    head_dim = dim // num_heads
    if q_block_count != chunks * spatial_blocks:
        raise ValueError(
            f"Official block selection expects q_block_count=chunks*spatial_blocks, "
            f"got {q_block_count} vs {chunks}*{spatial_blocks}"
        )

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


def block_sparse_official_mask_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    grid: Tuple[int, int, int],
    topk_ratio: float = 2.0,
    local_num: Optional[int] = None,
    local_range: int = 9,
) -> torch.Tensor:
    """Probe-H path: official-style spatial local mask + chunk-grouped top-k.

    This keeps the full-sequence no-cache inference format from v6.2, but
    replaces the old per-query-block top-k selection with the official
    FlashVSR grouping used by `generate_draft_block_mask`.
    """
    if block_sparse_attn_func is None:
        raise RuntimeError("block_sparse_attn is unavailable; Stage2 block-sparse inference cannot run.")

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

    chunks = frames // win_f
    height_blocks = height // win_h
    width_blocks = width // win_w
    spatial_blocks = height_blocks * width_blocks
    causal_allowed = build_stage2_chunk_block_mask(
        batch_size=batch,
        num_heads=num_heads,
        latent_frames=frames,
        height_blocks=height_blocks,
        width_blocks=width_blocks,
        device=q.device,
        local_num=local_num,
    )
    spatial_local = build_official_spatial_local_mask(
        height_blocks,
        width_blocks,
        local_range=local_range,
        device=q.device,
    )
    spatial_local_full = spatial_local.repeat(chunks, chunks)
    allowed = causal_allowed & spatial_local_full.unsqueeze(0).unsqueeze(0)
    block_mask = _select_topk_blocks_official(
        q_blocks,
        k_blocks,
        num_heads=num_heads,
        allowed_mask=allowed,
        topk_ratio=topk_ratio,
        chunks=chunks,
        spatial_blocks=spatial_blocks,
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


def block_sparse_streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    grid: Tuple[int, int, int],
    pre_cache_k: Optional[torch.Tensor] = None,
    pre_cache_v: Optional[torch.Tensor] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official-style Stage2 streaming attention with per-block K/V cache.

    This mirrors FlashVSR inference semantics: each call receives either the
    first 6 latent frames or a following 2-latent-frame chunk. Causality comes
    from the fact that K/V only contains cached past blocks plus current blocks;
    block-sparse top-k then selects among those visible blocks.
    """
    if block_sparse_attn_func is None:
        raise RuntimeError("block_sparse_attn is unavailable; Stage2 streaming inference cannot run.")

    batch, tokens, dim = q.shape
    frames, height, width = [int(item) for item in grid]
    if tokens != frames * height * width:
        raise ValueError(f"Token/grid mismatch: tokens={tokens}, grid={(frames, height, width)}")
    if frames not in (2, 6):
        raise ValueError(f"Stage2 streaming expects f=6 for first chunk or f=2 afterward, got f={frames}")

    win = STAGE2_BLOCK_WINDOW
    win_f, win_h, win_w = win
    if frames % win_f != 0 or height % win_h != 0 or width % win_w != 0:
        raise ValueError(f"Stage2 grid must be divisible by {win}, got {(frames, height, width)}")

    q_blocks = _partition_3d_blocks(q.view(batch, frames, height, width, dim), win)
    k_blocks_cur = _partition_3d_blocks(k.view(batch, frames, height, width, dim), win)
    v_blocks_cur = _partition_3d_blocks(v.view(batch, frames, height, width, dim), win)
    if pre_cache_k is not None and pre_cache_v is not None:
        k_blocks = torch.cat([pre_cache_k, k_blocks_cur], dim=1)
        v_blocks = torch.cat([pre_cache_v, v_blocks_cur], dim=1)
    else:
        k_blocks = k_blocks_cur
        v_blocks = v_blocks_cur

    if q_blocks.shape[2] != 128 or k_blocks.shape[2] != 128:
        raise ValueError("Stage2 streaming block token count must be 128.")

    allowed = torch.ones(
        (batch, num_heads, q_blocks.shape[1], k_blocks.shape[1]),
        device=q.device,
        dtype=torch.bool,
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

    spatial_blocks = (height // win_h) * (width // win_w)
    keep_blocks = max(spatial_blocks, int(round(float(kv_ratio) * spatial_blocks)))
    cache_k = k_blocks[:, -keep_blocks:].detach()
    cache_v = v_blocks[:, -keep_blocks:].detach()
    return out_grid.view(batch, tokens, dim), cache_k, cache_v


def stage2_streaming_block_forward(
    block: torch.nn.Module,
    x: torch.Tensor,
    context: torch.Tensor,
    t_mod: torch.Tensor,
    freqs: torch.Tensor,
    *,
    grid: Tuple[int, int, int],
    pre_cache_k: Optional[torch.Tensor],
    pre_cache_v: Optional[torch.Tensor],
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
    ).chunk(6, dim=1)
    input_x = (block.norm1(x) * (1 + scale_msa) + shift_msa)

    self_attn = block.self_attn
    q = self_attn.norm_q(self_attn.q(input_x))
    k = self_attn.norm_k(self_attn.k(input_x))
    v = self_attn.v(input_x)
    q = rope_apply(q, freqs, self_attn.num_heads)
    k = rope_apply(k, freqs, self_attn.num_heads)
    attn_out, cache_k, cache_v = block_sparse_streaming_attention(
        q,
        k,
        v,
        num_heads=self_attn.num_heads,
        grid=grid,
        pre_cache_k=pre_cache_k,
        pre_cache_v=pre_cache_v,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
    )
    x = block.gate(x, gate_msa, self_attn.o(attn_out))
    x = x + block.cross_attn(block.norm3(x), context)
    input_x = (block.norm2(x) * (1 + scale_mlp) + shift_mlp)
    x = block.gate(x, gate_mlp, block.ffn(input_x))
    return x, cache_k, cache_v


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
    elif mode == "block_sparse_official_mask":
        out = block_sparse_official_mask_attention(
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
