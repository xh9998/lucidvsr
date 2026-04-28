import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from ..core.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    CrossAttention,
    GateModule,
    Head,
    RMSNorm,
    WanModel,
    flash_attention,
    modulate,
    rope_apply,
    sinusoidal_embedding_1d,
)

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn = None
    FLASH_ATTN_2_AVAILABLE = False


def _get_flash_attn_varlen_func():
    if FLASH_ATTN_3_AVAILABLE and hasattr(flash_attn_interface, "flash_attn_varlen_func"):
        return flash_attn_interface.flash_attn_varlen_func
    if FLASH_ATTN_2_AVAILABLE and hasattr(flash_attn, "flash_attn_varlen_func"):
        return flash_attn.flash_attn_varlen_func
    return None


def _build_segment_token_lengths(
    *,
    batch_size: int,
    frames_after_patchify: int,
    h_tokens: int,
    w_tokens: int,
    segment_lengths: Optional[Sequence[Sequence[int]]],
    sequence_lengths: Optional[Sequence[int]],
) -> Optional[List[List[int]]]:
    if segment_lengths is None:
        return None
    if len(segment_lengths) != batch_size:
        raise ValueError(f"segment_lengths batch mismatch: len={len(segment_lengths)} batch_size={batch_size}")
    if sequence_lengths is not None and len(sequence_lengths) != batch_size:
        raise ValueError(f"sequence_lengths batch mismatch: len={len(sequence_lengths)} batch_size={batch_size}")

    spatial_tokens_per_frame = h_tokens * w_tokens
    per_sample_token_lengths: List[List[int]] = []
    for sample_index, one_sample in enumerate(segment_lengths):
        one_sample_lengths = [int(length) for length in one_sample if int(length) > 0]
        valid_frames = sum(one_sample_lengths)
        if sequence_lengths is not None and valid_frames != int(sequence_lengths[sample_index]):
            raise ValueError(
                f"segment_lengths={one_sample_lengths} do not sum to sequence_length={sequence_lengths[sample_index]}"
            )
        if valid_frames > frames_after_patchify:
            raise ValueError(
                f"segment_lengths={one_sample_lengths} exceed patched frames={frames_after_patchify}"
            )
        per_sample_token_lengths.append([length * spatial_tokens_per_frame for length in one_sample_lengths])
    return per_sample_token_lengths


def _pack_segments(
    tensor: torch.Tensor,
    per_sample_token_lengths: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]]:
    packed_segments: List[torch.Tensor] = []
    segment_slices: List[List[Tuple[int, int]]] = []
    sample_lens: List[int] = []

    for sample_index, token_lengths in enumerate(per_sample_token_lengths):
        sample_slices: List[Tuple[int, int]] = []
        sample_offset = 0
        for token_length in token_lengths:
            if token_length <= 0:
                continue
            segment = tensor[sample_index, sample_offset : sample_offset + token_length]
            packed_segments.append(segment)
            sample_slices.append((sample_offset, sample_offset + token_length))
            sample_lens.append(token_length)
            sample_offset += token_length
        segment_slices.append(sample_slices)

    if not packed_segments:
        raise ValueError("At least one valid segment is required for packed attention")

    packed = torch.cat(packed_segments, dim=0)
    sample_lens_tensor = torch.tensor(sample_lens, dtype=torch.int32, device=tensor.device)
    return packed, sample_lens_tensor, segment_slices


def _unpack_segments(
    packed: torch.Tensor,
    *,
    batch_size: int,
    total_tokens: int,
    dim: int,
    segment_slices: Sequence[Sequence[Tuple[int, int]]],
) -> torch.Tensor:
    output = packed.new_zeros((batch_size, total_tokens, dim))
    packed_offset = 0
    for sample_index, sample_slices in enumerate(segment_slices):
        for start, end in sample_slices:
            token_length = end - start
            output[sample_index, start:end] = packed[packed_offset : packed_offset + token_length]
            packed_offset += token_length
    return output


def _packed_varlen_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    per_sample_token_lengths: Sequence[Sequence[int]],
) -> torch.Tensor:
    flash_attn_varlen_func = _get_flash_attn_varlen_func()
    if flash_attn_varlen_func is None:
        raise RuntimeError(
            "Packed joint attention requires flash_attn_varlen_func in the flashvsr environment. "
            "This path does not fall back to dense attention."
        )

    packed_q, sample_lens, segment_slices = _pack_segments(q, per_sample_token_lengths)
    packed_k, _, _ = _pack_segments(k, per_sample_token_lengths)
    packed_v, _, _ = _pack_segments(v, per_sample_token_lengths)

    cu_seqlens = nn.functional.pad(torch.cumsum(sample_lens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(sample_lens.max().item())
    head_dim = packed_q.shape[-1] // num_heads

    packed_q = rearrange(packed_q, "s (n d) -> s n d", n=num_heads).contiguous()
    packed_k = rearrange(packed_k, "s (n d) -> s n d", n=num_heads).contiguous()
    packed_v = rearrange(packed_v, "s (n d) -> s n d", n=num_heads).contiguous()

    packed_output = flash_attn_varlen_func(
        q=packed_q,
        k=packed_k,
        v=packed_v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
    )
    if isinstance(packed_output, tuple):
        packed_output = packed_output[0]
    packed_output = packed_output.reshape(-1, num_heads * head_dim).contiguous()
    return _unpack_segments(
        packed_output,
        batch_size=q.shape[0],
        total_tokens=q.shape[1],
        dim=q.shape[2],
        segment_slices=segment_slices,
    )


class JointAttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        per_sample_token_lengths: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        if per_sample_token_lengths is None:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return _packed_varlen_flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            per_sample_token_lengths=per_sample_token_lengths,
        )


class JointSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attn = JointAttentionModule(self.num_heads)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        *,
        per_sample_token_lengths: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v, per_sample_token_lengths=per_sample_token_lengths)
        return self.o(x)


class JointDiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = JointSelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        *,
        per_sample_token_lengths: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(input_x, freqs, per_sample_token_lengths=per_sample_token_lengths),
        )
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class WanJointModelV1(WanModel):
    """
    Experimental paper-aligned image/video joint model path.

    This version uses packed varlen flash attention for self-attention when
    per-sample segment lengths are provided. It does not use dense masks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList(
            [
                JointDiTBlock(
                    has_image_input=block.cross_attn.has_image_input,
                    dim=block.dim,
                    num_heads=block.self_attn.num_heads,
                    ffn_dim=block.ffn_dim,
                    eps=block.norm1.eps,
                )
                for block in self.blocks
            ]
        )

    def _patchify_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = self.patch_embedding(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c")
        return x, (f, h, w)

    def _build_per_sample_token_lengths(
        self,
        *,
        batch_size: int,
        frames_after_patchify: int,
        h_tokens: int,
        w_tokens: int,
        segment_lengths: Optional[Sequence[Sequence[int]]],
        sequence_lengths: Optional[Sequence[int]],
    ) -> Optional[List[List[int]]]:
        return _build_segment_token_lengths(
            batch_size=batch_size,
            frames_after_patchify=frames_after_patchify,
            h_tokens=h_tokens,
            w_tokens=w_tokens,
            segment_lengths=segment_lengths,
            sequence_lengths=sequence_lengths,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        segment_lengths: Optional[Sequence[Sequence[int]]] = None,
        sequence_lengths: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = self._patchify_tokens(x)
        per_sample_token_lengths = self._build_per_sample_token_lengths(
            batch_size=x.shape[0],
            frames_after_patchify=f,
            h_tokens=h,
            w_tokens=w,
            segment_lengths=segment_lengths,
            sequence_lengths=sequence_lengths,
        )

        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)

        for block in self.blocks:
            if self.training:

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

                x = gradient_checkpoint_forward(
                    block_forward,
                    use_gradient_checkpointing,
                    use_gradient_checkpointing_offload,
                    x,
                )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    per_sample_token_lengths=per_sample_token_lengths,
                )

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


def build_joint_wan_from_existing(dit: WanModel) -> WanJointModelV1:
    ffn_dim = dit.blocks[0].ffn_dim
    eps = dit.blocks[0].norm1.eps
    num_heads = dit.blocks[0].self_attn.num_heads
    out_dim = dit.head.head.out_features // math.prod(dit.patch_size)
    text_dim = dit.text_embedding[0].in_features
    joint_dit = WanJointModelV1(
        dim=dit.dim,
        in_dim=dit.in_dim,
        ffn_dim=ffn_dim,
        out_dim=out_dim,
        text_dim=text_dim,
        freq_dim=dit.freq_dim,
        eps=eps,
        patch_size=dit.patch_size,
        num_heads=num_heads,
        num_layers=len(dit.blocks),
        has_image_input=dit.has_image_input,
        has_image_pos_emb=getattr(dit, "has_image_pos_emb", False),
        has_ref_conv=getattr(dit, "has_ref_conv", False),
    ).to(device=next(dit.parameters()).device, dtype=next(dit.parameters()).dtype)
    joint_dit.load_state_dict(dit.state_dict(), strict=False)
    return joint_dit
