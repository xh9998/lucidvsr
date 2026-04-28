import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from einops import rearrange
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import modulate, sinusoidal_embedding_1d
from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v5_3_lora import (
    FlashVSRStage1TrainingModule,
    _align_lq_latents_to_dit_tokens,
    _concat_lq_latent_layers,
    _latent_length_from_raw_frames,
    _lq_latent_length_from_raw_frames,
    _patchify_tokens_with_segments,
    _resolve_per_sample_token_lengths,
    _resolve_raw_segment_lengths,
    flashvsr_stage1_split_exported_state,
)


def _as_namespace(config: Dict) -> argparse.Namespace:
    flat = {}
    for section in ("data", "model", "train", "lora", "runtime"):
        flat.update(config.get(section, {}) or {})
    return argparse.Namespace(**flat)


def _make_dataset(args: argparse.Namespace) -> FlashVSRTarStreamingDatasetV53:
    return FlashVSRTarStreamingDatasetV53(
        yubari_video_tar_url=getattr(args, "yubari_video_tar_url", None),
        takano_video_tar_url=getattr(args, "takano_video_tar_url", None),
        image_tar_root_url=getattr(args, "image_tar_url"),
        yubari_video_prob=getattr(args, "yubari_video_prob", None),
        takano_video_prob=getattr(args, "takano_video_prob", None),
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        stride=int(getattr(args, "stride", 1)),
        max_source_frames=int(getattr(args, "max_source_frames", 160)),
        enable_degradation=bool(getattr(args, "enable_degradation", True)),
        degradation_config_path=getattr(args, "degradation_config_path", None),
        degradation_seed=getattr(args, "degradation_seed", None),
        hq_prefix_frames=int(getattr(args, "hq_prefix_frames", 0)),
        control_dropout_prob=float(getattr(args, "control_dropout_prob", 0.0)),
        shuffle_buffer=int(getattr(args, "shuffle_buffer", 100)),
        global_seed=int(getattr(args, "global_seed", 0)),
        output_tensors=True,
        image_branch_num_frames=int(getattr(args, "image_branch_num_frames", 5)),
    )


def _make_model(args: argparse.Namespace, checkpoint: str, device: str) -> FlashVSRStage1TrainingModule:
    model = FlashVSRStage1TrainingModule(
        model_paths=getattr(args, "model_paths"),
        prompt_tensor_path=getattr(args, "prompt_tensor_path"),
        trainable_models=getattr(args, "trainable_models", "lq_proj_in"),
        lora_base_model=getattr(args, "lora_base_model", "dit"),
        lora_target_modules=getattr(args, "lora_target_modules", "q,k,v,o"),
        lora_rank=int(getattr(args, "lora_rank", 384)),
        resume_stage1_checkpoint=checkpoint,
        lq_proj_layer_num=int(getattr(args, "lq_proj_layer_num", 1)),
        lq_proj_scale=float(getattr(args, "lq_proj_scale", 1.0)),
        lq_proj_temporal_mode=getattr(args, "lq_proj_temporal_mode", "streaming"),
        zero_init_lq_proj_in=False,
        freeze_lq_proj_in=False,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        image_video_joint_packed=True,
        device=device,
    )
    model.eval()
    model.pipe.dit.eval()
    model.pipe.lq_proj_in.eval()
    model.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
    return model


def _tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().float().cpu()
    if frame.ndim != 3:
        raise ValueError(f"Expected CHW frame, got {tuple(frame.shape)}")
    if frame.min() < -0.1:
        frame = (frame + 1.0) / 2.0
    frame = frame.clamp(0, 1)
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _save_video_tensor(video: torch.Tensor, path: Path, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if video.ndim == 5:
        video = video[0]
    frames = [_tensor_to_pil(frame) for frame in video]
    save_video(frames, str(path), fps=fps)


def _save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _normalize_map(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    lo = float(np.percentile(array, 1))
    hi = float(np.percentile(array, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((array - lo) / (hi - lo), 0, 1)


def _heatmap_image(array: np.ndarray, title: str) -> Image.Image:
    normed = _normalize_map(array)
    red = (normed * 255).astype(np.uint8)
    blue = ((1.0 - normed) * 255).astype(np.uint8)
    green = (np.sqrt(normed) * 180).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    image = Image.fromarray(rgb).resize((rgb.shape[1] * 4, rgb.shape[0] * 4), Image.BICUBIC)
    canvas = Image.new("RGB", (image.width, image.height + 18), "white")
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((4, 2), title, fill=(0, 0, 0))
    return canvas


def _save_token_norms(
    tokens: torch.Tensor,
    token_lengths: Sequence[int],
    h_tokens: int,
    w_tokens: int,
    out_dir: Path,
    prefix: str,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = tokens[0].detach().float().cpu()
    offset = 0
    summary = {"shape": list(tokens.shape), "segments": []}
    for seg_idx, token_len in enumerate(token_lengths):
        seg = sample[offset : offset + int(token_len)]
        offset += int(token_len)
        if seg.numel() == 0:
            summary["segments"].append(
                {
                    "segment_index": seg_idx,
                    "token_len": int(token_len),
                    "latent_frames": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "empty": True,
                }
            )
            continue
        if seg.shape[0] != int(token_len):
            summary["segments"].append(
                {
                    "segment_index": seg_idx,
                    "requested_token_len": int(token_len),
                    "available_token_len": int(seg.shape[0]),
                    "empty": False,
                    "truncated": True,
                }
            )
            token_len = int(seg.shape[0])
        frames = int(token_len) // (h_tokens * w_tokens)
        if frames <= 0 or frames * h_tokens * w_tokens != int(token_len):
            summary["segments"].append(
                {
                    "segment_index": seg_idx,
                    "token_len": int(token_len),
                    "latent_frames": frames,
                    "not_grid_aligned": True,
                }
            )
            continue
        norm = seg.norm(dim=-1).reshape(frames, h_tokens, w_tokens).numpy()
        summary["segments"].append(
            {
                "segment_index": seg_idx,
                "token_len": int(token_len),
                "latent_frames": frames,
                "mean": float(norm.mean()),
                "std": float(norm.std()),
                "min": float(norm.min()),
                "max": float(norm.max()),
            }
        )
        for t in range(frames):
            _heatmap_image(norm[t], f"{prefix} seg{seg_idx} t{t}").save(out_dir / f"{prefix}_seg{seg_idx}_t{t:02d}.png")
    return summary


def _align_lq_latent_layers(
    lq_latents: Sequence[torch.Tensor],
    expected_tokens: int,
    tokens_per_frame: int,
    raw_segment_lengths: Sequence[Sequence[int]],
) -> Tuple[List[torch.Tensor], List[Dict]]:
    details = []
    for layer_idx, layer_latents in enumerate(lq_latents):
        detail = {"layer": layer_idx, "before_tokens": int(layer_latents.shape[1]), "segments": []}
        for sample_index, one_sample_raw_lengths in enumerate(raw_segment_lengths):
            for segment_index, raw_frames in enumerate(one_sample_raw_lengths):
                dit_frames = _latent_length_from_raw_frames(int(raw_frames))
                lq_frames = _lq_latent_length_from_raw_frames(int(raw_frames))
                detail["segments"].append(
                    {
                        "sample_index": sample_index,
                        "segment_index": segment_index,
                        "raw_frames": int(raw_frames),
                        "dit_latent_frames": int(dit_frames),
                        "lq_latent_frames": int(lq_frames),
                        "pad_front_latent_frames": int(max(0, dit_frames - lq_frames)),
                    }
                )
        details.append(detail)
    aligned = _align_lq_latents_to_dit_tokens(
        lq_latents,
        expected_tokens=expected_tokens,
        tokens_per_frame=tokens_per_frame,
        raw_segment_lengths=raw_segment_lengths,
    )
    return aligned, details


def _decode_latents(pipe, latents: torch.Tensor, path: Path, fps: int) -> None:
    with torch.no_grad():
        decoded = pipe.vae.decode(latents, device=pipe.device, tiled=False)
    frames = pipe.vae_output_to_video(decoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(path), fps=fps)


def _run_dit_probe(pipe, merged_inputs: Dict, out_dir: Path) -> Dict:
    dit = pipe.dit
    latents = merged_inputs["latents"]
    timestep = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) // 2].unsqueeze(0).to(
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    if "embedded_context" in merged_inputs:
        context = merged_inputs["embedded_context"]
    else:
        context = merged_inputs["context"]
        if context.ndim == 2:
            context = context.unsqueeze(0)
        context = dit.text_embedding(context)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    raw_segment_lengths = merged_inputs["segment_lengths"]
    latent_segment_lengths = [
        [_latent_length_from_raw_frames(length) for length in sample]
        for sample in raw_segment_lengths
    ]
    x, (f, h, w) = _patchify_tokens_with_segments(dit, latents, latent_segment_lengths=latent_segment_lengths)
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
        sequence_lengths=merged_inputs["sequence_lengths"],
        segment_lengths=raw_segment_lengths,
    )
    if per_sample_token_lengths is None:
        raise RuntimeError("v5.3 probe expected packed segment token lengths")

    summaries = {
        "patch_grid": {"f": int(f), "h": int(h), "w": int(w)},
        "raw_segment_lengths": raw_segment_lengths,
        "latent_segment_lengths": latent_segment_lengths,
        "per_sample_token_lengths": per_sample_token_lengths,
        "stages": {},
    }
    token_lengths = per_sample_token_lengths[0]
    summaries["stages"]["patch_tokens_before_lq"] = _save_token_norms(
        x, token_lengths, h, w, out_dir / "token_norms", "patch_before_lq"
    )

    lq_latents = merged_inputs.get("lq_latents")
    if lq_latents is not None and lq_latents:
        lq0 = lq_latents[0]
        lq_total_tokens = int(lq0.shape[1])
        if lq_total_tokens % (h * w) == 0:
            summaries["stages"]["lq_proj_layer0_raw_all"] = _save_token_norms(
                lq0,
                [lq_total_tokens],
                h,
                w,
                out_dir / "token_norms",
                "lq_proj_layer0_raw_all",
            )
        lq_latents, alignment_details = _align_lq_latent_layers(
            lq_latents,
            expected_tokens=int(x.shape[1]),
            tokens_per_frame=int(h * w),
            raw_segment_lengths=raw_segment_lengths,
        )
        summaries["lq_alignment"] = alignment_details
        lq0 = lq_latents[0]
        summaries["stages"]["lq_proj_layer0"] = _save_token_norms(
            lq0, token_lengths, h, w, out_dir / "token_norms", "lq_proj_layer0"
        )
        x = x + lq0 * float(pipe.lq_proj_scale)
        summaries["stages"]["patch_tokens_after_lq"] = _save_token_norms(
            x, token_lengths, h, w, out_dir / "token_norms", "patch_after_lq"
        )

    capture_blocks = {0, 1, 5, 10, 20, len(dit.blocks) - 1}
    for block_id, block in enumerate(dit.blocks):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
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
        input_x = modulate(block.norm1(x), shift_msa, scale_msa)
        x = block.gate(
            x,
            gate_msa,
            block.self_attn(input_x, freqs, per_sample_token_lengths=per_sample_token_lengths),
        )
        x = x + block.cross_attn(block.norm3(x), context)
        input_x = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
        if block_id in capture_blocks:
            summaries["stages"][f"dit_block_{block_id:02d}"] = _save_token_norms(
                x, token_lengths, h, w, out_dir / "token_norms", f"dit_block_{block_id:02d}"
            )

    head_out = dit.head(x, t)
    summaries["stages"]["dit_head_tokens"] = _save_token_norms(
        head_out, token_lengths, h, w, out_dir / "token_norms", "dit_head"
    )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    cfg = _as_namespace(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] constructing dataset", flush=True)
    dataset = _make_dataset(cfg)
    print("[probe] constructing dataloader", flush=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.custom_collate_fn)
    print("[probe] fetching one batch", flush=True)
    batch = next(iter(loader))
    print("[probe] batch fetched", {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}, flush=True)

    print("[probe] saving input videos", flush=True)
    _save_video_tensor(batch["video"], output_dir / "inputs" / "video_gt.mp4", fps=args.fps)
    _save_video_tensor(batch["lq_video"], output_dir / "inputs" / "video_lq.mp4", fps=args.fps)
    _save_video_tensor(batch["image_video"], output_dir / "inputs" / "image_pseudo_gt.mp4", fps=args.fps)
    _save_video_tensor(batch["image_lq_video"], output_dir / "inputs" / "image_pseudo_lq.mp4", fps=args.fps)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[probe] constructing model on {device}", flush=True)
    model = _make_model(cfg, args.checkpoint, device=device)
    print("[probe] model constructed", flush=True)
    pipe = model.pipe
    pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)

    print("[probe] building pipeline inputs", flush=True)
    inputs = model.get_pipeline_inputs(batch)
    merged_inputs = model.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
    with torch.no_grad():
        for unit in pipe.units:
            print(f"[probe] running unit {unit.__class__.__name__}", flush=True)
            merged_inputs = pipe.unit_runner(unit, pipe, *merged_inputs)
    shared = merged_inputs[0]

    print("[probe] decoding VAE branch reconstructions", flush=True)
    video_latent_frames = _latent_length_from_raw_frames(int(batch["video"].shape[1]))
    image_latent_frames = _latent_length_from_raw_frames(int(batch["image_video"].shape[1]))
    input_latents = shared["input_latents"]
    _decode_latents(pipe, input_latents[:, :, :video_latent_frames], output_dir / "vae_recon" / "video_branch_recon.mp4", args.fps)
    _decode_latents(pipe, input_latents[:, :, video_latent_frames : video_latent_frames + image_latent_frames], output_dir / "vae_recon" / "image_branch_recon.mp4", args.fps)

    print("[probe] running DiT token probe", flush=True)
    summaries = _run_dit_probe(pipe, shared, output_dir)
    summaries["input_shapes"] = {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)}
    summaries["sample_ids"] = {
        "video": batch.get("sample_id"),
        "image": batch.get("image_sample_id"),
        "source_dataset": batch.get("source_dataset"),
    }
    summaries["checkpoint"] = args.checkpoint
    summaries["config"] = args.config
    _save_json(summaries, output_dir / "branch_separation_summary.json")
    print("[probe] completed", flush=True)
    print(json.dumps({"output_dir": str(output_dir), "summary": str(output_dir / "branch_separation_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
