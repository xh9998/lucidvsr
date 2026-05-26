import argparse
import os
import sys
import types
from typing import Dict

import torch
from PIL import Image

if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is unavailable in this environment.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import VideoData, save_video
from wanvideo.model_inference.flashvsr.color_fix import apply_color_fix
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v5_3_lora import flashvsr_stage1_split_exported_state
from wanvideo.model_training.flashvsr.train_flashvsr_stage2_v6_1_lora import FlashVSRStage2Pipeline


def build_model_configs(base_model_dir: str):
    return [
        ModelConfig(path=os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=os.path.join(base_model_dir, "Wan2.1_VAE.pth")),
    ]


def infer_lq_proj_layer_num(lq_proj_state: Dict[str, torch.Tensor]) -> int | None:
    indices = []
    for key in lq_proj_state:
        if key.startswith("linear_layers."):
            layer_id = key[len("linear_layers.") :].split(".", 1)[0]
            if layer_id.isdigit():
                indices.append(int(layer_id))
    return None if not indices else max(indices) + 1


def load_lq_video_frames(input_video: str, height: int, width: int, upscale_factor: float):
    if abs(upscale_factor - 1.0) < 1e-8:
        return VideoData(input_video, height=height, width=width).raw_data(), height, width
    raw_frames = VideoData(input_video).raw_data()
    if not raw_frames:
        return raw_frames, height, width
    out_w = max(1, int(round(raw_frames[0].size[0] * upscale_factor)))
    out_h = max(1, int(round(raw_frames[0].size[1] * upscale_factor)))
    return [frame.resize((out_w, out_h), Image.BICUBIC) for frame in raw_frames], out_h, out_w


def torch_dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_stage2_pipe(args):
    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = flashvsr_stage1_split_exported_state(ckpt)
    effective_lq_proj_layer_num = args.lq_proj_layer_num or infer_lq_proj_layer_num(lq_proj_state)
    if effective_lq_proj_layer_num is None:
        raise ValueError("Cannot infer lq_proj_layer_num from checkpoint; pass --lq_proj_layer_num.")

    print(f"checkpoint={args.checkpoint_path}")
    print(f"lq_proj_keys={len(lq_proj_state)} lora_keys={len(lora_state)} other_keys={len(other_state)}")
    print(f"stage2_attention_mode={args.stage2_attention_mode} topk_ratio={args.stage2_topk_ratio} local_num={args.stage2_local_num}")
    pipe = FlashVSRStage2Pipeline.from_pretrained(
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
        lq_proj_layer_num=effective_lq_proj_layer_num,
        zero_init_lq_proj_in=False,
        stage2_attention_mode=args.stage2_attention_mode,
        stage2_topk_ratio=args.stage2_topk_ratio,
        stage2_local_num=args.stage2_local_num,
    )
    pipe.lq_proj_scale = float(args.lq_proj_scale)
    if lq_proj_state:
        pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)
    return pipe


def run_single_video(args, pipe):
    lq_video, effective_height, effective_width = load_lq_video_frames(
        args.input_video,
        args.height,
        args.width,
        args.input_bicubic_upscale,
    )
    lq_video = lq_video[: args.num_frames]
    if not lq_video:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")
    sr_video = pipe.infer_from_lq_streaming(
        lq_video=lq_video,
        height=effective_height,
        width=effective_width,
        num_frames=len(lq_video),
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        tiled=args.tiled,
        output_type="quantized",
        topk_ratio=args.stage2_topk_ratio,
        kv_ratio=args.stage2_kv_ratio,
    )
    if not args.disable_color_fix:
        if len(sr_video) != len(lq_video):
            aligned = min(len(sr_video), len(lq_video))
            print(f"[color_fix] frame_count_mismatch sr={len(sr_video)} lq={len(lq_video)} using={aligned}")
            sr_video = sr_video[:aligned]
            lq_for_color = lq_video[:aligned]
        else:
            lq_for_color = lq_video
        sr_video = apply_color_fix(sr_video, lq_for_color, method=args.color_fix_method)
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    save_video(sr_video, args.output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved_sr={args.output_video}")


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--lq_proj_scale", type=float, default=1.0)
    parser.add_argument(
        "--stage2_attention_mode",
        type=str,
        default="block_sparse_chunk_causal",
        choices=("block_sparse_chunk_causal", "block_sparse_official_mask", "dense_full"),
    )
    parser.add_argument("--stage2_topk_ratio", type=float, default=2.0)
    parser.add_argument("--stage2_local_num", type=int, default=-1)
    parser.add_argument("--stage2_kv_ratio", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--disable_color_fix", action="store_true", default=False)
    parser.add_argument("--color_fix_method", type=str, default="adain", choices=("adain", "wavelet"))
    parser.add_argument("--input_bicubic_upscale", type=float, default=4.0)


def main():
    parser = argparse.ArgumentParser(description="Stage2 v6.1 official-style streaming/cache inference.")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    args = parser.parse_args()
    pipe = build_stage2_pipe(args)
    run_single_video(args, pipe)


if __name__ == "__main__":
    main()
