import argparse
import os
import sys
import types
from typing import Dict, Tuple

import torch
from PIL import Image


if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import VideoData, save_video
from wanvideo.model_inference.flashvsr.color_fix import apply_color_fix
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v2 import WanFixedPromptFlashVSRStage1Pipeline


def split_flashvsr_ckpt(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    lq_proj_state = {}
    lora_state = {}
    other_state = {}
    for key, value in state_dict.items():
        if key.startswith("lq_proj_in."):
            lq_proj_state[key[len("lq_proj_in."):]] = value
        elif "lora_" in key:
            lora_state[key] = value
        else:
            other_state[key] = value
    return lq_proj_state, lora_state, other_state


def build_model_configs(base_model_dir: str):
    return [
        ModelConfig(path=os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=os.path.join(base_model_dir, "Wan2.1_VAE.pth")),
    ]


def infer_lq_proj_layer_num(lq_proj_state: Dict[str, torch.Tensor]) -> int | None:
    indices = []
    prefix = "linear_layers."
    for key in lq_proj_state:
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            layer_id = remainder.split(".", 1)[0]
            if layer_id.isdigit():
                indices.append(int(layer_id))
    if not indices:
        return None
    return max(indices) + 1


def load_lq_proj_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported lq_proj checkpoint format: {path}")
    return state


def load_lq_video_frames(input_video: str, height: int, width: int, upscale_factor: float):
    if abs(upscale_factor - 1.0) < 1e-8:
        frames = VideoData(input_video, height=height, width=width).raw_data()
        return frames, height, width

    raw_frames = VideoData(input_video).raw_data()
    if not raw_frames:
        return raw_frames, height, width
    scaled_frames = []
    out_w = max(1, int(round(raw_frames[0].size[0] * upscale_factor)))
    out_h = max(1, int(round(raw_frames[0].size[1] * upscale_factor)))
    for frame in raw_frames:
        scaled_frames.append(frame.resize((out_w, out_h), Image.BICUBIC))
    return scaled_frames, out_h, out_w


def torch_dtype_from_name(torch_dtype_name: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map[torch_dtype_name]


def build_flashvsr_stage1_pipe(args):
    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)
    if not lq_proj_state and args.lq_proj_checkpoint:
        print(f"lq_proj_from_checkpoint={args.lq_proj_checkpoint}")
        lq_proj_state = load_lq_proj_checkpoint(args.lq_proj_checkpoint)
    inferred_lq_proj_layer_num = infer_lq_proj_layer_num(lq_proj_state)
    effective_lq_proj_layer_num = args.lq_proj_layer_num or inferred_lq_proj_layer_num

    print(f"checkpoint={args.checkpoint_path}")
    print(f"lq_proj_keys={len(lq_proj_state)} lora_keys={len(lora_state)} other_keys={len(other_state)}")
    print(f"lq_proj_layer_num={effective_lq_proj_layer_num}")
    if other_state:
        print("warning: ckpt contains extra keys that are not lq_proj_in or LoRA.")
        for key in list(sorted(other_state.keys()))[:20]:
            print(f"  extra_key={key}")

    pipe = WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
        lq_proj_layer_num=effective_lq_proj_layer_num,
    )
    pipe.lq_proj_scale = float(args.lq_proj_scale)

    if args.disable_projection:
        print("projection=disabled")
    else:
        missing, unexpected = pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
        if missing:
            print(f"warning: missing lq_proj keys: {len(missing)}")
        if unexpected:
            print(f"warning: unexpected lq_proj keys: {len(unexpected)}")

    if args.disable_lora:
        print("lora=disabled")
    elif lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)

    original_model_fn = pipe.model_fn

    if abs(args.projection_scale - 1.0) > 1e-8:
        print(f"projection_scale={args.projection_scale}")

        def scaled_model_fn(*model_args, **model_kwargs):
            lq_latents = model_kwargs.get("lq_latents")
            if lq_latents is not None:
                model_kwargs["lq_latents"] = [layer * args.projection_scale for layer in lq_latents]
            return original_model_fn(*model_args, **model_kwargs)

        pipe.model_fn = scaled_model_fn

    return pipe


def run_single_video(args, pipe):
    lq_video, effective_height, effective_width = load_lq_video_frames(
        args.input_video,
        args.height,
        args.width,
        args.input_bicubic_upscale,
    )
    lq_video = lq_video[: args.num_frames]
    if len(lq_video) == 0:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")
    if abs(args.input_bicubic_upscale - 1.0) > 1e-8:
        print(f"input_bicubic_upscale={args.input_bicubic_upscale}")
        print(f"effective_size={effective_width}x{effective_height}")

    sr_video = pipe.infer_from_lq(
        lq_video=lq_video,
        height=effective_height,
        width=effective_width,
        num_frames=len(lq_video),
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        tiled=args.tiled,
        output_type="quantized",
    )
    if not args.disable_color_fix:
        sr_video = apply_color_fix(sr_video, lq_video, method=args.color_fix_method)
        print(f"color_fix={args.color_fix_method}")

    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_video(sr_video, args.output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved_sr={args.output_video}")

    if args.save_input_lq:
        lq_output = os.path.splitext(args.output_video)[0] + "_lq.mp4"
        save_video(lq_video, lq_output, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"saved_lq={lq_output}")


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
    parser.add_argument("--lq_proj_checkpoint", type=str, default=None)
    parser.add_argument("--lq_proj_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--save_input_lq", action="store_true", default=False)
    parser.add_argument("--disable_lora", action="store_true", default=False)
    parser.add_argument("--disable_projection", action="store_true", default=False)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--tiled", action="store_true", default=False)
    parser.add_argument("--disable_color_fix", action="store_true", default=False)
    parser.add_argument("--color_fix_method", type=str, default="adain", choices=("adain", "wavelet"))
    parser.add_argument("--input_bicubic_upscale", type=float, default=1.0)


def main():
    parser = argparse.ArgumentParser(description="使用 V2: Wan fixed-prompt baseline + projection 的 Stage1 推理测试。")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    args = parser.parse_args()

    pipe = build_flashvsr_stage1_pipe(args)
    run_single_video(args, pipe)


if __name__ == "__main__":
    main()
