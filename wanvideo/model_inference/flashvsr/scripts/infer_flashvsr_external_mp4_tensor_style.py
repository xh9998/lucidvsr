import argparse
import json
import os
import sys
import types

import numpy as np
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
from wanvideo.model_training.flashvsr.train_flashvsr_stage1 import FlashVSRStage1Pipeline


def split_flashvsr_ckpt(state_dict):
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


def infer_lq_proj_layer_num(lq_proj_state):
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


def pil_frames_to_unit_tensor(frames):
    tensors = []
    for frame in frames:
        array = np.array(frame, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def main():
    parser = argparse.ArgumentParser(description="把外部 mp4 先转成训练同风格 tensor，再做 FlashVSR 推理。")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)
    inferred_lq_proj_layer_num = infer_lq_proj_layer_num(lq_proj_state)
    effective_lq_proj_layer_num = args.lq_proj_layer_num or inferred_lq_proj_layer_num
    print(f"checkpoint={args.checkpoint_path}")
    print(f"lq_proj_keys={len(lq_proj_state)} lora_keys={len(lora_state)} other_keys={len(other_state)}")
    print(f"lq_proj_layer_num={effective_lq_proj_layer_num}")

    pipe = FlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
        lq_proj_layer_num=effective_lq_proj_layer_num,
    )
    pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)

    frames = VideoData(args.input_video, height=args.height, width=args.width).raw_data()
    frames = frames[: args.num_frames]
    if not frames:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")
    lq_tensor = pil_frames_to_unit_tensor(frames)

    os.makedirs(args.output_dir, exist_ok=True)
    save_video(frames, os.path.join(args.output_dir, "input_lq.mp4"), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

    sr_frames = pipe.infer_from_lq(
        lq_video=lq_tensor.unsqueeze(0),
        height=args.height,
        width=args.width,
        num_frames=int(lq_tensor.shape[0]),
        seed=args.seed,
        rand_device="cpu",
        num_inference_steps=args.num_inference_steps,
        tiled=True,
        output_type="quantized",
    )
    save_video(sr_frames, os.path.join(args.output_dir, "sr.mp4"), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint_path": args.checkpoint_path,
                "input_video": args.input_video,
                "num_inference_steps": args.num_inference_steps,
                "seed": args.seed,
                "num_input_frames": int(lq_tensor.shape[0]),
                "lq_proj_layer_num": effective_lq_proj_layer_num,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved_dir={args.output_dir}")


if __name__ == "__main__":
    main()
