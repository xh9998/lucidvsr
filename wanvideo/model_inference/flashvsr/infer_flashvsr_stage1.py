import argparse
import os
import sys
import types
from typing import Dict, Tuple

import torch


if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import VideoData, save_video
from wanvideo.model_training.flashvsr.train_flashvsr_stage1 import FlashVSRStage1Pipeline


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


def main():
    parser = argparse.ArgumentParser(description="使用 FlashVSR Stage1 训练 ckpt 做最小推理测试。")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--save_input_lq", action="store_true", default=False)
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
    if other_state:
        print("warning: ckpt contains extra keys that are not lq_proj_in or LoRA.")
        for key in list(sorted(other_state.keys()))[:20]:
            print(f"  extra_key={key}")

    pipe = FlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
        lq_proj_layer_num=effective_lq_proj_layer_num,
    )

    missing, unexpected = pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if missing:
        print(f"warning: missing lq_proj keys: {len(missing)}")
    if unexpected:
        print(f"warning: unexpected lq_proj keys: {len(unexpected)}")

    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)

    lq_video = VideoData(args.input_video, height=args.height, width=args.width).raw_data()
    lq_video = lq_video[: args.num_frames]
    if len(lq_video) == 0:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")

    sr_video = pipe.infer_from_lq(
        lq_video=lq_video,
        height=args.height,
        width=args.width,
        num_frames=len(lq_video),
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        output_type="quantized",
    )

    output_dir = os.path.dirname(args.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_video(sr_video, args.output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved_sr={args.output_video}")

    if args.save_input_lq:
        lq_output = os.path.splitext(args.output_video)[0] + "_lq.mp4"
        save_video(lq_video, lq_output, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"saved_lq={lq_output}")


if __name__ == "__main__":
    main()
