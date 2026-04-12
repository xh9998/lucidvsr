import argparse
import json
import os
import sys
import types

import torch

if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from einops import rearrange

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import VideoData
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v2 import (
    WanFixedPromptFlashVSRStage1Pipeline,
)
from wanvideo.model_inference.flashvsr.infer_flashvsr_stage1_v2 import (
    split_flashvsr_ckpt,
    infer_lq_proj_layer_num,
)


def build_model_configs(base_model_dir: str):
    return [
        ModelConfig(path=os.path.join(base_model_dir, "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=os.path.join(base_model_dir, "Wan2.1_VAE.pth")),
    ]


def tensor_stats(x: torch.Tensor):
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()),
        "abs_mean": float(xf.abs().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect V2 injection strength: compare x vs lq_latents stats.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
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
    lq_proj_state, lora_state, _ = split_flashvsr_ckpt(ckpt)
    lq_proj_layer_num = infer_lq_proj_layer_num(lq_proj_state)

    pipe = WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
        lq_proj_layer_num=lq_proj_layer_num,
    )
    pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=0)

    lq_video = VideoData(args.input_video, height=args.height, width=args.width).raw_data()
    lq_video = lq_video[: args.num_frames]
    if len(lq_video) == 0:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")

    lq_video = pipe.preprocess_video(lq_video).to(device=pipe.device, dtype=pipe.torch_dtype)
    lq_latents = pipe.lq_proj_in(lq_video)
    if lq_latents is None or len(lq_latents) == 0:
        raise RuntimeError("lq_latents is empty")

    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    length = (args.num_frames - 1) // 4 + 1
    noise_shape = (
        1,
        pipe.vae.model.z_dim,
        length,
        args.height // pipe.vae.upsampling_factor,
        args.width // pipe.vae.upsampling_factor,
    )
    latents = pipe.generate_noise(noise_shape, seed=args.seed, rand_device="cpu").to(device=pipe.device, dtype=pipe.torch_dtype)

    patchified = pipe.dit.patchify(latents)
    if isinstance(patchified, tuple):
        x, (f, h, w) = patchified
    else:
        x = patchified
        if x.ndim == 5:
            _, _, f, h, w = x.shape
            x = rearrange(x, "b c f h w -> b (f h w) c")
        elif x.ndim == 3:
            f = latents.shape[2] // pipe.dit.patch_size[0]
            h = latents.shape[3] // pipe.dit.patch_size[1]
            w = latents.shape[4] // pipe.dit.patch_size[2]
        else:
            raise ValueError(f"Unsupported patchify output shape: {tuple(x.shape)}")

    tokens_per_frame = h * w
    expected_tokens = x.shape[1]
    aligned = []
    for layer_latents in lq_latents:
        current_tokens = layer_latents.shape[1]
        if current_tokens < expected_tokens:
            pad_tokens = expected_tokens - current_tokens
            padding = torch.zeros(
                layer_latents.shape[0],
                pad_tokens,
                layer_latents.shape[2],
                device=layer_latents.device,
                dtype=layer_latents.dtype,
            )
            aligned.append(torch.cat([padding, layer_latents], dim=1))
        elif current_tokens > expected_tokens:
            trim_tokens = current_tokens - expected_tokens
            aligned.append(layer_latents[:, trim_tokens:, :])
        else:
            aligned.append(layer_latents)

    stats = {
        "checkpoint_path": args.checkpoint_path,
        "input_video": args.input_video,
        "num_inference_steps": args.num_inference_steps,
        "x": tensor_stats(x),
        "lq_latents_layer0": tensor_stats(aligned[0]),
        "ratio_std_lq_to_x": float(aligned[0].detach().float().std().item() / max(x.detach().float().std().item(), 1e-12)),
        "ratio_absmean_lq_to_x": float(aligned[0].detach().float().abs().mean().item() / max(x.detach().float().abs().mean().item(), 1e-12)),
        "grid": {"f": int(f), "h": int(h), "w": int(w), "tokens_per_frame": int(tokens_per_frame)},
        "num_lq_layers": len(aligned),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(args.output_json)


if __name__ == "__main__":
    main()
