import argparse
from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.utils.data import save_video


def main():
    parser = argparse.ArgumentParser(description="Pure Wan T2V with text prompt and cfg=1.")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(Path(args.base_model_dir) / "diffusion_pytorch_model.safetensors"), **vram_config),
            ModelConfig(path=str(Path(args.base_model_dir) / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
            ModelConfig(path=str(Path(args.base_model_dir) / "Wan2.1_VAE.pth"), **vram_config),
        ],
        tokenizer_config=ModelConfig(path=str(Path(args.base_model_dir) / "google/umt5-xxl")),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
    )
    video = pipe(
        prompt=prompt,
        negative_prompt="",
        seed=args.seed,
        tiled=True,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=args.fps, quality=5)
    print(f"saved_video={output_path}")


if __name__ == "__main__":
    main()
