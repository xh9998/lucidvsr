import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_ShapeChecker,
)
from diffsynth.utils.data import VideoData, save_video
from wanvideo.model_training.flashvsr.train_flashvsr_stage1 import (
    FlashVSRLQProjIn,
    flashvsr_stage1_model_fn,
    _build_release_style_lq_latents,
)


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


def infer_lq_proj_layer_num(lq_proj_state: Dict[str, torch.Tensor]) -> int | None:
    indices = []
    prefix = "linear_layers."
    for key in lq_proj_state:
        if key.startswith(prefix):
            layer_id = key[len(prefix):].split(".", 1)[0]
            if layer_id.isdigit():
                indices.append(int(layer_id))
    if not indices:
        return None
    return max(indices) + 1


def build_model_configs(base_model_dir: str):
    base = Path(base_model_dir)
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
    return [
        ModelConfig(path=str(base / "diffusion_pytorch_model.safetensors"), **vram_config),
        ModelConfig(path=str(base / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
        ModelConfig(path=str(base / "Wan2.1_VAE.pth"), **vram_config),
    ]


class WanAlignedLQVideoEmbedder:
    def process(self, pipe, lq_video, height, width):
        if lq_video is None:
            return {}
        if torch.is_tensor(lq_video):
            lq_video = pipe.preprocess_video(lq_video)
        else:
            resized = [frame.resize((width, height)) if frame.size != (width, height) else frame for frame in lq_video]
            lq_video = pipe.preprocess_video(resized)
        lq_input = lq_video.to(device=pipe.device, dtype=pipe.torch_dtype)
        if pipe.lq_proj_mode == "stream":
            lq_latents = _build_release_style_lq_latents(pipe.lq_proj_in, lq_input)
        else:
            lq_latents = pipe.lq_proj_in(lq_input)
        return {"lq_latents": lq_latents}


class WanAlignedFlashVSRStage1Pipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        base_model_dir: str,
        lq_proj_state: Dict[str, torch.Tensor],
        lora_state: Dict[str, torch.Tensor],
        lq_proj_layer_num: int,
        lq_proj_mode: str,
        torch_dtype=torch.bfloat16,
        device="cuda",
    ):
        base = Path(base_model_dir)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=build_model_configs(base_model_dir),
            tokenizer_config=ModelConfig(path=str(base / "google/umt5-xxl")),
            vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2,
        )
        pipe.__class__ = WanAlignedFlashVSRStage1Pipeline
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.model_fn = flashvsr_stage1_model_fn
        pipe.compilable_models = ["dit"]
        pipe.lq_proj_mode = lq_proj_mode
        pipe.lq_proj_in = FlashVSRLQProjIn(
            in_dim=3,
            out_dim=pipe.dit.dim,
            layer_num=int(lq_proj_layer_num),
            zero_init_output=False,
        ).to(device=device, dtype=torch_dtype)
        missing, unexpected = pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
        if missing:
            print(f"warning: missing lq_proj keys: {len(missing)}")
        if unexpected:
            print(f"warning: unexpected lq_proj keys: {len(unexpected)}")
        if lora_state:
            pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)
        pipe._lq_unit = WanAlignedLQVideoEmbedder()
        return pipe

    @torch.no_grad()
    def infer_from_lq(
        self,
        prompt: str,
        negative_prompt: str,
        lq_video,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        framewise_decoding: bool = False,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "input_video": None,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": framewise_decoding,
            "vace_reference_image": None,
            "sliding_window_size": None,
            "sliding_window_stride": None,
            "lq_video": lq_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        if "latents" not in inputs_shared:
            inputs_shared["latents"] = inputs_shared["noise"]
        inputs_shared.update(self._lq_unit.process(self, lq_video=inputs_shared["lq_video"], height=height, width=width))

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def main():
    parser = argparse.ArgumentParser(description="Wan-aligned FlashVSR Stage1 inference with text prompt and LQ projection.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=89)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lq_proj_layer_num", type=int, default=None)
    parser.add_argument("--lq_proj_mode", type=str, default="fullclip", choices=("fullclip", "stream"))
    parser.add_argument("--save_input_lq", action="store_true", default=False)
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)
    inferred_lq_proj_layer_num = infer_lq_proj_layer_num(lq_proj_state)
    effective_lq_proj_layer_num = args.lq_proj_layer_num or inferred_lq_proj_layer_num

    print(f"checkpoint={args.checkpoint_path}")
    print(f"lq_proj_keys={len(lq_proj_state)} lora_keys={len(lora_state)} other_keys={len(other_state)}")
    print(f"lq_proj_layer_num={effective_lq_proj_layer_num} lq_proj_mode={args.lq_proj_mode}")

    pipe = WanAlignedFlashVSRStage1Pipeline.from_pretrained(
        base_model_dir=args.base_model_dir,
        lq_proj_state=lq_proj_state,
        lora_state=lora_state,
        lq_proj_layer_num=effective_lq_proj_layer_num,
        lq_proj_mode=args.lq_proj_mode,
        torch_dtype=torch.bfloat16,
        device=args.device,
    )

    lq_video = VideoData(args.input_video, height=args.height, width=args.width).raw_data()
    lq_video = lq_video[: args.num_frames]
    if len(lq_video) == 0:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")

    sr_video = pipe.infer_from_lq(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        lq_video=lq_video,
        height=args.height,
        width=args.width,
        num_frames=len(lq_video),
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        output_type="quantized",
    )

    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(sr_video, str(output_path), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved_sr={output_path}")

    if args.save_input_lq:
        lq_output = output_path.with_name(output_path.stem + "_lq.mp4")
        save_video(lq_video, str(lq_output), fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
        print(f"saved_lq={lq_output}")


if __name__ == "__main__":
    main()
