import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange

from diffsynth.core import ModelConfig
from diffsynth.pipelines.wan_video import (
    WanVideoPipeline,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_ShapeChecker,
)
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.utils.data import save_video
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d


class WanFixedPromptUnit(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={},
            input_params_nega={},
            output_params=("context",),
        )

    def process(self, pipe):
        if pipe.fixed_prompt_tensor is None:
            pipe.fixed_prompt_tensor = torch.load(pipe.prompt_tensor_path, map_location="cpu")
        context = pipe.fixed_prompt_tensor.to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"context": context}


class WanFixedPromptPipeline(WanVideoPipeline):
    @staticmethod
    def from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=None,
        prompt_tensor_path=None,
    ):
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs or [],
            tokenizer_config=None,
        )
        pipe.__class__ = WanFixedPromptPipeline
        pipe.prompt_tensor_path = prompt_tensor_path
        pipe.fixed_prompt_tensor = None
        pipe.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanFixedPromptUnit(),
        ]
        pipe.post_units = []
        pipe.in_iteration_models = ("dit",)
        pipe.in_iteration_models_2 = tuple()
        pipe.compilable_models = ["dit"]
        return pipe

    @staticmethod
    def _model_fn_wan_fixed_prompt_t2v(
        dit,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        embedded_context: torch.Tensor,
    ):
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

        x = latents
        context = embedded_context
        if x.shape[0] != context.shape[0]:
            x = torch.concat([x] * context.shape[0], dim=0)

        x = dit.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        for block in dit.blocks:
            x = block(x, context, t_mod, freqs)

        x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x

    @torch.no_grad()
    def __call__(
        self,
        *,
        height: int,
        width: int,
        num_frames: int,
        seed: int = 0,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        output_type: str = "quantized",
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
        inputs_shared = {
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "framewise_decoding": False,
            "vace_reference_image": None,
        }
        inputs_posi = {}
        inputs_nega = {}
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        debug_context = inputs_posi.get("context")
        print(
            "[wan_fixed_prompt_t2v] after_units "
            f"shared_keys={sorted(inputs_shared.keys())} "
            f"posi_keys={sorted(inputs_posi.keys())} "
            f"context_type={type(debug_context).__name__ if debug_context is not None else 'None'} "
            f"context_shape={tuple(debug_context.shape) if torch.is_tensor(debug_context) else None}",
            flush=True,
        )

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        inputs_shared["latents"] = inputs_shared["noise"]
        raw_context = inputs_posi["context"]
        dit = models["dit"]
        embedded_context = dit.text_embedding(raw_context)
        print(
            "[wan_fixed_prompt_t2v] embedded_context "
            f"raw_shape={tuple(raw_context.shape)} "
            f"embedded_shape={tuple(embedded_context.shape)} "
            f"dtype={embedded_context.dtype}",
            flush=True,
        )
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self._model_fn_wan_fixed_prompt_t2v(
                dit=dit,
                latents=inputs_shared["latents"],
                timestep=timestep,
                embedded_context=embedded_context,
            )
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )

        self.load_models_to_device(["vae"])
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def build_model_configs(base_model_dir: str):
    return [
        ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
        ModelConfig(path=str(Path(base_model_dir) / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
    ]


def main():
    parser = argparse.ArgumentParser(description="Wan T2V baseline using fixed prompt tensor directly.")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()

    pipe = WanFixedPromptPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=build_model_configs(args.base_model_dir),
        prompt_tensor_path=args.prompt_tensor_path,
    )
    video = pipe(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        tiled=args.tiled,
    )
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=args.fps, quality=5)
    print(f"saved_video={output_path}")


if __name__ == "__main__":
    main()
