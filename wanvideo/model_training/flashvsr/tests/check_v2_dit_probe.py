import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator

from diffsynth import ModelConfig, load_state_dict
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.model_training.flashvsr.train_flashvsr_stage1_v2_debug_compare import (
    WanFixedPromptFlashVSRStage1Pipeline,
    collect_fixed_validation_samples,
    flashvsr_stage1_export,
    sinusoidal_embedding_1d,
)


def _save_tensor(root: str, name: str, tensor: torch.Tensor):
    os.makedirs(root, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(root, f"{name}.pt"))


def _load_args_from_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    merged = {}
    for section in ("data", "model", "validation"):
        merged.update(cfg.get(section, {}))
    return merged


def _build_fixed_sample(args_dict: dict):
    dataset = FlashVSRStreamingDataset(
        internal_url=args_dict.get("internal_url"),
        metadata_url=args_dict.get("metadata_url"),
        metadata_source=args_dict.get("metadata_source"),
        max_parquet_records=args_dict.get("max_parquet_records"),
        min_overall_score=args_dict.get("min_overall_score", -1.0),
        require_qwen35_parse_success=args_dict.get("require_qwen35_parse_success", False),
        image_internal_url=args_dict.get("image_internal_url", ""),
        image_dataset_prob=args_dict.get("image_dataset_prob", 0.0),
        height=args_dict["height"],
        width=args_dict["width"],
        num_frames=args_dict["num_frames"],
        stride=args_dict.get("stride", 1),
        max_source_frames=args_dict.get("max_source_frames"),
        enable_degradation=args_dict.get("enable_degradation", True),
        degradation_config_path=args_dict.get("degradation_config_path"),
        degradation_seed=args_dict.get("degradation_seed"),
        hq_prefix_frames=args_dict.get("hq_prefix_frames", 0),
        control_dropout_prob=args_dict.get("control_dropout_prob", 0.0),
        shuffle_buffer=args_dict.get("shuffle_buffer", 128),
        global_seed=args_dict.get("global_seed"),
        output_tensors=True,
    )
    return deepcopy(collect_fixed_validation_samples(dataset, 1)[0])


def _build_pipe(args_dict: dict, device: str):
    model_paths = args_dict["model_paths"] if isinstance(args_dict["model_paths"], list) else json.loads(args_dict["model_paths"])
    base_model_dir = str(Path(model_paths[0]).resolve().parent)
    return WanFixedPromptFlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=str(Path(base_model_dir) / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(Path(base_model_dir) / "Wan2.1_VAE.pth")),
        ],
        prompt_tensor_path=args_dict["prompt_tensor_path"],
        lq_proj_layer_num=args_dict.get("lq_proj_layer_num"),
    )


def _load_stage1_weights(pipe, ckpt_path: str):
    ckpt = load_state_dict(ckpt_path, device="cpu")
    exported = flashvsr_stage1_export(ckpt) if any(k.startswith("pipe.") for k in ckpt) else ckpt
    lq_proj_state = {k[len("lq_proj_in."):]: v for k, v in exported.items() if k.startswith("lq_proj_in.")}
    lora_state = {k: v for k, v in exported.items() if "lora_" in k}
    pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    pipe.clear_lora(verbose=0)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=0)


def _align_lq_latents(x: torch.Tensor, lq_latents):
    if x.ndim > 3:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
    _, expected_tokens, _ = x.shape
    if lq_latents is None:
        return None
    aligned = []
    for layer_latents in lq_latents:
        current_tokens = layer_latents.shape[1]
        if current_tokens == expected_tokens:
            aligned.append(layer_latents)
            continue
        raise ValueError(f"Unexpected token mismatch x={expected_tokens}, lq={current_tokens}")
    return aligned


def _probe_step0(pipe, lq_video: torch.Tensor, num_frames: int, height: int, width: int, seed: int, num_inference_steps: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

    noise = pipe.generate_noise(
        shape=(1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
        seed=seed,
        device=pipe.device,
        rand_device="cpu",
    )
    noise = noise.to(dtype=pipe.torch_dtype)
    latents = noise.clone()
    if pipe.fixed_prompt_tensor is None:
        if pipe.prompt_tensor_path is None:
            raise ValueError("prompt_tensor_path is required for fixed-prompt probing.")
        pipe.fixed_prompt_tensor = torch.load(pipe.prompt_tensor_path, map_location="cpu")
    raw_context = pipe.fixed_prompt_tensor.to(device=pipe.device, dtype=pipe.torch_dtype)
    embedded_context = pipe.dit.text_embedding(raw_context)

    lq_input = pipe.preprocess_video(lq_video).to(device=pipe.device, dtype=pipe.torch_dtype)
    lq_latents = pipe.lq_proj_in(lq_input)

    timestep = pipe.scheduler.timesteps[0].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
    dit = pipe.dit
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = embedded_context if embedded_context.ndim == 3 else embedded_context.unsqueeze(0)

    patchified = dit.patchify(latents)
    if isinstance(patchified, tuple):
        x, (f, h, w) = patchified
        if x.ndim == 5:
            x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], f * h * w, x.shape[1])
    else:
        x = patchified
        f = latents.shape[2] // dit.patch_size[0]
        h = latents.shape[3] // dit.patch_size[1]
        w = latents.shape[4] // dit.patch_size[2]
    if x.ndim > 3:
        x = x.reshape(x.shape[0], -1, x.shape[-1])
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    aligned_lq_latents = _align_lq_latents(x, lq_latents)
    block0_in = x.clone()
    if aligned_lq_latents is not None:
        block0_in = block0_in + aligned_lq_latents[0] * float(pipe.lq_proj_scale)
    block0_out = dit.blocks[0](block0_in, context, t_mod, freqs)

    _save_tensor(out_dir, "00_fixed_prompt_raw", raw_context)
    _save_tensor(out_dir, "00_embedded_context", context)
    _save_tensor(out_dir, "10_noise_init", noise)
    _save_tensor(out_dir, "11_latents_init", latents)
    _save_tensor(out_dir, "20_t", t)
    _save_tensor(out_dir, "21_t_mod", t_mod)
    _save_tensor(out_dir, "22_patchified_x", x)
    if aligned_lq_latents is not None:
        _save_tensor(out_dir, "23_aligned_lq_latents0", aligned_lq_latents[0])
    _save_tensor(out_dir, "24_block0_input_after_inject", block0_in)
    _save_tensor(out_dir, "25_block0_output", block0_out)

    with torch.no_grad():
        pipe.debug_tensor_dump_dir = out_dir
        pipe.infer_from_lq(
            lq_video=lq_video,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            rand_device="cpu",
            num_inference_steps=num_inference_steps,
            tiled=True,
            output_type="quantized",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dit_mode", choices=("eval", "train"), default="eval")
    parser.add_argument("--compare_to", default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    args_dict = _load_args_from_cfg(args.config)
    sample = _build_fixed_sample(args_dict)
    hr = sample["video"]
    lq = sample["lq_video"].unsqueeze(0)

    out_dir = os.path.join(args.output_dir, args.variant)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    pipe = _build_pipe(args_dict, device="cuda")
    pipe.lq_proj_scale = float(args_dict.get("lq_proj_scale", 1.0))
    _load_stage1_weights(pipe, args_dict["resume_stage1_checkpoint"])
    if args.dit_mode == "train":
        pipe.dit.train()
    else:
        pipe.dit.eval()
    pipe.lq_proj_in.eval()

    _probe_step0(
        pipe=pipe,
        lq_video=lq,
        num_frames=int(hr.shape[0]),
        height=int(hr.shape[-2]),
        width=int(hr.shape[-1]),
        seed=int(args_dict.get("global_seed", 20260407)),
        num_inference_steps=int(args_dict.get("validation_num_inference_steps", 50)),
        out_dir=out_dir,
    )

    if args.compare_to is not None:
        for key in ("20_t", "21_t_mod", "22_patchified_x", "23_aligned_lq_latents0", "24_block0_input_after_inject", "25_block0_output", "12_step0_noise_pred"):
            ref_path = os.path.join(args.compare_to, f"{key}.pt")
            cur_path = os.path.join(out_dir, f"{key}.pt")
            if not (os.path.exists(ref_path) and os.path.exists(cur_path)):
                continue
            ref = torch.load(ref_path, map_location="cpu").float()
            cur = torch.load(cur_path, map_location="cpu").float()
            diff = (ref - cur).abs()
            print(f"{key}: max={float(diff.max())} mean={float(diff.mean())}", flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
