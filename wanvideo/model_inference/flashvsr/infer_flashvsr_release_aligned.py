import argparse
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import imageio
import numpy as np
import torch
from einops import rearrange
from PIL import Image


def add_repo_to_path(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    utils_dir = os.path.join(repo_root, "examples", "WanVSR")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)


def ensure_modelscope_stub() -> None:
    if "modelscope" in sys.modules:
        return
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub


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


def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", os.path.basename(name))]


def largest_8n1_leq(n: int) -> int:
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device="cuda"):
    t = torch.from_numpy(np.asarray(img, np.uint8)).to(device=device, dtype=torch.float32)
    t = t.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
    return t.to(dtype)


def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH


def upscale_then_center_crop(img: Image.Image, scale: int, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW, sH = w0 * scale, h0 * scale
    up = img.resize((sW, sH), Image.BICUBIC)
    left = max(0, (sW - tW) // 2)
    top = max(0, (sH - tH) // 2)
    return up.crop((left, top, left + tW, top + tH))


def prepare_input_tensor(path: str, scale: int = 4, dtype=torch.bfloat16, device="cuda"):
    reader = imageio.get_reader(path)
    first = Image.fromarray(reader.get_data(0)).convert("RGB")
    w0, h0 = first.size
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}
    fps_val = meta.get("fps", 30)
    fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
    try:
        total = reader.count_frames()
    except Exception:
        total = meta.get("nframes", 0)
    if not total or total <= 0:
        idx = 0
        try:
            while True:
                reader.get_data(idx)
                idx += 1
        except Exception:
            total = idx

    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    indices = list(range(total)) + [total - 1] * 4
    F = largest_8n1_leq(len(indices))
    indices = indices[:F]
    frames = []
    try:
        for idx in indices:
            img = Image.fromarray(reader.get_data(idx)).convert("RGB")
            img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
    finally:
        try:
            reader.close()
        except Exception:
            pass
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid, tH, tW, F, fps


def save_video(frames, save_path, fps=30, quality=6):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    try:
        for frame in frames:
            writer.append_data(np.array(frame))
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Use original FlashVSR full pipeline style with stage1 ckpt.")
    parser.add_argument("--flashvsr_repo", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--prompt_tensor_path", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--sparse_ratio", type=float, default=2.0)
    parser.add_argument("--kv_ratio", type=float, default=3.0)
    parser.add_argument("--local_range", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--debug_dump_dir", type=str, default="")
    args = parser.parse_args()

    add_repo_to_path(args.flashvsr_repo)
    ensure_modelscope_stub()

    from diffsynth import FlashVSRFullPipeline, ModelManager
    from diffsynth.models.utils import load_state_dict
    from utils.utils import Causal_LQ4x_Proj

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    if args.debug_dump_dir:
        os.environ["FLASHVSR_DEBUG_DUMP_DIR"] = args.debug_dump_dir

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt = load_state_dict(args.checkpoint_path)
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)

    mm = ModelManager(torch_dtype=torch_dtype, device="cpu")
    mm.load_models(
        [
            os.path.join(args.base_model_dir, "diffusion_pytorch_model.safetensors"),
            os.path.join(args.base_model_dir, "Wan2.1_VAE.pth"),
        ]
    )
    if lora_state:
        mm.load_lora(state_dict=lora_state)

    pipe = FlashVSRFullPipeline.from_model_manager(mm, device=args.device)
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(args.device, dtype=torch_dtype)
    missing, unexpected = pipe.denoising_model().LQ_proj_in.load_state_dict(lq_proj_state, strict=False)
    pipe.denoising_model().LQ_proj_in.to(args.device)
    pipe.vae.model.encoder = None
    pipe.vae.model.conv1 = None
    pipe.to(args.device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    context_tensor = torch.load(args.prompt_tensor_path, map_location="cpu")
    pipe.init_cross_kv(context_tensor=context_tensor)
    pipe.load_models_to_device(["dit", "vae"])

    lq_video, height, width, num_frames, fps = prepare_input_tensor(
        args.input_video,
        scale=args.scale,
        dtype=torch_dtype,
        device=args.device,
    )

    result = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=False,
        LQ_video=lq_video,
        num_frames=num_frames,
        height=height,
        width=width,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=args.sparse_ratio * 768 * 1280 / (height * width),
        kv_ratio=args.kv_ratio,
        local_range=args.local_range,
        color_fix=True,
    )

    sr_path = os.path.join(args.output_dir, "sr.mp4")
    lq_path = os.path.join(args.output_dir, "input_lq.mp4")
    meta_path = os.path.join(args.output_dir, "meta.json")
    save_video(result, sr_path, fps=fps, quality=6)
    save_video(rearrange(lq_video[0], "c t h w -> t h w c").add(1).mul(127.5).clamp(0, 255).byte().cpu().numpy(), lq_path, fps=fps, quality=6)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": args.checkpoint_path,
                "input_video": args.input_video,
                "num_inference_steps": args.num_inference_steps,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "lq_proj_keys": len(lq_proj_state),
                "lora_keys": len(lora_state),
                "other_keys": len(other_state),
                "missing_lq_proj_keys": missing,
                "unexpected_lq_proj_keys": unexpected,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_sr={sr_path}")
    print(f"saved_lq={lq_path}")
    print(f"saved_meta={meta_path}")


if __name__ == "__main__":
    main()
