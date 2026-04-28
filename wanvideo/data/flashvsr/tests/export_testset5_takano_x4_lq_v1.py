import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import DegradationModel
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import circular_lowpass_kernel
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import filter2D
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import random_add_gaussian_noise_pt
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import random_add_poisson_noise_pt


REPO_ROOT = Path(__file__).resolve().parents[4]
DEGRADATION_CONFIG_PATH = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_realesrgan_with_second.yaml"
)
TAKANO_METADATA_URL = (
    "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00000.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00001.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00000.parquet,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/00001.parquet"
)


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    tensor = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in tensor:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _save_video(frames: List[Image.Image], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_video(frames, path, fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def _build_dataset(seed: int, height: int, width: int, num_frames: int) -> FlashVSRParquetTarDatasetV2:
    return FlashVSRParquetTarDatasetV2(
        metadata_source="takano",
        metadata_url=TAKANO_METADATA_URL,
        takano_dataset_prob=1.0,
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(64, num_frames),
        enable_degradation=False,
        degradation_config_path=DEGRADATION_CONFIG_PATH,
        global_seed=seed,
        output_tensors=True,
        max_parquet_records=256,
    )


def _build_degrader() -> DegradationModel:
    return DegradationModel(config_path=DEGRADATION_CONFIG_PATH)


def _apply_degradation_without_restore(degrader: DegradationModel, video: torch.Tensor) -> torch.Tensor:
    frames = video.detach().float()
    if frames.ndim != 4:
        raise ValueError(f"Expected video tensor [T,C,H,W], got {tuple(frames.shape)}")
    images = frames.to(degrader.device, non_blocking=True)
    _, _, ori_h, ori_w = images.size()
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    pulse_tensor = torch.zeros(21, 21).float()
    pulse_tensor[10, 10] = 1

    kernel = degrader._sample_kernel(
        kernel_range,
        degrader.opt["sinc_prob"],
        degrader.opt["kernel_list"],
        degrader.opt["kernel_prob"],
        degrader.opt["blur_sigma"],
        degrader.opt["betag_range"],
        degrader.opt["betap_range"],
    )
    kernel2 = degrader._sample_kernel(
        kernel_range,
        degrader.opt["sinc_prob2"],
        degrader.opt["kernel_list2"],
        degrader.opt["kernel_prob2"],
        degrader.opt["blur_sigma2"],
        degrader.opt["betag_range2"],
        degrader.opt["betap_range2"],
    )

    if np.random.uniform() < degrader.opt["final_sinc_prob"]:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor

    kernel = torch.FloatTensor(np.expand_dims(kernel, axis=0)).to(degrader.device)
    kernel2 = torch.FloatTensor(np.expand_dims(kernel2, axis=0)).to(degrader.device)
    sinc_kernel = torch.FloatTensor(np.expand_dims(sinc_kernel, axis=0)).to(degrader.device)
    degrader.jpeger = degrader.jpeger.to(degrader.device)

    updown_type1 = random.choices(["up", "down", "keep"], degrader.opt["resize_prob"])[0]
    scale_rand1 = degrader._sample_resize_scale(updown_type1, degrader.opt["resize_range"])
    mode1 = random.choice(["area", "bilinear", "bicubic"])
    use_gaussian_noise1 = np.random.uniform() < degrader.opt["gaussian_noise_prob"]
    noise_sigma1 = np.random.uniform(*degrader.opt["noise_range"]) if use_gaussian_noise1 else None
    poisson_scale1 = np.random.uniform(*degrader.opt["poisson_scale_range"]) if not use_gaussian_noise1 else None
    jpeg_quality1 = np.random.uniform(*degrader.opt["jpeg_range"])

    use_second_blur = np.random.uniform() < degrader.opt["second_blur_prob"]
    updown_type2 = random.choices(["up", "down", "keep"], degrader.opt["resize_prob2"])[0]
    scale_rand2 = degrader._sample_resize_scale(updown_type2, degrader.opt["resize_range2"])
    mode2 = random.choice(["area", "bilinear", "bicubic"])
    use_gaussian_noise2 = np.random.uniform() < degrader.opt["gaussian_noise_prob2"]
    noise_sigma2 = np.random.uniform(*degrader.opt["noise_range2"]) if use_gaussian_noise2 else None
    poisson_scale2 = np.random.uniform(*degrader.opt["poisson_scale_range2"]) if not use_gaussian_noise2 else None
    jpeg_quality2 = np.random.uniform(*degrader.opt["jpeg_range2"])
    final_order_sinc_first = np.random.uniform() < 0.5
    mode_final = random.choice(["area", "bilinear", "bicubic"])

    out = filter2D(images, kernel)
    out = F.interpolate(out, scale_factor=scale_rand1, mode=mode1)
    if use_gaussian_noise1:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=[noise_sigma1, noise_sigma1],
            clip=True,
            rounds=False,
            gray_prob=degrader.opt["gray_noise_prob"],
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=[poisson_scale1, poisson_scale1],
            gray_prob=degrader.opt["gray_noise_prob"],
            clip=True,
            rounds=False,
        )

    out = torch.clamp(out, 0, 1)
    out = degrader.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality1))

    if use_second_blur:
        out = filter2D(out, kernel2)
        out = F.interpolate(
            out,
            size=(int(ori_h / degrader.opt["scale"] * scale_rand2), int(ori_w / degrader.opt["scale"] * scale_rand2)),
            mode=mode2,
        )
        if use_gaussian_noise2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=[noise_sigma2, noise_sigma2],
                clip=True,
                rounds=False,
                gray_prob=degrader.opt["gray_noise_prob2"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=[poisson_scale2, poisson_scale2],
                gray_prob=degrader.opt["gray_noise_prob2"],
                clip=True,
                rounds=False,
            )

        if final_order_sinc_first:
            out = F.interpolate(out, size=(ori_h // degrader.opt["scale"], ori_w // degrader.opt["scale"]), mode=mode_final)
            out = filter2D(out, sinc_kernel)
            out = torch.clamp(out, 0, 1)
            out = degrader.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
        else:
            out = torch.clamp(out, 0, 1)
            out = degrader.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
            out = F.interpolate(out, size=(ori_h // degrader.opt["scale"], ori_w // degrader.opt["scale"]), mode=mode_final)
            out = filter2D(out, sinc_kernel)
    else:
        out = F.interpolate(out, size=(ori_h // degrader.opt["scale"], ori_w // degrader.opt["scale"]), mode="bicubic")

    out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    return out.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 5 Takano clips with GT + x4-small LQ using original degradation without final bicubic restore.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    dataset = _build_dataset(seed=args.seed, height=args.height, width=args.width, num_frames=args.num_frames)
    degrader = _build_degrader()
    iterator = iter(dataset)

    gt_dir = os.path.join(args.output_root, "gt")
    lq_dir = os.path.join(args.output_root, "lq_x4")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    seen_ids = set()
    summary: Dict[str, Any] = {
        "output_root": args.output_root,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "samples": [],
    }

    trials = 0
    while len(summary["samples"]) < args.num_samples:
        trials += 1
        if trials > args.num_samples * 20:
            raise RuntimeError(f"Could not collect {args.num_samples} unique samples, got {len(summary['samples'])}")
        sample = next(iterator)
        sample_id = str(sample.get("sample_id"))
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)

        video = sample["video"][: args.num_frames]
        lq_small = _apply_degradation_without_restore(degrader, video)
        prefix = f"takano_{len(summary['samples']):02d}"
        gt_path = os.path.join(gt_dir, f"{prefix}_gt.mp4")
        lq_path = os.path.join(lq_dir, f"{prefix}_lq_x4.mp4")
        _save_video(_tensor_to_pil_frames(video), gt_path, args.fps)
        _save_video(_tensor_to_pil_frames(lq_small), lq_path, args.fps)

        summary["samples"].append(
            {
                "prefix": prefix,
                "sample_id": sample_id,
                "media_path": sample.get("media_path"),
                "fps": args.fps,
                "gt_path": gt_path,
                "lq_x4_path": lq_path,
                "gt_shape_tchw": list(video.shape),
                "lq_x4_shape_tchw": list(lq_small.shape),
            }
        )

    with open(os.path.join(args.output_root, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({"output_root": args.output_root, "num_samples": len(summary["samples"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
