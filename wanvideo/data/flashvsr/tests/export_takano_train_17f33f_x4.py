import argparse
import json
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except Exception:
    import torchvision.transforms.functional as tvf

    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    mod.rgb_to_grayscale = tvf.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = mod

if "modelscope" not in sys.modules:
    stub = types.ModuleType("modelscope")

    def _snapshot_download(*args, **kwargs):
        raise RuntimeError("modelscope is not available in this environment, but snapshot_download was unexpectedly called.")

    stub.snapshot_download = _snapshot_download
    sys.modules["modelscope"] = stub

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.parquet_tar_dataset_v2 import FlashVSRParquetTarDatasetV2
from wanvideo.data.flashvsr.degradation.realesrgan_kernels import (
    DegradationModel,
    PILImageType,
    circular_lowpass_kernel,
    filter2D,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEGRADATION_CONFIG_PATH = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_realesrgan_with_second.yaml"
)
DEFAULT_TAKANO_METADATA_URL = (
    "s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/,"
    "s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/"
)


@dataclass
class ExportedSample:
    sample: Dict[str, Any]
    prefix: str


class DegradationModelX4Only(DegradationModel):
    """Original RealESRGAN-style degradation without the final bicubic restore."""

    def degrade_batch_consistent_x4(self, images, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

        input_was_pil_list = isinstance(images, list) and len(images) > 0 and isinstance(images[0], PILImageType)
        if input_was_pil_list:
            tensor_list = []
            for img in images:
                np_img = np.array(img, copy=False)
                if np_img.ndim == 2:
                    np_img = np.repeat(np_img[..., None], 3, axis=2)
                np_img = np_img.astype(np.float32) / 255.0
                tensor_list.append(torch.from_numpy(np_img).permute(2, 0, 1))
            images = torch.stack(tensor_list, dim=0)

        assert isinstance(images, torch.Tensor)
        assert images.dim() == 4
        images = images.to(self.device, non_blocking=True)

        _, _, ori_h, ori_w = images.size()
        kernel_range = [2 * v + 1 for v in range(3, 11)]
        pulse_tensor = torch.zeros(21, 21).float()
        pulse_tensor[10, 10] = 1

        kernel = self._sample_kernel(
            kernel_range,
            self.opt["sinc_prob"],
            self.opt["kernel_list"],
            self.opt["kernel_prob"],
            self.opt["blur_sigma"],
            self.opt["betag_range"],
            self.opt["betap_range"],
        )
        kernel2 = self._sample_kernel(
            kernel_range,
            self.opt["sinc_prob2"],
            self.opt["kernel_list2"],
            self.opt["kernel_prob2"],
            self.opt["blur_sigma2"],
            self.opt["betag_range2"],
            self.opt["betap_range2"],
        )

        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = pulse_tensor

        kernel = torch.FloatTensor(np.expand_dims(kernel, axis=0)).to(self.device)
        kernel2 = torch.FloatTensor(np.expand_dims(kernel2, axis=0)).to(self.device)
        sinc_kernel = torch.FloatTensor(np.expand_dims(sinc_kernel, axis=0)).to(self.device)
        self.jpeger = self.jpeger.to(self.device)

        updown_type1 = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
        scale_rand1 = self._sample_resize_scale(updown_type1, self.opt["resize_range"])
        mode1 = random.choice(["area", "bilinear", "bicubic"])
        use_gaussian_noise1 = np.random.uniform() < self.opt["gaussian_noise_prob"]
        noise_sigma1 = np.random.uniform(*self.opt["noise_range"]) if use_gaussian_noise1 else None
        poisson_scale1 = np.random.uniform(*self.opt["poisson_scale_range"]) if not use_gaussian_noise1 else None
        jpeg_quality1 = np.random.uniform(*self.opt["jpeg_range"])

        use_second_blur = np.random.uniform() < self.opt["second_blur_prob"]
        updown_type2 = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[0]
        scale_rand2 = self._sample_resize_scale(updown_type2, self.opt["resize_range2"])
        mode2 = random.choice(["area", "bilinear", "bicubic"])
        use_gaussian_noise2 = np.random.uniform() < self.opt["gaussian_noise_prob2"]
        noise_sigma2 = np.random.uniform(*self.opt["noise_range2"]) if use_gaussian_noise2 else None
        poisson_scale2 = np.random.uniform(*self.opt["poisson_scale_range2"]) if not use_gaussian_noise2 else None
        jpeg_quality2 = np.random.uniform(*self.opt["jpeg_range2"])
        final_order_sinc_first = np.random.uniform() < 0.5
        mode_final = random.choice(["area", "bilinear", "bicubic"])

        out = filter2D(images.contiguous(), kernel)
        out = F.interpolate(out, scale_factor=scale_rand1, mode=mode1)
        if use_gaussian_noise1:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=[noise_sigma1, noise_sigma1],
                clip=True,
                rounds=False,
                gray_prob=self.opt["gray_noise_prob"],
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=[poisson_scale1, poisson_scale1],
                gray_prob=self.opt["gray_noise_prob"],
                clip=True,
                rounds=False,
            )

        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality1))

        if use_second_blur:
            out = filter2D(out.contiguous(), kernel2)
            out = F.interpolate(
                out,
                size=(int(ori_h / self.opt["scale"] * scale_rand2), int(ori_w / self.opt["scale"] * scale_rand2)),
                mode=mode2,
            )
            if use_gaussian_noise2:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=[noise_sigma2, noise_sigma2],
                    clip=True,
                    rounds=False,
                    gray_prob=self.opt["gray_noise_prob2"],
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=[poisson_scale2, poisson_scale2],
                    gray_prob=self.opt["gray_noise_prob2"],
                    clip=True,
                    rounds=False,
                )

            if final_order_sinc_first:
                out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode=mode_final)
                out = filter2D(out.contiguous(), sinc_kernel)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
            else:
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
                out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode=mode_final)
                out = filter2D(out.contiguous(), sinc_kernel)
        else:
            out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode="bicubic")

        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        if input_was_pil_list:
            pil_frames: List[Image.Image] = []
            out = out.detach().cpu().float()
            for frame in out:
                array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
                pil_frames.append(Image.fromarray(array))
            return pil_frames
        return out


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    tensor = video.detach().cpu().float().clamp(0, 1)
    frames: List[Image.Image] = []
    for frame in tensor:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _build_dataset(*, metadata_url: str, seed: int, height: int, width: int, num_frames: int, max_parquet_records: int | None):
    return FlashVSRParquetTarDatasetV2(
        metadata_url=metadata_url,
        metadata_source="takano",
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(160, num_frames),
        enable_degradation=False,
        degradation_config_path=None,
        global_seed=seed,
        output_tensors=True,
        max_parquet_records=max_parquet_records,
    )


def _collect_unique_samples(dataset, count: int) -> List[ExportedSample]:
    iterator = iter(dataset)
    collected: List[ExportedSample] = []
    seen = set()
    max_trials = max(count * 20, 64)
    for idx in range(max_trials):
        sample = next(iterator)
        sample_id = str(sample.get("sample_id"))
        if sample_id in seen:
            continue
        seen.add(sample_id)
        prefix = f"takano_{len(collected):02d}"
        collected.append(ExportedSample(sample=sample, prefix=prefix))
        if len(collected) >= count:
            break
    if len(collected) < count:
        raise RuntimeError(f"Not enough unique samples: got {len(collected)}, need {count}")
    return collected


def _save_video_frames(frames: List[Image.Image], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_video(frames, path, fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def _sample_seed_to_int(value: Any) -> int:
    if torch.is_tensor(value):
        value = value.detach().cpu().view(-1)[0].item()
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--metadata_url", default=DEFAULT_TAKANO_METADATA_URL)
    parser.add_argument("--degradation_config_path", default=DEFAULT_DEGRADATION_CONFIG_PATH)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames_full", type=int, default=33)
    parser.add_argument("--num_frames_short", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_parquet_records", type=int, default=512)
    args = parser.parse_args()

    if args.num_frames_full < args.num_frames_short:
        raise ValueError("num_frames_full must be >= num_frames_short")

    output_17 = os.path.join(args.output_root, "takano_train_17f")
    output_33 = os.path.join(args.output_root, "takano_train_33f")
    os.makedirs(output_17, exist_ok=True)
    os.makedirs(output_33, exist_ok=True)

    dataset = _build_dataset(
        metadata_url=args.metadata_url,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames_full,
        max_parquet_records=args.max_parquet_records,
    )
    degrader = DegradationModelX4Only(config_path=args.degradation_config_path, device="cpu")

    chosen = _collect_unique_samples(dataset, count=args.num_samples)
    records: List[Dict[str, Any]] = []

    for exported in chosen:
        sample = exported.sample
        prefix = exported.prefix
        gt_full = sample["video"]
        gt_short = gt_full[: args.num_frames_short]
        sample_seed = _sample_seed_to_int(sample.get("sample_seed", args.seed))

        gt_full_frames = _tensor_to_pil_frames(gt_full)
        gt_short_frames = gt_full_frames[: args.num_frames_short]
        lq_full_frames = degrader.degrade_batch_consistent_x4(gt_full_frames, seed=sample_seed)
        lq_short_frames = lq_full_frames[: args.num_frames_short]

        paths = {
            "gt_17": os.path.join(output_17, "gt", f"{prefix}_gt.mp4"),
            "lq_x4_17": os.path.join(output_17, "lq_x4", f"{prefix}_lq_x4.mp4"),
            "gt_33": os.path.join(output_33, "gt", f"{prefix}_gt.mp4"),
            "lq_x4_33": os.path.join(output_33, "lq_x4", f"{prefix}_lq_x4.mp4"),
        }
        _save_video_frames(gt_short_frames, paths["gt_17"], fps=args.fps)
        _save_video_frames(lq_short_frames, paths["lq_x4_17"], fps=args.fps)
        _save_video_frames(gt_full_frames, paths["gt_33"], fps=args.fps)
        _save_video_frames(lq_full_frames, paths["lq_x4_33"], fps=args.fps)

        records.append(
            {
                "prefix": prefix,
                "sample_id": sample.get("sample_id"),
                "sample_seed": sample_seed,
                "media_path": sample.get("media_path"),
                "paths": paths,
            }
        )

    with open(os.path.join(args.output_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata_url": args.metadata_url,
                "degradation_config_path": args.degradation_config_path,
                "num_frames_full": args.num_frames_full,
                "num_frames_short": args.num_frames_short,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps({"output_root": args.output_root, "num_samples": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
