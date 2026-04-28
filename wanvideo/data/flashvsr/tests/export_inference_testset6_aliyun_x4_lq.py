import argparse
import json
import os
import random
import sys
import types
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

from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.data.flashvsr.degradation.aliyun_video_degradation import AliyunVideoCompressionDegradationModel


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEGRADATION_CONFIG_PATH = str(
    REPO_ROOT / "wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1.yaml"
)
DEFAULT_TAKANO_MANIFEST = "/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
DEFAULT_YUBARI_ROOT = "conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"


def _looks_like_manifest(path: str) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(".txt") or lowered.endswith(".jsonl") or lowered.endswith(".manifest")


def _load_manifest_entries(path: str, *, seed: int, limit: int = 512) -> List[str]:
    entries: List[str] = []
    is_jsonl = str(path).lower().endswith(".jsonl")
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if is_jsonl:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    value = payload.get("path") or payload.get("url") or payload.get("media_url")
                else:
                    value = payload
                if value:
                    entries.append(str(value))
            else:
                entries.append(line)
    rng = random.Random(seed)
    rng.shuffle(entries)
    return entries[:limit]


def _resolve_dataset_url(url: str, *, seed: int) -> str:
    if os.path.isfile(url) and _looks_like_manifest(url):
        entries = _load_manifest_entries(url, seed=seed)
        if not entries:
            raise ValueError(f"Manifest has no entries: {url}")
        return ",".join(entries)
    return url


class AliyunVideoCompressionX4LQ(AliyunVideoCompressionDegradationModel):
    """Aliyun degradation without the final bicubic restore to GT resolution."""

    def _apply_degradation(self, gt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        self.jpeger = self.jpeger.to(gt.device)
        self.usm_sharpener = self.usm_sharpener.to(gt.device)
        if self.opt["gt_usm"]:
            gt = self.usm_sharpener(gt)

        ori_h, ori_w = gt.size()[2:4]
        out = self._first_stage(gt, params)

        if not self.opt["disable_second_stage"]:
            out = self._second_stage(out, params, ori_h=ori_h, ori_w=ori_w)
        else:
            out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode="bicubic")

        return torch.clamp((out * 255.0).round(), 0, 255) / 255.0

    def _first_stage(self, gt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        out = self._filter(gt, params["kernel1"])
        out = F.interpolate(out, scale_factor=params["resize_scale"], mode=params["resize_mode"])
        batch_size = out.size(0)
        gray_noise = self._expand_batch_param(params["gray_noise"], batch_size, out.device)
        sigma = self._expand_batch_param(params["sigma"], batch_size, out.device)
        noise_scale = self._expand_batch_param(params["noise_scale"], batch_size, out.device)
        if params["gray"]:
            from basicsr.data.degradations import add_gaussian_noise_pt

            out = add_gaussian_noise_pt(out, sigma=sigma, clip=True, rounds=False, gray_noise=gray_noise)
        else:
            from basicsr.data.degradations import add_poisson_noise_pt

            out = add_poisson_noise_pt(out, scale=noise_scale, gray_noise=gray_noise, clip=True, rounds=False)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=params["jpeg_p"])
        return self._apply_video_compression(out, params["codec"], params["bitrate"])

    def _second_stage(self, out: torch.Tensor, params: Dict[str, Any], *, ori_h: int, ori_w: int) -> torch.Tensor:
        if params["second_blur"]:
            out = self._filter(out, params["kernel2"])
        out = F.interpolate(
            out,
            size=(int(ori_h / self.opt["scale"] * params["resize_scale2"]), int(ori_w / self.opt["scale"] * params["resize_scale2"])),
            mode=params["resize_mode2"],
        )
        batch_size = out.size(0)
        gray_noise2 = self._expand_batch_param(params["gray_noise2"], batch_size, out.device)
        sigma2 = self._expand_batch_param(params["sigma2"], batch_size, out.device)
        noise_scale2 = self._expand_batch_param(params["noise_scale2"], batch_size, out.device)
        if params["gray2"]:
            from basicsr.data.degradations import add_gaussian_noise_pt

            out = add_gaussian_noise_pt(out, sigma=sigma2, clip=True, rounds=False, gray_noise=gray_noise2)
        else:
            from basicsr.data.degradations import add_poisson_noise_pt

            out = add_poisson_noise_pt(out, scale=noise_scale2, gray_noise=gray_noise2, clip=True, rounds=False)

        for op_index in params["operations"]:
            if op_index == 1:
                out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode=params["resize_mode3"])
                out = self._filter(out, params["sinc_kernel"])
            elif op_index == 2:
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=params["jpeg_p2"])
            else:
                out = self._apply_video_compression(out, params["codec2"], params["bitrate2"])
        return out

    @staticmethod
    def _filter(images: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        from basicsr.utils.img_process_util import filter2D

        return filter2D(images, kernel)


def _pil_frames_to_tensor(frames: List[Image.Image]) -> torch.Tensor:
    tensors = []
    for frame in frames:
        array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(tensors, dim=0)


def _tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    video = video.detach().cpu().float().clamp(0, 1)
    frames = []
    for frame in video:
        array = (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        frames.append(Image.fromarray(array))
    return frames


def _build_dataset(
    *,
    source: str,
    url: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
) -> FlashVSRStreamingDataset:
    resolved_url = _resolve_dataset_url(url, seed=seed)
    return FlashVSRStreamingDataset(
        internal_url=resolved_url,
        image_internal_url=None,
        image_dataset_prob=0.0,
        height=height,
        width=width,
        num_frames=num_frames,
        stride=1,
        max_source_frames=max(160, num_frames),
        enable_degradation=False,
        degradation_seed=None,
        hq_prefix_frames=0,
        control_dropout_prob=0.0,
        shuffle_buffer=100,
        global_seed=seed,
        output_tensors=False,
        metadata_url=None,
        metadata_source=source,
    )


def _collect_samples(dataset: FlashVSRStreamingDataset, *, count: int, source: str) -> List[Dict[str, Any]]:
    iterator = iter(dataset)
    samples: List[Dict[str, Any]] = []
    seen = set()
    trials = 0
    while len(samples) < count and trials < count * 200:
        trials += 1
        sample = next(iterator)
        sample_id = str(sample.get("sample_id") or f"{source}_{trials}")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        sample["source_dataset"] = source
        samples.append(sample)
    if len(samples) < count:
        raise RuntimeError(f"Only collected {len(samples)} samples for {source}, need {count}")
    return samples


def _save_video(frames: List[Image.Image], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_video(frames, path, fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])


def _degrade_x4(degrader: AliyunVideoCompressionX4LQ, frames: List[Image.Image], seed: int) -> List[Image.Image]:
    tensor = _pil_frames_to_tensor(frames).to(degrader.device)
    with torch.no_grad():
        lq = degrader.degrade_batch_consistent(tensor, seed=seed)
    return _tensor_to_pil_frames(lq)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 17-frame inference testset with Aliyun x4-small LQ.")
    parser.add_argument("--output_root", default="/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq")
    parser.add_argument("--takano_url", default=DEFAULT_TAKANO_MANIFEST)
    parser.add_argument("--yubari_url", default=DEFAULT_YUBARI_ROOT)
    parser.add_argument("--degradation_config_path", default=DEFAULT_DEGRADATION_CONFIG_PATH)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--num_per_source", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root)
    gt_dir = output_root / "gt"
    lq_dir = output_root / "lq"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    degrader = AliyunVideoCompressionX4LQ(config_path=args.degradation_config_path)
    datasets = {
        "takano": _build_dataset(
            source="takano",
            url=args.takano_url,
            seed=args.seed + 101,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        ),
        "yubari": _build_dataset(
            source="yubari",
            url=args.yubari_url,
            seed=args.seed + 202,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        ),
    }

    summary: Dict[str, Any] = {
        "output_root": str(output_root),
        "height": args.height,
        "width": args.width,
        "lq_height": args.height // 4,
        "lq_width": args.width // 4,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "seed": args.seed,
        "degradation_config_path": args.degradation_config_path,
        "lq_rule": "Aliyun degradation with final bicubic restore disabled; LQ remains 1/4 GT size.",
        "samples": [],
    }

    for source in ("takano", "yubari"):
        samples = _collect_samples(datasets[source], count=args.num_per_source, source=source)
        for index, sample in enumerate(samples):
            prefix = f"{source}_{index:02d}"
            frames = sample["video"]
            sample_seed = int(sample.get("sample_seed", args.seed + index))
            lq_frames = _degrade_x4(degrader, frames, seed=sample_seed)
            gt_path = gt_dir / f"{prefix}_gt.mp4"
            lq_path = lq_dir / f"{prefix}_lq.mp4"
            _save_video(frames, str(gt_path), fps=args.fps)
            _save_video(lq_frames, str(lq_path), fps=args.fps)
            item = {
                "prefix": prefix,
                "source_dataset": source,
                "sample_id": sample.get("sample_id"),
                "sample_seed": sample_seed,
                "gt_path": str(gt_path),
                "lq_path": str(lq_path),
                "gt_size": [args.width, args.height],
                "lq_size": [args.width // 4, args.height // 4],
            }
            summary["samples"].append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"[done] wrote {len(summary['samples'])} samples to {output_root}", flush=True)
    print(f"[done] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
