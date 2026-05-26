import os
import random
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wanvideo.data.flashvsr.degradation import build_degradation_model

from .tar_streaming_dataset_v53 import FlashVSRTarStreamingDatasetV53


def _node_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _resolve_device(value: Optional[str], default: str = "cpu") -> str:
    value = (value or default).strip().lower()
    if value in ("", "cpu", "none"):
        return "cpu"
    if value in ("cuda", "gpu", "auto"):
        if torch.cuda.is_available():
            return f"cuda:{_node_local_rank()}"
        return "cpu"
    return value


class ConsistentClipDegradationOnDevice:
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        self.config_path = config_path
        self.device = device or "cpu"
        self.model = None
        self.model_pid = None

    def _get_model(self):
        current_pid = os.getpid()
        if self.model is None or self.model_pid != current_pid:
            device = _resolve_device(self.device)
            if device.startswith("cuda"):
                torch.cuda.set_device(int(device.split(":", 1)[1]))
            self.model = build_degradation_model(config_path=self.config_path, device=device)
            self.model_pid = current_pid
        return self.model

    def __getstate__(self):
        state = dict(self.__dict__)
        state["model"] = None
        state["model_pid"] = None
        return state

    def degrade_batch_consistent(self, images: List[Image.Image], seed: Optional[int] = None) -> List[Image.Image]:
        return self._get_model().degrade_batch_consistent(images, seed=seed)


class ConsistentClipGTSharpen:
    """Real-ESRGAN-style USM sharpening for the isolated Stage1 USM experiment."""

    def __init__(
        self,
        radius: int = 50,
        sigma: float = 0.0,
        weight: float = 0.5,
        threshold: float = 10.0,
        backend: str = "opencv",
        device: Optional[str] = None,
    ):
        if radius % 2 == 0:
            radius += 1
        self.radius = int(radius)
        self.sigma = float(sigma)
        self.weight = float(weight)
        self.threshold = float(threshold)
        self.backend = backend
        self.device = device or "cpu"

    def sharpen_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        if not images:
            return images
        if self.backend == "torch":
            return self._sharpen_batch_torch(images)
        return [self._sharpen_one_opencv(image) for image in images]

    def _sharpen_one_opencv(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        blur = cv2.GaussianBlur(array, (self.radius, self.radius), self.sigma)
        residual = array - blur
        mask = (np.abs(residual) * 255.0 > self.threshold).astype(np.float32)
        soft_mask = cv2.GaussianBlur(mask, (self.radius, self.radius), self.sigma)
        sharp = np.clip(array + self.weight * residual, 0.0, 1.0)
        output = soft_mask * sharp + (1.0 - soft_mask) * array
        output = (output * 255.0).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def _gaussian_kernel_1d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = self.sigma
        if sigma <= 0:
            sigma = 0.3 * ((self.radius - 1) * 0.5 - 1) + 0.8
        coords = torch.arange(self.radius, device=device, dtype=dtype) - (self.radius - 1) / 2
        kernel = torch.exp(-(coords * coords) / (2 * sigma * sigma))
        return kernel / kernel.sum()

    def _blur_torch(self, images: torch.Tensor) -> torch.Tensor:
        channels = images.shape[1]
        kernel_1d = self._gaussian_kernel_1d(images.device, images.dtype)
        kernel_h = kernel_1d.view(1, 1, self.radius, 1).repeat(channels, 1, 1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, self.radius).repeat(channels, 1, 1, 1)
        pad = self.radius // 2
        out = F.conv2d(F.pad(images, (0, 0, pad, pad), mode="reflect"), kernel_h, groups=channels)
        out = F.conv2d(F.pad(out, (pad, pad, 0, 0), mode="reflect"), kernel_w, groups=channels)
        return out

    def _sharpen_batch_torch(self, images: List[Image.Image]) -> List[Image.Image]:
        device = torch.device(_resolve_device(self.device))
        arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
        tensor = torch.from_numpy(np.stack(arrays, axis=0)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            blur = self._blur_torch(tensor)
            residual = tensor - blur
            mask = (residual.abs() * 255.0 > self.threshold).to(tensor.dtype)
            soft_mask = self._blur_torch(mask)
            sharp = (tensor + self.weight * residual).clamp(0.0, 1.0)
            output = soft_mask * sharp + (1.0 - soft_mask) * tensor
        output_np = (output.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        return [Image.fromarray(array) for array in output_np]


class FlashVSRTarStreamingDatasetV53USMGT(FlashVSRTarStreamingDatasetV53):
    """Isolated v5.3.5 Stage1 finetune dataset with sharpened GT.

    This class intentionally keeps all USM/GPU-degradation behavior out of the
    shared v5.3 dataset so unrelated Stage1/2/3 experiments keep their existing
    data path.
    """

    def __init__(
        self,
        *args,
        gt_sharpen: bool = True,
        gt_sharpen_backend: str = "opencv",
        gt_sharpen_device: Optional[str] = None,
        degradation_device: Optional[str] = None,
        **kwargs,
    ):
        enable_degradation = bool(kwargs.get("enable_degradation", True))
        degradation_config_path = kwargs.get("degradation_config_path")
        super().__init__(*args, **kwargs)
        self.gt_sharpener = (
            ConsistentClipGTSharpen(backend=gt_sharpen_backend, device=gt_sharpen_device)
            if gt_sharpen
            else None
        )
        if enable_degradation:
            self.degradation_model = ConsistentClipDegradationOnDevice(
                config_path=degradation_config_path,
                device=degradation_device,
            )

    def _build_gt_clip(self, hr_frames: List[Image.Image]) -> List[Image.Image]:
        if self.gt_sharpener is None:
            return [frame.copy() for frame in hr_frames]
        return self.gt_sharpener.sharpen_batch(hr_frames)

    def _process_video_bytes(self, video_bytes: bytes, sample_id: str, rng: random.Random):
        frames = self._extract_frames(video_bytes)
        if frames is None:
            return None
        clip = self._select_clip(frames, rng)
        if clip is None:
            return None
        sample_seed = self._next_sample_seed(rng)
        gt_clip = self._build_gt_clip(clip)
        return self._maybe_convert_output({
            "video": gt_clip,
            "lq_video": self._build_lq_clip(gt_clip, rng=rng, sample_seed=sample_seed),
            "sample_id": sample_id,
            "source_type": "video",
            "sample_seed": sample_seed,
        })

    def _process_image(self, image: Image.Image, sample_id: str, rng: random.Random):
        try:
            image = image.convert("RGB")
            if not self._meets_min_resolution(*image.size):
                return None
            sample_seed = self._next_sample_seed(rng)
            pseudo_rng = random.Random(sample_seed)
            frames = self.image_pseudo_video_generator.generate(image, seed=sample_seed, rng=pseudo_rng)
            frames = [self.frame_processor(frame) for frame in frames]
            gt_frames = self._build_gt_clip(frames)
            return self._maybe_convert_output({
                "video": gt_frames,
                "lq_video": self._build_lq_clip(gt_frames, rng=rng, sample_seed=sample_seed),
                "sample_id": sample_id,
                "source_type": "image",
                "sample_seed": sample_seed,
            })
        except Exception as error:
            import warnings

            warnings.warn(f"Failed to process paired image sample: {error}")
            return None
