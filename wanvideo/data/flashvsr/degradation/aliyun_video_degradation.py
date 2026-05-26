import io
import logging
import math
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Union

import av
import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except Exception:
    import torchvision.transforms.functional as tvf

    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    mod.rgb_to_grayscale = tvf.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = mod

try:
    from PIL import Image as PILImageModule

    PILImageType = PILImageModule.Image
except Exception:
    PILImageModule = None
    PILImageType = ()

try:
    from basicsr.data.degradations import (
        add_gaussian_noise_pt,
        add_poisson_noise_pt,
        circular_lowpass_kernel,
        random_mixed_kernels,
    )
    from basicsr.utils import DiffJPEG, USMSharp
    from basicsr.utils.img_process_util import filter2D
except Exception as exc:
    raise ImportError(
        "Aliyun-style degradation requires basicsr. Install it in the training environment."
    ) from exc


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "params_aliyun_video_compression_v1.yaml"
)


class AliyunVideoCompressionDegradationModel:
    def __init__(self, opt=None, config_path=None, device=None):
        if device is None:
            try:
                worker_info = torch.utils.data.get_worker_info()
            except Exception:
                worker_info = None
            if worker_info is not None:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        if config_path is None:
            config_path = str(DEFAULT_CONFIG_PATH)
        if opt is None:
            opt = config_path

        if isinstance(opt, str):
            with open(opt, "r", encoding="utf-8") as file:
                cfg = yaml.safe_load(file)
        else:
            cfg = dict(opt)

        kernel_info = cfg.get("kernel_info", {})
        self.opt = {
            "degradation_type": cfg.get("degradation_type", "aliyun_video_compression_v1"),
            "scale": cfg.get("scale", 4),
            "resize_prob": cfg.get("resize_prob", [0.2, 0.7, 0.1]),
            "resize_range": cfg.get("resize_range", [0.15, 1.5]),
            "gaussian_noise_prob": cfg.get("gaussian_noise_prob", 0.5),
            "noise_range": cfg.get("noise_range", [1, 30]),
            "poisson_scale_range": cfg.get("poisson_scale_range", [0.05, 3.0]),
            "gray_noise_prob": cfg.get("gray_noise_prob", 0.4),
            "jpeg_range": cfg.get("jpeg_range", [30, 95]),
            "second_blur_prob": cfg.get("second_blur_prob", 0.8),
            "resize_prob2": cfg.get("resize_prob2", [0.3, 0.4, 0.3]),
            "resize_range2": cfg.get("resize_range2", [0.3, 1.2]),
            "gaussian_noise_prob2": cfg.get("gaussian_noise_prob2", 0.5),
            "noise_range2": cfg.get("noise_range2", [1, 25]),
            "poisson_scale_range2": cfg.get("poisson_scale_range2", [0.05, 2.5]),
            "gray_noise_prob2": cfg.get("gray_noise_prob2", 0.4),
            "jpeg_range2": cfg.get("jpeg_range2", [30, 95]),
            "codec": cfg.get("codec", ["libx264"]),
            "codec_prob": cfg.get("codec_prob", [1.0]),
            "bitrate": cfg.get("bitrate", [150000, 800000]),
            "codec2": cfg.get("codec2", ["libx264"]),
            "codec_prob2": cfg.get("codec_prob2", [1.0]),
            "bitrate2": cfg.get("bitrate2", [150000, 800000]),
            "disable_second_stage": bool(cfg.get("disable_second_stage", False)),
            "gt_usm": cfg.get("gt_usm", False),
            "kernel_list": kernel_info.get(
                "kernel_list",
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
            ),
            "kernel_prob": kernel_info.get("kernel_prob", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            "sinc_prob": kernel_info.get("sinc_prob", 0.1),
            "blur_sigma": kernel_info.get("blur_sigma", [0.2, 3.0]),
            "betag_range": kernel_info.get("betag_range", [0.5, 4.0]),
            "betap_range": kernel_info.get("betap_range", [1.0, 2.0]),
            "kernel_list2": kernel_info.get(
                "kernel_list2",
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
            ),
            "kernel_prob2": kernel_info.get("kernel_prob2", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            "sinc_prob2": kernel_info.get("sinc_prob2", 0.1),
            "blur_sigma2": kernel_info.get("blur_sigma2", [0.2, 1.5]),
            "betag_range2": kernel_info.get("betag_range2", [0.5, 4.0]),
            "betap_range2": kernel_info.get("betap_range2", [1.0, 2.0]),
            "final_sinc_prob": kernel_info.get("final_sinc_prob", 0.8),
        }
        self.device = torch.device(device)
        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpener = USMSharp()

    def degrade_batch_consistent(self, images: Union[torch.Tensor, List[Any]], seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda" and torch.cuda.is_available():
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

        params = self._sample_degradation_params(images.device)
        out = self._apply_degradation(images, params)

        if input_was_pil_list:
            result = []
            for idx in range(out.size(0)):
                out_np = (out[idx].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                result.append(PILImageModule.fromarray(out_np))
            return result
        return out

    def _sample_degradation_params(self, device: torch.device) -> Dict[str, Any]:
        kernel_range = [2 * v + 1 for v in range(3, 11)]
        pulse_tensor = torch.zeros(21, 21, device=device).float()
        pulse_tensor[10, 10] = 1

        kernel1 = self._sample_kernel(
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
            sinc_kernel = torch.FloatTensor(sinc_kernel).to(device)
        else:
            sinc_kernel = pulse_tensor

        first_updown = random.choices(["up", "down", "keep"], self.opt["resize_prob"])[0]
        second_updown = random.choices(["up", "down", "keep"], self.opt["resize_prob2"])[0]
        return {
            "kernel1": torch.FloatTensor(kernel1).unsqueeze(0).to(device),
            "kernel2": torch.FloatTensor(kernel2).unsqueeze(0).to(device),
            "sinc_kernel": sinc_kernel.unsqueeze(0),
            "resize_scale": self._sample_resize_scale(first_updown, self.opt["resize_range"]),
            "resize_mode": random.choice(["area", "bilinear", "bicubic"]),
            "gray": np.random.uniform() < self.opt["gaussian_noise_prob"],
            "gray_noise": (torch.rand(1, device=device) < self.opt["gray_noise_prob"]).float(),
            "sigma": torch.empty(1, device=device).uniform_(*self.opt["noise_range"]),
            "noise_scale": torch.empty(1, device=device).uniform_(*self.opt["poisson_scale_range"]),
            "jpeg_p": torch.empty(1, device=device).uniform_(*self.opt["jpeg_range"]),
            "codec": random.choices(self.opt["codec"], weights=self.opt["codec_prob"])[0],
            "bitrate": np.random.randint(self.opt["bitrate"][0], self.opt["bitrate"][1] + 1),
            "second_blur": np.random.uniform() < self.opt["second_blur_prob"],
            "resize_scale2": self._sample_resize_scale(second_updown, self.opt["resize_range2"]),
            "resize_mode2": random.choice(["area", "bilinear", "bicubic"]),
            "gray2": np.random.uniform() < self.opt["gaussian_noise_prob2"],
            "gray_noise2": (torch.rand(1, device=device) < self.opt["gray_noise_prob2"]).float(),
            "sigma2": torch.empty(1, device=device).uniform_(*self.opt["noise_range2"]),
            "noise_scale2": torch.empty(1, device=device).uniform_(*self.opt["poisson_scale_range2"]),
            "jpeg_p2": torch.empty(1, device=device).uniform_(*self.opt["jpeg_range2"]),
            "resize_mode3": random.choice(["area", "bilinear", "bicubic"]),
            "operations": random.sample([1, 2, 3], k=3),
            "codec2": random.choices(self.opt["codec2"], weights=self.opt["codec_prob2"])[0],
            "bitrate2": np.random.randint(self.opt["bitrate2"][0], self.opt["bitrate2"][1] + 1),
        }

    @staticmethod
    def _expand_batch_param(value: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device)
        else:
            tensor = torch.tensor(value, device=device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.numel() == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size)
        return tensor.reshape(batch_size)

    def _apply_degradation(self, gt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        self.jpeger = self.jpeger.to(gt.device)
        self.usm_sharpener = self.usm_sharpener.to(gt.device)
        if self.opt["gt_usm"]:
            gt = self.usm_sharpener(gt)

        ori_h, ori_w = gt.size()[2:4]
        out = filter2D(gt.contiguous(), params["kernel1"])
        out = F.interpolate(out, scale_factor=params["resize_scale"], mode=params["resize_mode"])
        batch_size = out.size(0)
        gray_noise = self._expand_batch_param(params["gray_noise"], batch_size, out.device)
        sigma = self._expand_batch_param(params["sigma"], batch_size, out.device)
        noise_scale = self._expand_batch_param(params["noise_scale"], batch_size, out.device)
        if params["gray"]:
            out = add_gaussian_noise_pt(
                out,
                sigma=sigma,
                clip=True,
                rounds=False,
                gray_noise=gray_noise,
            )
        else:
            out = add_poisson_noise_pt(
                out,
                scale=noise_scale,
                gray_noise=gray_noise,
                clip=True,
                rounds=False,
            )
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=params["jpeg_p"])
        out = self._apply_video_compression(out, params["codec"], params["bitrate"])

        if not self.opt["disable_second_stage"]:
            if params["second_blur"]:
                out = filter2D(out.contiguous(), params["kernel2"])
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
                out = add_gaussian_noise_pt(
                    out,
                    sigma=sigma2,
                    clip=True,
                    rounds=False,
                    gray_noise=gray_noise2,
                )
            else:
                out = add_poisson_noise_pt(
                    out,
                    scale=noise_scale2,
                    gray_noise=gray_noise2,
                    clip=True,
                    rounds=False,
                )

            for op_index in params["operations"]:
                if op_index == 1:
                    out = F.interpolate(
                        out,
                        size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                        mode=params["resize_mode3"],
                    )
                    out = filter2D(out.contiguous(), params["sinc_kernel"])
                elif op_index == 2:
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=params["jpeg_p2"])
                else:
                    out = self._apply_video_compression(out, params["codec2"], params["bitrate2"])
        else:
            out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode="bicubic")

        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
        return out

    def _apply_video_compression(self, images: torch.Tensor, codec: str, bitrate: int) -> torch.Tensor:
        frames = []
        for idx in range(images.shape[0]):
            frames.append(self._apply_video_compression_single(images[idx : idx + 1], codec=codec, bitrate=bitrate))
        return torch.cat(frames, dim=0)

    def _apply_video_compression_single(self, image_tensor: torch.Tensor, codec: str, bitrate: int) -> torch.Tensor:
        if image_tensor.dim() != 4 or image_tensor.shape[0] != 1:
            raise ValueError("Input tensor must have shape (1, C, H, W)")

        logging.getLogger("libav").setLevel(logging.CRITICAL)
        original_size = (image_tensor.shape[2], image_tensor.shape[3])
        even_size = (
            original_size[0] + (original_size[0] % 2),
            original_size[1] + (original_size[1] % 2),
        )
        resized = F.interpolate(image_tensor, size=even_size, mode="bilinear", align_corners=False)
        img = (resized.squeeze(0).detach().cpu().clamp(0, 1) * 255).byte().numpy()

        buf = io.BytesIO()
        with av.open(buf, "w", format="mp4") as container:
            stream = container.add_stream(codec, rate=1)
            stream.height = img.shape[1]
            stream.width = img.shape[2]
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = int(bitrate)

            img_hwc = np.transpose(img, (1, 2, 0))
            frame = av.VideoFrame.from_ndarray(img_hwc, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        buf.seek(0)
        decoded = None
        with av.open(buf, "r", format="mp4") as container:
            for frame in container.decode(video=0):
                decoded = frame.to_rgb().to_ndarray()
                break
        if decoded is None:
            raise RuntimeError("Video compression decode produced no frame.")

        out = torch.from_numpy(np.transpose(decoded, (2, 0, 1))).float().div_(255.0).unsqueeze(0)
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return out.to(image_tensor.device)

    def _sample_kernel(self, kernel_range, sinc_prob, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range):
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None,
            )
        pad_size = (21 - kernel_size) // 2
        return np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    def _sample_resize_scale(self, updown_type, resize_range):
        if updown_type == "up":
            return np.random.uniform(1, resize_range[1])
        if updown_type == "down":
            return np.random.uniform(resize_range[0], 1)
        return 1
