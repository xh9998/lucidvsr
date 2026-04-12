import math
import random
import sys
import types
from pathlib import Path
from typing import Union

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
        circular_lowpass_kernel,
        random_add_gaussian_noise_pt,
        random_add_poisson_noise_pt,
        random_mixed_kernels,
    )
    from basicsr.utils import DiffJPEG
    from basicsr.utils.img_process_util import filter2D
except Exception as exc:
    raise ImportError(
        "RealESRGAN degradation requires basicsr. Install it in the training environment."
    ) from exc


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "params_realesrgan_with_second.yaml"
)


class DegradationModel:
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
        flattened = {
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
            "kernel_list": kernel_info.get(
                "kernel_list",
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
            ),
            "kernel_prob": kernel_info.get("kernel_prob", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            "sinc_prob": kernel_info.get("sinc_prob", 0.1),
            "blur_sigma": kernel_info.get("blur_sigma", [0.2, 3]),
            "betag_range": kernel_info.get("betag_range", [0.5, 4]),
            "betap_range": kernel_info.get("betap_range", [1, 2]),
            "kernel_list2": kernel_info.get(
                "kernel_list2",
                ["iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"],
            ),
            "kernel_prob2": kernel_info.get("kernel_prob2", [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]),
            "sinc_prob2": kernel_info.get("sinc_prob2", 0.1),
            "blur_sigma2": kernel_info.get("blur_sigma2", [0.2, 1.5]),
            "betag_range2": kernel_info.get("betag_range2", [0.5, 4]),
            "betap_range2": kernel_info.get("betap_range2", [1, 2]),
            "final_sinc_prob": kernel_info.get("final_sinc_prob", 0.8),
        }

        flattened["device"] = device
        self.opt = flattened
        self.device = torch.device(device)
        self.jpeger = DiffJPEG(differentiable=False)
        self.scale = self.opt["scale"]

    def degrade_batch_consistent(self, images: Union[torch.Tensor, list], seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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

        out = filter2D(images, kernel)
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
            out = filter2D(out, kernel2)
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
                out = filter2D(out, sinc_kernel)
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
            else:
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=out.new_full((out.size(0),), jpeg_quality2))
                out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode=mode_final)
                out = filter2D(out, sinc_kernel)
        else:
            out = F.interpolate(out, size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]), mode="bicubic")

        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
        out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")

        if input_was_pil_list:
            result = []
            for idx in range(out.size(0)):
                out_np = (out[idx].clamp(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
                result.append(PILImageModule.fromarray(out_np))
            return result
        return out

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
