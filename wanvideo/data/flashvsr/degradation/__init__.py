from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .realesrgan_kernels import DegradationModel


def _load_degradation_cfg(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(Path(config_path), "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Degradation config must parse to dict: {config_path}")
    return payload


def build_degradation_model(config_path: Optional[str] = None, device=None):
    cfg = _load_degradation_cfg(config_path)
    degradation_type = cfg.get("degradation_type", "realesrgan_with_second")
    if degradation_type == "aliyun_video_compression_v1":
        from .aliyun_video_degradation import AliyunVideoCompressionDegradationModel

        return AliyunVideoCompressionDegradationModel(config_path=config_path, device=device)
    return DegradationModel(config_path=config_path, device=device)


__all__ = ["DegradationModel", "build_degradation_model"]
