from typing import Dict, Optional

import torch
from PIL import Image

from .joint_batching_v1 import collate_image_video_joint_v1
from .streaming_dataset import FlashVSRStreamingDataset


class FlashVSRTarStreamingDatasetV3(FlashVSRStreamingDataset):
    """
    Tar-only streaming dataset for the current Yubari + picked17k workflow.

    Design intent:
    - no parquet metadata/index path at all
    - both sources are discovered purely from tar roots / tar manifests
    - keep the old WebDataset-style tar iteration / shuffle behavior
    - stay compatible with the existing non-packed training path

    Current behavior choice:
    - Yubari is treated as the video source
    - picked17k is treated as the image source
    - picked17k images are emitted as true single-frame samples (`f=1`)
    - therefore mixed image/video training should use packed collate
    """

    def __init__(
        self,
        yubari_video_tar_url: str,
        picked17k_image_tar_url: str,
        picked17k_dataset_prob: float,
        height: int,
        width: int,
        num_frames: int,
        stride: int = 1,
        max_source_frames: int = 160,
        enable_degradation: bool = True,
        degradation_seed: Optional[int] = None,
        hq_prefix_frames: int = 0,
        control_dropout_prob: float = 0.0,
        shuffle_buffer: int = 100,
        global_seed: Optional[int] = None,
        image_as_single_frame: bool = True,
        output_tensors: bool = True,
    ):
        if not yubari_video_tar_url:
            raise ValueError("yubari_video_tar_url is required for tar_v3")
        if not picked17k_image_tar_url:
            raise ValueError("picked17k_image_tar_url is required for tar_v3")
        if not 0.0 <= float(picked17k_dataset_prob) <= 1.0:
            raise ValueError(f"picked17k_dataset_prob must be in [0,1], got {picked17k_dataset_prob}")

        super().__init__(
            internal_url=yubari_video_tar_url,
            image_internal_url=picked17k_image_tar_url,
            image_dataset_prob=float(picked17k_dataset_prob),
            metadata_url=None,
            metadata_source="auto",
            max_parquet_records=None,
            min_overall_score=None,
            require_qwen35_parse_success=False,
            height=height,
            width=width,
            num_frames=num_frames,
            stride=stride,
            max_source_frames=max_source_frames,
            enable_degradation=enable_degradation,
            degradation_seed=degradation_seed,
            hq_prefix_frames=hq_prefix_frames,
            control_dropout_prob=control_dropout_prob,
            shuffle_buffer=shuffle_buffer,
            global_seed=global_seed,
            output_tensors=output_tensors,
        )
        self.image_as_single_frame = bool(image_as_single_frame)

        if self.parquet_records:
            raise RuntimeError("tar_v3 should not construct parquet records")

        if self.video_file_urls:
            raise ValueError(
                "tar_v3 only supports tar-based Yubari inputs, "
                f"but discovered non-tar video files: {self.video_file_urls[:8]}"
            )
        if self.image_file_urls:
            raise ValueError(
                "tar_v3 only supports tar-based picked17k inputs, "
                f"but discovered non-tar image files: {self.image_file_urls[:8]}"
            )
        if not self.video_tar_urls:
            raise ValueError(f"tar_v3 discovered no Yubari tar shards under: {yubari_video_tar_url}")
        if not self.image_tar_urls:
            raise ValueError(f"tar_v3 discovered no picked17k tar shards under: {picked17k_image_tar_url}")

        if self.output_tensors and self.image_as_single_frame:
            self.custom_collate_fn = collate_image_video_joint_v1

    def _process_image(self, image: Image.Image, sample_id: str, rng) -> Optional[Dict]:
        if not self.image_as_single_frame:
            return super()._process_image(image, sample_id=sample_id, rng=rng)

        try:
            frame = self.frame_processor(image.convert("RGB"))
            sample_seed = self._next_sample_seed(rng)
            lq_frames = self._build_lq_clip([frame], rng=rng, sample_seed=sample_seed)
            if self.output_tensors:
                return {
                    "video": torch.stack([self._pil_to_tensor(frame)], dim=0),
                    "lq_video": torch.stack([self._pil_to_tensor(lq_frames[0])], dim=0),
                    "sample_seed": torch.tensor(sample_seed, dtype=torch.long),
                    "sample_id": sample_id,
                    "source_type": "image",
                }
            return {
                "video": [frame],
                "lq_video": lq_frames,
                "sample_id": sample_id,
                "source_type": "image",
                "sample_seed": sample_seed,
            }
        except Exception as error:
            import warnings

            warnings.warn(f"Failed to process tar_v3 single-frame image sample: {error}")
            return None
