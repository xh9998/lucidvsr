import os
import random
import tempfile
import warnings
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .streaming_dataset import VIDEO_EXTENSIONS, FlashVSRStreamingDataset, PseudoVideoGenerator


class FlashVSRTarStreamingDatasetV532YubariFrames(FlashVSRStreamingDataset):
    """
    Correct v5.3.2 experimental dataset:

    - video branch: one normal Yubari video clip
    - image branch: one single frame sampled from another Yubari video clip
    - the sampled frame is then expanded into one fake-image-video branch

    So each sample is still:
    - one video
    - one image pseudo-video
    """

    def __init__(
        self,
        image_tar_root_url: str,
        height: int,
        width: int,
        num_frames: int,
        yubari_video_tar_url: Optional[str] = None,
        takano_video_tar_url: Optional[str] = None,
        yubari_video_prob: Optional[float] = None,
        takano_video_prob: Optional[float] = None,
        stride: int = 1,
        max_source_frames: int = 160,
        enable_degradation: bool = True,
        degradation_seed: Optional[int] = None,
        hq_prefix_frames: int = 0,
        control_dropout_prob: float = 0.0,
        shuffle_buffer: int = 100,
        global_seed: Optional[int] = None,
        output_tensors: bool = True,
        image_branch_num_frames: Optional[int] = None,
    ):
        if not yubari_video_tar_url:
            raise ValueError("v5.3.2 yubari-frameimage dataset requires yubari_video_tar_url")
        super().__init__(
            internal_url=yubari_video_tar_url,
            image_internal_url=None,
            image_dataset_prob=0.0,
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
        if not self.video_urls and not self.video_manifest_urls:
            raise ValueError(f"v5.3.2 yubari-frameimage discovered no Yubari video source under: {yubari_video_tar_url}")
        if not self.output_tensors:
            raise ValueError("v5.3.2 yubari-frameimage requires output_tensors=True")
        self.image_branch_num_frames = (
            max(1, int(image_branch_num_frames))
            if image_branch_num_frames is not None
            else max(1, ((int(num_frames) - 1) // 4) + 1)
        )
        self.image_pseudo_video_generator = PseudoVideoGenerator(
            target_height=height,
            target_width=width,
            num_frames=self.image_branch_num_frames,
        )
        self.custom_collate_fn = self.paired_tensor_collate_fn

        # Explicitly ignored in this dedicated experimental branch.
        self._ignored_image_tar_root_url = image_tar_root_url
        self._ignored_takano_video_tar_url = takano_video_tar_url
        self._ignored_yubari_video_prob = yubari_video_prob
        self._ignored_takano_video_prob = takano_video_prob

    @staticmethod
    def paired_tensor_collate_fn(batch):
        if not batch:
            raise ValueError("Empty batch")
        tensor_keys = ("video", "lq_video", "image_video", "image_lq_video")
        output = {key: torch.stack([sample[key] for sample in batch], dim=0) for key in tensor_keys}
        passthrough_keys = (
            "sample_id",
            "image_sample_id",
            "source_type",
            "image_source_type",
            "source_dataset",
        )
        for key in passthrough_keys:
            output[key] = [sample.get(key) for sample in batch]
        return output

    @staticmethod
    def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
        array = frame.detach().cpu().permute(1, 2, 0).to(dtype=torch.float32).numpy()
        array = np.clip(np.round(array * 255.0), 0, 255).astype("uint8")
        return Image.fromarray(array)

    def _build_image_branch_from_single_frame(
        self,
        frame: torch.Tensor,
        rng: random.Random,
        sample_id: Optional[str],
    ) -> Dict[str, Any]:
        image = self._tensor_frame_to_pil(frame)
        sample_seed = self._next_sample_seed(rng)
        pseudo_rng = random.Random(sample_seed)
        frames = self.image_pseudo_video_generator.generate(image, seed=sample_seed, rng=pseudo_rng)
        frames = [self.frame_processor(one_frame) for one_frame in frames]
        processed = self._maybe_convert_output(
            {
                "video": frames,
                "lq_video": self._build_lq_clip(frames, rng=rng, sample_seed=sample_seed),
                "sample_id": sample_id,
                "source_type": "image_from_yubari_video_frame",
                "sample_seed": sample_seed,
            }
        )
        return processed

    def _build_image_branch_from_pil_frame(
        self,
        frame: Image.Image,
        rng: random.Random,
        sample_id: Optional[str],
    ) -> Dict[str, Any]:
        sample_seed = self._next_sample_seed(rng)
        pseudo_rng = random.Random(sample_seed)
        frames = self.image_pseudo_video_generator.generate(frame.convert("RGB"), seed=sample_seed, rng=pseudo_rng)
        frames = [self.frame_processor(one_frame) for one_frame in frames]
        processed = self._maybe_convert_output(
            {
                "video": frames,
                "lq_video": self._build_lq_clip(frames, rng=rng, sample_seed=sample_seed),
                "sample_id": sample_id,
                "source_type": "image_from_yubari_video_frame",
                "sample_seed": sample_seed,
            }
        )
        return processed

    def _extract_single_random_frame(self, video_bytes: bytes, rng: random.Random) -> Optional[Image.Image]:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                cap.release()
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 1:
                cap.release()
                return None
            frame_index = rng.randrange(1, frame_count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok or frame_bgr is None:
                return None
            frame_h, frame_w = frame_bgr.shape[:2]
            if not self._meets_min_resolution(frame_w, frame_h):
                return None
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        except Exception as error:
            warnings.warn(f"Failed to decode single frame for v5.3.2 image branch: {error}")
            return None
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _build_image_branch_from_video_bytes(
        self,
        video_bytes: bytes,
        rng: random.Random,
        sample_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        frame = self._extract_single_random_frame(video_bytes, rng=rng)
        if frame is None:
            return None
        return self._build_image_branch_from_pil_frame(
            frame,
            rng=rng,
            sample_id=sample_id,
        )

    def _iterate_yubari_video_bytes_from_tar_urls(
        self,
        tar_urls: List[str],
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        datapipe = self._make_torchdata_tar_pipe(list(tar_urls), rng=rng if self.global_seed is not None else None)
        while True:
            for file_name, stream_item in datapipe:
                if not str(file_name).endswith(VIDEO_EXTENSIONS):
                    continue
                yield {
                    "sample_id": os.path.basename(str(file_name)),
                    "video_bytes": stream_item.read(),
                }

    def _iterate_yubari_video_bytes_from_direct_urls(
        self,
        file_urls: List[str],
        manifest_urls: List[str],
        rng: random.Random,
    ) -> Iterator[Dict[str, Any]]:
        urls = self._split_for_process_and_worker(list(file_urls))
        for url in self._iter_deterministic_permutation(urls, rng) if urls else []:
            yield {
                "sample_id": os.path.basename(url),
                "video_bytes": self._open_binary(url),
            }
        if manifest_urls:
            for url in self._iter_manifest_entries(list(manifest_urls)):
                yield {
                    "sample_id": os.path.basename(url),
                    "video_bytes": self._open_binary(url),
                }

    def _image_source_iterator(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        iterators: List[Iterator[Dict[str, Any]]] = []
        if self.video_tar_urls:
            iterators.append(self._iterate_yubari_video_bytes_from_tar_urls(self.video_tar_urls, rng=rng))
        if self.video_file_urls or self.video_manifest_urls:
            iterators.append(
                self._iterate_yubari_video_bytes_from_direct_urls(
                    file_urls=list(self.video_file_urls),
                    manifest_urls=list(self.video_manifest_urls),
                    rng=rng,
                )
            )
        if not iterators:
            raise ValueError("v5.3.2 yubari-frameimage requires at least one Yubari video source")
        if len(iterators) == 1:
            while True:
                yield next(iterators[0])
        while True:
            yield next(iterators[rng.randrange(len(iterators))])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = self._make_iteration_rng()
        video_iter = self._video_iterator(rng=rng)
        image_source_iter = self._image_source_iterator(rng=rng)
        while True:
            video_sample = next(video_iter)
            image_branch = None
            image_source_sample = None
            while image_branch is None:
                image_source_sample = next(image_source_iter)
                image_branch = self._build_image_branch_from_video_bytes(
                    image_source_sample["video_bytes"],
                    rng=rng,
                    sample_id=image_source_sample.get("sample_id"),
                )
            yield {
                "video": video_sample["video"],
                "lq_video": video_sample["lq_video"],
                "image_video": image_branch["video"],
                "image_lq_video": image_branch["lq_video"],
                "sample_id": video_sample.get("sample_id"),
                "image_sample_id": image_source_sample.get("sample_id"),
                "source_type": video_sample.get("source_type", "video"),
                "image_source_type": image_branch.get("source_type", "image_from_yubari_video_frame"),
                "source_dataset": "yubari",
            }

    def validation_video_iterator(self, rng: Optional[random.Random] = None) -> Iterator[Dict[str, Any]]:
        yield from self._video_iterator(rng=rng or self._make_iteration_rng())
