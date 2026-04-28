import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from .joint_batching_v5 import collate_image_video_joint_v5
from .streaming_dataset import VIDEO_EXTENSIONS, FlashVSRStreamingDataset


def _projector_group_size_from_num_frames(num_frames: int) -> int:
    num_frames = max(1, int(num_frames))
    # Group images to match the latent-time count of the GT video branch:
    #   17 raw frames -> 5 latent-time slices
    #   89 raw frames -> 23 latent-time slices
    return max(1, ((num_frames - 1) // 4) + 1)


class FlashVSRTarStreamingDatasetV5(FlashVSRStreamingDataset):
    """
    Isolated V5 tar-only joint dataset.

    Intent:
    - keep old v2/v4 dataset paths untouched
    - use tar-only Yubari video + tar-only picked image workflow
    - treat images as true single-frame samples at the data level
    - group multiple images into one training sample so one grouped-image sample
      matches one video's projector-time count:
        projector_group_size = max(1, (num_frames - 1) // 4)

    Examples:
    - 17 frames -> 5 grouped images
    - 89 frames -> 23 grouped images
    """

    def __init__(
        self,
        picked17k_image_tar_url: str,
        picked17k_dataset_prob: float,
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
    ):
        if not yubari_video_tar_url and not takano_video_tar_url:
            raise ValueError("tar_v5 requires at least one video source")
        if not picked17k_image_tar_url:
            raise ValueError("picked17k_image_tar_url is required for tar_v5")
        if not 0.0 <= float(picked17k_dataset_prob) <= 1.0:
            raise ValueError(f"picked17k_dataset_prob must be in [0,1], got {picked17k_dataset_prob}")

        video_roots = [url for url in (yubari_video_tar_url, takano_video_tar_url) if url]
        combined_video_root = ",".join(video_roots) if video_roots else None
        self.yubari_video_prob, self.takano_video_prob = self._resolve_video_probs(
            yubari_video_tar_url=yubari_video_tar_url,
            takano_video_tar_url=takano_video_tar_url,
            yubari_video_prob=yubari_video_prob,
            takano_video_prob=takano_video_prob,
        )

        super().__init__(
            internal_url=combined_video_root,
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

        if self.parquet_records:
            raise RuntimeError("tar_v5 should not construct parquet records")
        if not self.image_tar_urls and not self.image_file_urls and not self.image_manifest_urls:
            raise ValueError(f"tar_v5 discovered no picked-image samples under: {picked17k_image_tar_url}")
        if not self.output_tensors:
            raise ValueError("tar_v5 currently requires output_tensors=True")

        (
            self.yubari_video_manifest_urls,
            self.yubari_video_urls,
            self.yubari_video_tar_urls,
            self.yubari_video_file_urls,
        ) = self._discover_video_source(yubari_video_tar_url)
        (
            self.takano_video_manifest_urls,
            self.takano_video_urls,
            self.takano_video_tar_urls,
            self.takano_video_file_urls,
        ) = self._discover_video_source(takano_video_tar_url)
        self._validate_video_source("yubari", yubari_video_tar_url, self.yubari_video_tar_urls, self.yubari_video_file_urls)
        self._validate_video_source("takano", takano_video_tar_url, self.takano_video_tar_urls, self.takano_video_file_urls)

        self.image_group_size = _projector_group_size_from_num_frames(num_frames)
        self.custom_collate_fn = collate_image_video_joint_v5

    @staticmethod
    def _resolve_video_probs(
        yubari_video_tar_url: Optional[str],
        takano_video_tar_url: Optional[str],
        yubari_video_prob: Optional[float],
        takano_video_prob: Optional[float],
    ) -> Tuple[float, float]:
        has_yubari = bool(yubari_video_tar_url)
        has_takano = bool(takano_video_tar_url)
        if has_yubari and has_takano:
            if yubari_video_prob is None and takano_video_prob is None:
                return 0.5, 0.5
            if yubari_video_prob is None or takano_video_prob is None:
                raise ValueError("When both video sources are enabled, yubari_video_prob and takano_video_prob must both be set.")
            total = float(yubari_video_prob) + float(takano_video_prob)
            if total <= 0:
                raise ValueError("yubari_video_prob + takano_video_prob must be > 0.")
            return float(yubari_video_prob) / total, float(takano_video_prob) / total
        if has_yubari:
            return 1.0, 0.0
        if has_takano:
            return 0.0, 1.0
        raise ValueError("No enabled video source found.")

    def _discover_video_source(
        self,
        base_url: Optional[str],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        if not base_url:
            return [], [], [], []
        manifest_urls, urls = self._discover_sample_sources(base_url, VIDEO_EXTENSIONS + (".tar",))
        tar_urls = [url for url in urls if str(url).endswith(".tar")]
        file_urls = [url for url in urls if not str(url).endswith(".tar")]
        return manifest_urls, urls, tar_urls, file_urls

    @staticmethod
    def _validate_video_source(
        source_name: str,
        base_url: Optional[str],
        tar_urls: Sequence[str],
        file_urls: Sequence[str],
    ) -> None:
        if not base_url:
            return
        if file_urls:
            raise ValueError(
                f"tar_v5 only supports tar-based {source_name} inputs, "
                f"but discovered non-tar video files: {list(file_urls)[:8]}"
            )
        if not tar_urls:
            raise ValueError(f"tar_v5 discovered no {source_name} tar shards under: {base_url}")

    def _process_image(self, image: Image.Image, sample_id: str, rng) -> Optional[Dict]:
        """
        V5 images stay as true single-frame samples here. Grouping happens in
        `__iter__` so the grouped sample can combine independent images.
        """
        try:
            image = image.convert("RGB")
            if not self._meets_min_resolution(*image.size):
                return None
            frame = self.frame_processor(image)
            sample_seed = self._next_sample_seed(rng)
            lq_frames = self._build_lq_clip([frame], rng=rng, sample_seed=sample_seed)
            return {
                "video": torch.stack([self._pil_to_tensor(frame)], dim=0),
                "lq_video": torch.stack([self._pil_to_tensor(lq_frames[0])], dim=0),
                "sample_seed": torch.tensor(sample_seed, dtype=torch.long),
                "sample_id": sample_id,
                "source_type": "image",
                "sample_kind": "single_image",
                "segment_lengths": [1],
                "image_group_size": 1,
            }
        except Exception as error:
            import warnings

            warnings.warn(f"Failed to process tar_v5 single-frame image sample: {error}")
            return None

    def _wrap_video_sample(self, sample: Dict) -> Dict:
        sequence_length = int(sample["video"].shape[0])
        wrapped = dict(sample)
        wrapped["segment_lengths"] = [sequence_length]
        wrapped["sample_kind"] = "video"
        wrapped["image_group_size"] = 0
        wrapped["target_num_frames"] = sequence_length
        return wrapped

    def _wrap_video_sample_from_source(self, sample: Dict, source_dataset: str) -> Dict:
        wrapped = self._wrap_video_sample(sample)
        wrapped["source_dataset"] = source_dataset
        return wrapped

    def _build_grouped_image_sample(self, samples: List[Dict]) -> Dict:
        if len(samples) != self.image_group_size:
            raise ValueError(
                f"Expected {self.image_group_size} images for grouped sample, got {len(samples)}"
            )
        sample_ids = [str(sample.get("sample_id")) for sample in samples]
        seeds = [sample["sample_seed"] for sample in samples]
        return {
            "video": torch.cat([sample["video"] for sample in samples], dim=0),
            "lq_video": torch.cat([sample["lq_video"] for sample in samples], dim=0),
            "sample_seed": torch.stack(seeds, dim=0),
            "sample_id": "|".join(sample_ids),
            "source_type": "image",
            "sample_kind": "image_group",
            "segment_lengths": [1] * self.image_group_size,
            "image_group_size": self.image_group_size,
            "target_num_frames": self.num_frames,
        }

    def _group_image_iterator(self, rng) -> Iterator[Dict]:
        image_iter = self._image_iterator(rng=rng)
        pending: List[Dict] = []
        while True:
            pending.append(next(image_iter))
            if len(pending) == self.image_group_size:
                yield self._build_grouped_image_sample(pending)
                pending = []

    def _iterate_video_source(
        self,
        source_dataset: str,
        tar_urls: Sequence[str],
        file_urls: Sequence[str],
        manifest_urls: Sequence[str],
        rng,
    ) -> Iterator[Dict]:
        iterators: List[Iterator[Dict]] = []
        if tar_urls:
            iterators.append(self._iterate_tar_videos_for_urls(list(tar_urls), rng=rng))
        if file_urls or manifest_urls:
            iterators.append(
                self._iterate_direct_videos_for_urls(
                    file_urls=list(file_urls),
                    manifest_urls=list(manifest_urls),
                    rng=rng,
                )
            )
        if not iterators:
            return
        if len(iterators) == 1:
            while True:
                yield self._wrap_video_sample_from_source(next(iterators[0]), source_dataset)
        while True:
            yield self._wrap_video_sample_from_source(next(iterators[rng.randrange(len(iterators))]), source_dataset)

    def _iterate_tar_videos_for_urls(self, tar_urls: Sequence[str], rng: random.Random) -> Iterator[Dict]:
        if not tar_urls:
            return
        datapipe = self._make_torchdata_tar_pipe(list(tar_urls), rng=rng if self.global_seed is not None else None)
        while True:
            for file_name, stream_item in datapipe:
                if not str(file_name).endswith(VIDEO_EXTENSIONS):
                    continue
                processed = self._process_video_bytes(
                    stream_item.read(),
                    sample_id=os.path.basename(str(file_name)),
                    rng=rng,
                )
                if processed is not None:
                    if not self.output_tensors:
                        processed["tar_member_path"] = str(file_name)
                    yield processed

    def _iterate_direct_videos_for_urls(
        self,
        file_urls: Sequence[str],
        manifest_urls: Sequence[str],
        rng: random.Random,
    ) -> Iterator[Dict]:
        urls = self._split_for_process_and_worker(list(file_urls))
        for url in self._iter_deterministic_permutation(urls, rng) if urls else []:
            sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
            if sample is not None:
                yield sample
        if manifest_urls:
            for url in self._iter_manifest_entries(list(manifest_urls)):
                sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
                if sample is not None:
                    yield sample

    def __iter__(self) -> Iterator[Dict]:
        rng = self._make_iteration_rng()
        has_video_source = bool(
            self.yubari_video_urls
            or self.yubari_video_manifest_urls
            or self.takano_video_urls
            or self.takano_video_manifest_urls
        )
        has_image_source = bool(self.image_urls or self.image_manifest_urls)
        video_iters: List[Tuple[float, Iterator[Dict]]] = []
        if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls):
            video_iters.append(
                (
                    self.yubari_video_prob,
                    self._iterate_video_source(
                        source_dataset="yubari",
                        tar_urls=self.yubari_video_tar_urls,
                        file_urls=self.yubari_video_file_urls,
                        manifest_urls=self.yubari_video_manifest_urls,
                        rng=rng,
                    ),
                )
            )
        if self.takano_video_prob > 0 and (self.takano_video_urls or self.takano_video_manifest_urls):
            video_iters.append(
                (
                    self.takano_video_prob,
                    self._iterate_video_source(
                        source_dataset="takano",
                        tar_urls=self.takano_video_tar_urls,
                        file_urls=self.takano_video_file_urls,
                        manifest_urls=self.takano_video_manifest_urls,
                        rng=rng,
                    ),
                )
            )
        if len(video_iters) == 1:
            video_iter = video_iters[0][1]
        elif video_iters:
            weights = [item[0] for item in video_iters]
            sources = [item[1] for item in video_iters]
            video_iter = None
        else:
            video_iter = None
        grouped_image_iter = self._group_image_iterator(rng=rng) if has_image_source else None

        while True:
            use_image = grouped_image_iter is not None and (
                (not video_iters) or rng.random() < self.image_dataset_prob
            )
            if use_image:
                yield next(grouped_image_iter)
            else:
                if video_iter is not None:
                    yield next(video_iter)
                else:
                    yield next(rng.choices(sources, weights=weights, k=1)[0])
