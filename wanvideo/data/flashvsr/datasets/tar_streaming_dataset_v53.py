import os
import random
import json
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from .streaming_dataset import IMAGE_EXTENSIONS, PseudoVideoGenerator, VIDEO_EXTENSIONS, FlashVSRStreamingDataset
from .parquet_index import normalize_remote_url


class FlashVSRTarStreamingDatasetV53(FlashVSRStreamingDataset):
    """
    Author-style paired dual-branch tar dataset.

    Each yielded training sample contains:
    - one real video branch
    - one image branch expanded into a pseudo-video by the base streaming path

    The training loop can then compute:
    - loss_video
    - loss_image
    - final_loss = loss_video + loss_image
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
        degradation_config_path: Optional[str] = None,
        degradation_seed: Optional[int] = None,
        hq_prefix_frames: int = 0,
        control_dropout_prob: float = 0.0,
        shuffle_buffer: int = 100,
        global_seed: Optional[int] = None,
        output_tensors: bool = True,
        image_branch_num_frames: Optional[int] = None,
    ):
        if not yubari_video_tar_url and not takano_video_tar_url:
            raise ValueError("tar_v53 requires at least one video source")
        self.yubari_video_prob, self.takano_video_prob = self._resolve_video_probs(
            yubari_video_tar_url=yubari_video_tar_url,
            takano_video_tar_url=takano_video_tar_url,
            yubari_video_prob=yubari_video_prob,
            takano_video_prob=takano_video_prob,
        )
        self.image_branch_num_frames = (
            max(1, int(image_branch_num_frames))
            if image_branch_num_frames is not None
            else max(1, ((int(num_frames) - 1) // 4) + 1)
        )
        super().__init__(
            internal_url=None,
            image_internal_url=image_tar_root_url,
            image_dataset_prob=0.5,
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
            degradation_config_path=degradation_config_path,
            degradation_seed=degradation_seed,
            hq_prefix_frames=hq_prefix_frames,
            control_dropout_prob=control_dropout_prob,
            shuffle_buffer=shuffle_buffer,
            global_seed=global_seed,
            output_tensors=output_tensors,
        )
        if self.parquet_records:
            raise RuntimeError("tar_v53 should not construct parquet records")
        self._expand_image_manifest_entries()
        if not self.image_tar_urls and not self.image_file_urls and not self.image_manifest_urls:
            raise ValueError(f"tar_v53 discovered no image samples under: {image_tar_root_url}")
        if not self.output_tensors:
            raise ValueError("tar_v53 currently requires output_tensors=True")
        self.image_pseudo_video_generator = PseudoVideoGenerator(
            target_height=height,
            target_width=width,
            num_frames=self.image_branch_num_frames,
        )

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
        self._validate_video_source(
            "yubari",
            yubari_video_tar_url,
            self.yubari_video_manifest_urls,
            self.yubari_video_tar_urls,
            self.yubari_video_file_urls,
        )
        self._validate_video_source(
            "takano",
            takano_video_tar_url,
            self.takano_video_manifest_urls,
            self.takano_video_tar_urls,
            self.takano_video_file_urls,
        )
        self.custom_collate_fn = self.paired_tensor_collate_fn

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
        if manifest_urls and not urls:
            expanded_urls: List[str] = []
            for manifest_path in manifest_urls:
                expanded_urls.extend(self._load_manifest_entries_once(manifest_path))
            urls = expanded_urls
            manifest_urls = []
        tar_urls = [url for url in urls if str(url).endswith(".tar")]
        file_urls = [url for url in urls if not str(url).endswith(".tar")]
        return manifest_urls, urls, tar_urls, file_urls

    def _expand_image_manifest_entries(self) -> None:
        if not self.image_manifest_urls:
            return
        remaining_manifest_urls: List[str] = []
        expanded_urls: List[str] = list(self.image_urls)
        for manifest_path in self.image_manifest_urls:
            entries = self._load_manifest_entries_once(manifest_path)
            if not entries:
                remaining_manifest_urls.append(manifest_path)
                continue
            expanded_urls.extend(entries)
        self.image_urls = expanded_urls
        self.image_tar_urls = [url for url in expanded_urls if str(url).endswith(".tar")]
        self.image_file_urls = [
            url
            for url in expanded_urls
            if str(url).endswith(IMAGE_EXTENSIONS)
        ]
        unknown_urls = [
            url
            for url in expanded_urls
            if not str(url).endswith(".tar") and not str(url).endswith(IMAGE_EXTENSIONS)
        ]
        if unknown_urls:
            raise ValueError(
                "tar_v53 image manifest contains unsupported entries: "
                f"{unknown_urls[:8]}"
            )
        self.image_manifest_urls = remaining_manifest_urls

    @staticmethod
    def _load_manifest_entries_once(manifest_path: str) -> List[str]:
        entries: List[str] = []
        is_jsonl = str(manifest_path).lower().endswith(".jsonl")
        with open(manifest_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if is_jsonl:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        value = payload.get("path") or payload.get("url")
                    else:
                        value = payload
                    if value:
                        entries.append(normalize_remote_url(str(value)))
                else:
                    entries.append(normalize_remote_url(line))
        return entries

    @staticmethod
    def _validate_video_source(
        source_name: str,
        base_url: Optional[str],
        manifest_urls: Sequence[str],
        tar_urls: Sequence[str],
        file_urls: Sequence[str],
    ) -> None:
        if not base_url:
            return
        if file_urls:
            raise ValueError(
                f"tar_v53 only supports tar-based {source_name} inputs, "
                f"but discovered non-tar video files: {list(file_urls)[:8]}"
            )
        if not tar_urls and not manifest_urls:
            raise ValueError(f"tar_v53 discovered no {source_name} tar shards under: {base_url}")

    @staticmethod
    def paired_tensor_collate_fn(batch):
        if not batch:
            raise ValueError("Empty batch")
        keys = (
            "video",
            "lq_video",
            "image_video",
            "image_lq_video",
        )
        output = {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}
        passthrough_keys = (
            "sample_id",
            "image_sample_id",
            "source_type",
            "image_source_type",
        )
        for key in passthrough_keys:
            output[key] = [sample.get(key) for sample in batch]
        return output

    def _process_image(self, image: Image.Image, sample_id: str, rng: random.Random) -> Optional[Dict[str, Any]]:
        try:
            image = image.convert("RGB")
            if not self._meets_min_resolution(*image.size):
                return None
            sample_seed = self._next_sample_seed(rng)
            pseudo_rng = random.Random(sample_seed)
            frames = self.image_pseudo_video_generator.generate(image, seed=sample_seed, rng=pseudo_rng)
            frames = [self.frame_processor(frame) for frame in frames]
            return self._maybe_convert_output({
                "video": frames,
                "lq_video": self._build_lq_clip(frames, rng=rng, sample_seed=sample_seed),
                "sample_id": sample_id,
                "source_type": "image",
                "sample_seed": sample_seed,
            })
        except Exception as error:
            import warnings
            warnings.warn(f"Failed to process paired image sample: {error}")
            return None

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
                sample = next(iterators[0])
                sample["source_dataset"] = source_dataset
                yield sample
        while True:
            sample = next(iterators[rng.randrange(len(iterators))])
            sample["source_dataset"] = source_dataset
            yield sample

    def _iterate_tar_videos_for_urls(self, tar_urls: Sequence[str], rng) -> Iterator[Dict]:
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
        rng,
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
        if not (
            self.yubari_video_urls
            or self.yubari_video_manifest_urls
            or self.takano_video_urls
            or self.takano_video_manifest_urls
        ):
            raise ValueError("tar_v53 requires at least one video source")
        if not (self.image_urls or self.image_manifest_urls):
            raise ValueError("tar_v53 requires at least one image source")
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
            weights = None
            sources = None
        else:
            weights = [item[0] for item in video_iters]
            sources = [item[1] for item in video_iters]
            video_iter = None

        def reset_video_iter() -> Iterator[Dict]:
            if len(video_iters) == 1:
                return self._iterate_video_source(
                    source_dataset="yubari" if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls) else "takano",
                    tar_urls=self.yubari_video_tar_urls if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls) else self.takano_video_tar_urls,
                    file_urls=self.yubari_video_file_urls if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls) else self.takano_video_file_urls,
                    manifest_urls=self.yubari_video_manifest_urls if self.yubari_video_prob > 0 and (self.yubari_video_urls or self.yubari_video_manifest_urls) else self.takano_video_manifest_urls,
                    rng=rng,
                )
            assert sources is not None and weights is not None
            return rng.choices(sources, weights=weights, k=1)[0]

        image_iter = self._image_iterator(rng=rng)
        while True:
            current_video_iter = video_iter if video_iter is not None else reset_video_iter()
            try:
                video_sample = next(current_video_iter)
            except StopIteration:
                current_video_iter = reset_video_iter()
                video_sample = next(current_video_iter)
            if video_iter is not None:
                video_iter = current_video_iter

            try:
                image_sample = next(image_iter)
            except StopIteration:
                image_iter = self._image_iterator(rng=rng)
                image_sample = next(image_iter)
            yield {
                "video": video_sample["video"],
                "lq_video": video_sample["lq_video"],
                "image_video": image_sample["video"],
                "image_lq_video": image_sample["lq_video"],
                "sample_id": video_sample.get("sample_id"),
                "image_sample_id": image_sample.get("sample_id"),
                "source_type": video_sample.get("source_type", "video"),
                "image_source_type": image_sample.get("source_type", "image"),
                "source_dataset": video_sample.get("source_dataset"),
            }

    def validation_video_iterator(self, rng: Optional[random.Random] = None) -> Iterator[Dict]:
        rng = rng or self._make_iteration_rng()
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
        if not video_iters:
            raise ValueError("tar_v53 validation requires at least one video source")
        if len(video_iters) == 1:
            while True:
                yield next(video_iters[0][1])
        weights = [item[0] for item in video_iters]
        sources = [item[1] for item in video_iters]
        while True:
            yield next(rng.choices(sources, weights=weights, k=1)[0])
