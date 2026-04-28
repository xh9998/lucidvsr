import json
import math
import os
import random
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torch.distributed as dist

from diffsynth.core.data.operators import ImageCropAndResize
from wanvideo.data.flashvsr.degradation import build_degradation_model
from .media_reader_v2 import FlashVSRMediaReaderV2
from .parquet_index import (
    FlashVSRParquetRecord,
    _discover_parquet_urls,
    _build_takano_record,
    _build_image_record,
    _read_parquet_frame,
    iter_parquet_row_dicts_from_url,
)
from .source_index_v2 import _discover_yubari_video_root_urls, build_source_index_v2

YUBARI_DEFAULT_SHARD_START = 0
YUBARI_DEFAULT_SHARD_END = 6714


class ConsistentClipDegradation:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model = None
        self.model_pid = None

    def _get_model(self):
        current_pid = os.getpid()
        if self.model is None or self.model_pid != current_pid:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
            else:
                device = "cpu"
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


class PseudoVideoGenerator:
    def __init__(self, target_height: int, target_width: int, num_frames: int):
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

    def _ensure_large_enough(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        if img_w >= self.target_width * 2 and img_h >= self.target_height * 2:
            return image
        scale = max(2.0 * self.target_width / max(img_w, 1), 2.0 * self.target_height / max(img_h, 1))
        return image.resize((int(round(img_w * scale)), int(round(img_h * scale))), Image.LANCZOS)

    def _pan(self, image: Image.Image, rng: random.Random) -> List[Image.Image]:
        image = self._ensure_large_enough(image)
        img_w, img_h = image.size
        max_x = max(0, img_w - self.target_width)
        max_y = max(0, img_h - self.target_height)
        start_x = rng.randint(0, max_x) if max_x > 0 else 0
        start_y = rng.randint(0, max_y) if max_y > 0 else 0
        end_x = rng.randint(0, max_x) if max_x > 0 else 0
        end_y = rng.randint(0, max_y) if max_y > 0 else 0
        frames: List[Image.Image] = []
        for idx in range(self.num_frames):
            alpha = idx / max(1, self.num_frames - 1)
            cur_x = int(round((1 - alpha) * start_x + alpha * end_x))
            cur_y = int(round((1 - alpha) * start_y + alpha * end_y))
            frames.append(image.crop((cur_x, cur_y, cur_x + self.target_width, cur_y + self.target_height)))
        return frames

    def _zoom(self, image: Image.Image, rng: random.Random) -> List[Image.Image]:
        image = self._ensure_large_enough(image)
        img_w, img_h = image.size
        max_zoom = min(img_w / self.target_width, img_h / self.target_height, 2.5)
        zoom = rng.uniform(1.1, max_zoom)
        center_x = img_w // 2
        center_y = img_h // 2
        frames: List[Image.Image] = []
        for idx in range(self.num_frames):
            alpha = idx / max(1, self.num_frames - 1)
            current_zoom = zoom - (zoom - 1.0) * alpha
            crop_w = int(round(self.target_width * current_zoom))
            crop_h = int(round(self.target_height * current_zoom))
            left = max(0, min(img_w - crop_w, center_x - crop_w // 2))
            top = max(0, min(img_h - crop_h, center_y - crop_h // 2))
            frame = image.crop((left, top, left + crop_w, top + crop_h))
            frames.append(frame.resize((self.target_width, self.target_height), Image.LANCZOS))
        if rng.random() < 0.5:
            frames.reverse()
        return frames

    def generate(self, image: Image.Image, seed: Optional[int] = None, rng: Optional[random.Random] = None) -> List[Image.Image]:
        if rng is None:
            rng = random.Random(seed) if seed is not None else random.Random()
        return self._pan(image, rng) if rng.random() < 0.5 else self._zoom(image, rng)


class FlashVSRParquetTarDatasetV2(IterableDataset):
    """
    A standalone parquet+tar dataset for future Takano-style metadata roots.

    Design goals:
    - metadata_url points to parquet shards
    - parquet rows resolve to tar shard path + tar member path
    - deterministic seed control across rank/worker
    - no dependency on current streaming_dataset iterator mix logic
    - optional image input can be treated as single-frame video (paper-aligned)
    """

    load_from_cache = False

    def __init__(
        self,
        metadata_url: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        stride: int = 1,
        max_source_frames: int = 160,
        metadata_source: str = "takano",
        image_metadata_url: Optional[str] = None,
        image_internal_url: Optional[str] = None,
        image_dataset_prob: float = 0.0,
        takano_dataset_prob: Optional[float] = None,
        yubari_dataset_prob: Optional[float] = None,
        image_as_single_frame: bool = True,
        yubari_video_metadata_url: Optional[str] = None,
        yubari_sidecar_metadata_url: Optional[str] = None,
        yubari_video_tar_url: Optional[str] = None,
        yubari_sidecar_tar_url: Optional[str] = None,
        yubari_shard_start: Optional[int] = None,
        yubari_shard_end: Optional[int] = None,
        enable_degradation: bool = False,
        degradation_config_path: Optional[str] = None,
        global_seed: Optional[int] = None,
        output_tensors: bool = True,
        max_parquet_records: Optional[int] = None,
        max_yubari_records: Optional[int] = None,
        media_cache_dir: Optional[str] = None,
        parquet_prewarm_files_per_source: int = 8,
    ):
        super().__init__()
        self.metadata_url = metadata_url
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.stride = stride
        self.max_source_frames = max_source_frames
        self.metadata_source = metadata_source
        self.image_metadata_url = image_metadata_url
        self.image_internal_url = image_internal_url
        self.image_dataset_prob = image_dataset_prob
        self.takano_dataset_prob = takano_dataset_prob
        self.yubari_dataset_prob = yubari_dataset_prob
        self.image_as_single_frame = image_as_single_frame
        self.yubari_video_metadata_url = yubari_video_metadata_url
        self.yubari_sidecar_metadata_url = yubari_sidecar_metadata_url
        self.yubari_video_tar_url = yubari_video_tar_url
        self.yubari_sidecar_tar_url = yubari_sidecar_tar_url
        if (
            yubari_video_tar_url
            and not yubari_sidecar_tar_url
            and yubari_shard_start is None
            and yubari_shard_end is None
        ):
            yubari_shard_start = YUBARI_DEFAULT_SHARD_START
            yubari_shard_end = YUBARI_DEFAULT_SHARD_END
        self.yubari_shard_start = yubari_shard_start
        self.yubari_shard_end = yubari_shard_end
        self.enable_degradation = enable_degradation
        self.degradation_config_path = degradation_config_path
        self.global_seed = global_seed
        self.output_tensors = output_tensors
        self.max_yubari_records = max_yubari_records
        self.parquet_prewarm_files_per_source = max(0, int(parquet_prewarm_files_per_source))

        self.frame_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=height * width,
            height_division_factor=16,
            width_division_factor=16,
        )
        self.pseudo_video_generator = PseudoVideoGenerator(height, width, num_frames)
        self.degradation_model = ConsistentClipDegradation(config_path=degradation_config_path) if enable_degradation else None
        self.media_reader = FlashVSRMediaReaderV2(
            frame_processor=self.frame_processor,
            num_frames=num_frames,
            stride=stride,
            max_source_frames=max_source_frames,
            media_cache_dir=media_cache_dir,
        )
        self._maybe_prewarm_parquet_cache()
        self.source_index = build_source_index_v2(
            media_reader=self.media_reader,
            metadata_url=metadata_url,
            metadata_source=metadata_source,
            image_metadata_url=image_metadata_url,
            image_internal_url=image_internal_url,
            yubari_video_metadata_url=yubari_video_metadata_url,
            yubari_sidecar_metadata_url=yubari_sidecar_metadata_url,
            yubari_video_tar_url=yubari_video_tar_url,
            yubari_sidecar_tar_url=yubari_sidecar_tar_url,
            yubari_shard_start=self.yubari_shard_start,
            yubari_shard_end=self.yubari_shard_end,
            max_parquet_records=max_parquet_records,
            max_yubari_records=max_yubari_records,
        )
        self.records = self.source_index.records
        self.image_urls = self.source_index.image_urls
        self.takano_parquet_urls = self.source_index.takano_parquet_urls
        self.image_parquet_urls = self.source_index.image_parquet_urls
        self.yubari_video_root_pairs = self.source_index.yubari_video_root_pairs
        self.takano_records = [record for record in self.records if record.dataset_source == "takano"]
        self.yubari_records = [record for record in self.records if record.dataset_source == "yubari"]
        self.image_records = [record for record in self.records if record.dataset_source == "image"]
        if not self.records and not self.image_urls and not self.takano_parquet_urls and not self.image_parquet_urls and not self.yubari_video_root_pairs:
            raise ValueError(
                f"No records discovered. metadata_url={metadata_url}, yubari_video_metadata_url={yubari_video_metadata_url}, "
                f"yubari_video_tar_url={yubari_video_tar_url}, image_internal_url={image_internal_url}"
            )
        self.source_sampling_probs = self._resolve_source_sampling_probs()

    def _maybe_prewarm_parquet_cache(self) -> None:
        cache_dir = os.environ.get("FLASHVSR_PARQUET_CACHE_DIR")
        if not cache_dir or self.parquet_prewarm_files_per_source <= 0:
            return

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        urls_to_warm: List[str] = []

        if local_rank == 0:
            urls_to_warm.extend(
                _discover_parquet_urls(self.metadata_url)[: self.parquet_prewarm_files_per_source]
            )
            urls_to_warm.extend(
                _discover_parquet_urls(self.image_metadata_url)[: self.parquet_prewarm_files_per_source]
            )
            if self.yubari_video_tar_url:
                urls_to_warm.extend(
                    _discover_yubari_video_root_urls(
                        self.media_reader,
                        self.yubari_video_tar_url,
                        (".parquet",),
                        shard_start=self.yubari_shard_start,
                        shard_end=self.yubari_shard_end,
                        max_files=self.parquet_prewarm_files_per_source,
                    )
                )

            unique_urls = list(dict.fromkeys(urls_to_warm))
            print(
                f"[parquet_prewarm rank={rank} local_rank={local_rank}] "
                f"warming {len(unique_urls)} parquet shards into {cache_dir}",
                flush=True,
            )
            for parquet_url in unique_urls:
                _read_parquet_frame(parquet_url)
            print(
                f"[parquet_prewarm rank={rank} local_rank={local_rank}] done",
                flush=True,
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _make_iteration_rng(self) -> random.Random:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rank = 0
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        base_seed = int(self.global_seed) if self.global_seed is not None else 20260413
        return random.Random(base_seed + rank * 1000003 + worker_id * 1000033)

    def _split_for_process_and_worker(self, items: Sequence[Any]) -> List[Any]:
        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        sharded = list(items)[rank::world_size]
        worker = get_worker_info()
        if worker is None:
            return sharded
        return sharded[worker.id :: worker.num_workers]

    @staticmethod
    def _coprime_step(length: int, rng: random.Random) -> int:
        if length <= 1:
            return 1
        step = rng.randrange(1, length)
        while math.gcd(step, length) != 1:
            step = rng.randrange(1, length)
        return step

    def _iter_deterministic_permutation(self, items: Sequence[Any], rng: random.Random) -> Iterator[Any]:
        items = list(items)
        length = len(items)
        if length == 0:
            return
        while True:
            if length == 1:
                yield items[0]
                continue
            start = rng.randrange(length)
            step = self._coprime_step(length, rng)
            for offset in range(length):
                yield items[(start + offset * step) % length]

    def _iter_yubari_locality_permutation(self, records: Sequence[FlashVSRParquetRecord], rng: random.Random) -> Iterator[FlashVSRParquetRecord]:
        grouped: "OrderedDict[str, List[FlashVSRParquetRecord]]" = OrderedDict()
        for record in records:
            grouped.setdefault(record.media_path, []).append(record)
        shard_paths = list(grouped.keys())
        while True:
            for shard_path in self._iter_deterministic_permutation(shard_paths, rng):
                shard_records = grouped[shard_path]
                if len(shard_records) == 1:
                    yield shard_records[0]
                    continue
                start = rng.randrange(len(shard_records))
                step = self._coprime_step(len(shard_records), rng)
                for offset in range(len(shard_records)):
                    yield shard_records[(start + offset * step) % len(shard_records)]

    def _split_grouped_records_for_process_and_worker(self, records: Sequence[FlashVSRParquetRecord]) -> List[FlashVSRParquetRecord]:
        grouped: "OrderedDict[str, List[FlashVSRParquetRecord]]" = OrderedDict()
        for record in records:
            grouped.setdefault(record.media_path, []).append(record)
        groups = list(grouped.values())

        rank = 0
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        sharded_groups = groups[rank::world_size]

        worker = get_worker_info()
        if worker is not None:
            sharded_groups = sharded_groups[worker.id :: worker.num_workers]

        flattened: List[FlashVSRParquetRecord] = []
        for group in sharded_groups:
            flattened.extend(group)
        return flattened

    @staticmethod
    def _should_use_yubari_locality(records: Sequence[FlashVSRParquetRecord]) -> bool:
        return bool(records) and all(
            record.dataset_source == "yubari" and record.metadata.get("yubari_video_root_index")
            for record in records
        )

    def _next_sample_seed(self, rng: random.Random) -> int:
        return rng.randint(0, 2**31 - 1)

    def _pil_to_tensor(self, frame: Image.Image) -> torch.Tensor:
        array = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.bfloat16)

    def _build_lq_clip(self, hr_frames: List[Image.Image], sample_seed: int) -> List[Image.Image]:
        if self.degradation_model is None:
            return [frame.copy() for frame in hr_frames]
        return self.degradation_model.degrade_batch_consistent(hr_frames, seed=sample_seed)

    def _process_image_bytes(self, image_bytes: bytes, sample_id: str, rng: random.Random, source_dataset: str = "image") -> Optional[Dict[str, Any]]:
        image = self.media_reader.decode_image_bytes(image_bytes)
        if image is None:
            warnings.warn(f"Failed to decode image sample {sample_id}")
            return None
        sample_seed = self._next_sample_seed(rng)
        if self.image_as_single_frame:
            frames = [self.frame_processor(image)]
        else:
            pseudo_rng = random.Random(sample_seed)
            frames = self.pseudo_video_generator.generate(image=image, seed=sample_seed, rng=pseudo_rng)
            frames = [self.frame_processor(frame) for frame in frames]
        sample = {
            "video": frames,
            "lq_video": self._build_lq_clip(frames, sample_seed=sample_seed),
            "sample_seed": sample_seed,
            "sample_id": sample_id,
            "media_path": sample_id,
            "tar_member_path": None,
            "source_dataset": source_dataset,
            "caption_text": None,
            "metadata": {},
            "source_type": "image",
        }
        return self._convert_output(sample)

    def _convert_output(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self.output_tensors:
            return sample
        return {
            "video": torch.stack([self._pil_to_tensor(frame) for frame in sample["video"]], dim=0),
            "lq_video": torch.stack([self._pil_to_tensor(frame) for frame in sample["lq_video"]], dim=0),
            "sample_seed": torch.tensor(sample["sample_seed"], dtype=torch.long),
            "sample_id": sample["sample_id"],
            "media_path": sample["media_path"],
            "tar_member_path": sample.get("tar_member_path"),
            "source_dataset": sample["source_dataset"],
        }

    def _process_record(self, record: FlashVSRParquetRecord, rng: random.Random) -> Optional[Dict[str, Any]]:
        if record.dataset_source == "image" or str(record.media_path).lower().endswith((".jpg", ".jpeg", ".png")):
            image_bytes = self.media_reader.read_media_bytes(record.media_path)
            if image_bytes is None:
                return None
            sample = self._process_image_bytes(
                image_bytes=image_bytes,
                sample_id=record.sample_id,
                rng=rng,
                source_dataset=record.dataset_source,
            )
            if sample is not None and record.caption_text is not None:
                sample["caption_text"] = record.caption_text
            return sample
        resolved_member_path = record.tar_member_path
        resolved_sample_id = record.sample_id
        if record.dataset_source == "yubari" and record.metadata.get("yubari_tar_shard_only"):
            member_index = rng.randint(0, 15)
            extracted_member = self.media_reader.extract_indexed_tar_member_bytes(
                record.media_path,
                suffixes=(".mp4",),
                member_index=member_index,
            )
            if extracted_member is None:
                return None
            resolved_member_path, video_bytes = extracted_member
            resolved_sample_id = os.path.basename(resolved_member_path)
        elif record.dataset_source == "yubari" and record.metadata.get("yubari_video_root_index"):
            data_offset = record.metadata.get("data_offset")
            data_size = record.metadata.get("data_size")
            if data_offset is not None and data_size is not None:
                video_bytes = self.media_reader.read_media_range(record.media_path, data_offset, data_size)
            else:
                video_bytes = self.media_reader.extract_tar_member_bytes(record.media_path, record.tar_member_path)
        elif record.tar_member_path is None:
            video_bytes = self.media_reader.read_media_bytes(record.media_path)
        else:
            video_bytes = self.media_reader.extract_tar_member_bytes(record.media_path, record.tar_member_path)
        if video_bytes is None:
            return None
        frames = self.media_reader.extract_frames(video_bytes)
        if frames is None:
            return None
        clip = self.media_reader.select_clip(frames, rng)
        if clip is None:
            return None
        sample_seed = self._next_sample_seed(rng)
        sample = {
            "video": clip,
            "lq_video": self._build_lq_clip(clip, sample_seed=sample_seed),
            "sample_seed": sample_seed,
            "sample_id": resolved_sample_id,
            "media_path": record.media_path,
            "tar_member_path": resolved_member_path,
            "source_dataset": record.dataset_source,
            "caption_text": record.caption_text,
            "metadata": {**record.metadata, "resolved_video_file_name": resolved_member_path},
            "source_type": "video",
        }
        if record.dataset_source == "yubari":
            metadata_tar_url = record.metadata.get("metadata_tar_url")
            metadata_file_name = record.metadata.get("metadata_file_name")
            if metadata_tar_url and metadata_file_name:
                metadata_bytes = self.media_reader.extract_tar_member_bytes(metadata_tar_url, metadata_file_name)
                if metadata_bytes is not None:
                    try:
                        sidecar = json.loads(metadata_bytes.decode("utf-8"))
                        sample["metadata"] = {**record.metadata, "sidecar": sidecar}
                        sample["caption_text"] = (
                            sidecar.get("caption")
                            or sidecar.get("title")
                            or record.caption_text
                        )
                    except Exception as error:
                        warnings.warn(f"Failed to parse Yubari metadata {metadata_file_name}: {error}")
        return self._convert_output(sample)

    def _iter_images(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        urls = self._split_for_process_and_worker(self.image_urls)
        if not urls:
            return
        for url in self._iter_deterministic_permutation(urls, rng):
            image_bytes = self.media_reader.read_media_bytes(url)
            if image_bytes is None:
                continue
            sample = self._process_image_bytes(
                image_bytes=image_bytes,
                sample_id=os.path.basename(url),
                rng=rng,
                source_dataset="image",
            )
            if sample is not None:
                yield sample

    def _iter_image_records(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        records = self._split_for_process_and_worker(self.image_records)
        if not records:
            return
        for record in self._iter_deterministic_permutation(records, rng):
            sample = self._process_record(record, rng)
            if sample is not None:
                yield sample

    def _iter_image_records_lazy(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        parquet_urls = self._split_for_process_and_worker(self.image_parquet_urls)
        if not parquet_urls:
            return
        for parquet_url in self._iter_deterministic_permutation(parquet_urls, rng):
            for row in iter_parquet_row_dicts_from_url(
                parquet_url,
                columns=[
                    "TARGET_S3_PATH",
                    "ASSET_NAME",
                    "DESCRIPTION",
                    "JAPANESE_DESCRIPTION",
                    "KOREAN_DESCRIPTION",
                    "FRENCH_DESCRIPTION",
                    "SPANISH_DESCRIPTION",
                    "GERMAN_DESCRIPTION",
                    "ASSET_ID",
                    "CATALOG",
                    "BATCH_SUFFIX",
                    "CATEGORIES",
                    "SUBCATEGORIES",
                    "KEYWORDS",
                    "LOCATION",
                    "TYPE",
                    "MD5",
                    "MAX_WIDTH",
                    "MAX_HEIGHT",
                    "SIZE",
                    "HAS_PEOPLE",
                    "FULL_BIOMETRIC_CONSENT",
                    "IS_ADULT",
                ],
            ):
                record = _build_image_record(row)
                if record is None:
                    continue
                sample = self._process_record(record, rng)
                if sample is not None:
                    yield sample

    def _iter_takano_records_lazy(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        parquet_urls = self._split_for_process_and_worker(self.takano_parquet_urls)
        if not parquet_urls:
            return
        for parquet_url in self._iter_deterministic_permutation(parquet_urls, rng):
            for row in iter_parquet_row_dicts_from_url(
                parquet_url,
                columns=[
                    "path_lucid",
                    "path",
                    "qwen35_output",
                    "MAX_WIDTH",
                    "MAX_HEIGHT",
                    "video_path",
                    "org_path",
                    "metadata_path",
                    "HIGH_RES_TARGET_S3_PATH",
                    "qwen35_parse_success",
                    "resolution",
                    "frame_rate",
                    "duration",
                    "width",
                    "height",
                    "source",
                ],
            ):
                record = _build_takano_record(row)
                if record is None:
                    continue
                sample = self._process_record(record, rng)
                if sample is not None:
                    yield sample

    def _iter_yubari_records_lazy(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        pairs = self._split_for_process_and_worker(self.yubari_video_root_pairs)
        if not pairs:
            return
        for pair in self._iter_deterministic_permutation(pairs, rng):
            parquet_url = pair["parquet_url"]
            video_tar_url = pair["video_tar_url"]
            for row in iter_parquet_row_dicts_from_url(
                parquet_url,
                columns=["file_name", "data_offset", "data_size"],
            ):
                file_name = row.get("file_name")
                if not str(file_name).endswith(".mp4"):
                    continue
                sample_id = os.path.basename(str(file_name))
                sample_key = os.path.splitext(sample_id)[0]
                data_offset = row.get("data_offset")
                data_size = row.get("data_size")
                record = FlashVSRParquetRecord(
                    dataset_source="yubari",
                    sample_id=sample_id,
                    media_path=video_tar_url,
                    tar_member_path=str(file_name),
                    caption_text=None,
                    metadata={
                        "sample_key": sample_key,
                        "video_tar_url": video_tar_url,
                        "video_file_name": str(file_name),
                        "yubari_video_root_index": True,
                        "data_offset": int(data_offset) if data_offset is not None else None,
                        "data_size": int(data_size) if data_size is not None else None,
                    },
                )
                sample = self._process_record(record, rng)
                if sample is not None:
                    yield sample

    def _iter_records_by_source(self, source_name: str, rng: random.Random) -> Optional[Iterator[Dict[str, Any]]]:
        if source_name == "takano":
            if self.takano_parquet_urls:
                return self._iter_takano_records_lazy(rng)
            records = self.takano_records
        elif source_name == "yubari":
            if self.yubari_video_root_pairs:
                return self._iter_yubari_records_lazy(rng)
            records = self.yubari_records
        elif source_name == "image":
            if self.image_parquet_urls:
                return self._iter_image_records_lazy(rng)
            if self.image_records:
                return self._iter_image_records(rng)
            if self.image_urls:
                return self._iter_images(rng)
            return None
        else:
            return None

        if not records:
            return None
        if source_name == "yubari" and self._should_use_yubari_locality(records):
            ordered_records = self._iter_yubari_locality_permutation(
                self._split_grouped_records_for_process_and_worker(records), rng
            )
        else:
            ordered_records = self._iter_deterministic_permutation(self._split_for_process_and_worker(records), rng)
        return (self._process_record(record, rng) for record in ordered_records)

    def _resolve_source_sampling_probs(self) -> Dict[str, float]:
        available_sources: Dict[str, bool] = {
            "takano": bool(self.takano_records or self.takano_parquet_urls),
            "yubari": bool(self.yubari_records or self.yubari_video_root_pairs),
            "image": bool(self.image_records or self.image_urls or self.image_parquet_urls),
        }
        raw_probs: Dict[str, Optional[float]] = {
            "takano": self.takano_dataset_prob,
            "yubari": self.yubari_dataset_prob,
            "image": self.image_dataset_prob,
        }
        explicit = {key: value for key, value in raw_probs.items() if value is not None and value > 0}
        if explicit:
            total = sum(value for key, value in explicit.items() if available_sources.get(key, False))
            if total <= 0:
                raise ValueError("Source probabilities were provided but none match available sources")
            return {
                key: ((raw_probs[key] or 0.0) / total) if available_sources.get(key, False) and (raw_probs[key] or 0.0) > 0 else 0.0
                for key in ("takano", "yubari", "image")
            }

        available_keys = [key for key, present in available_sources.items() if present]
        if not available_keys:
            raise ValueError("No available sources for sampling")
        default_prob = 1.0 / len(available_keys)
        return {key: (default_prob if key in available_keys else 0.0) for key in ("takano", "yubari", "image")}

    def _sample_source_name(self, rng: random.Random) -> str:
        ticket = rng.random()
        cumulative = 0.0
        last_key = "takano"
        for key in ("takano", "yubari", "image"):
            prob = self.source_sampling_probs.get(key, 0.0)
            if prob <= 0:
                continue
            cumulative += prob
            last_key = key
            if ticket <= cumulative:
                return key
        return last_key

    @staticmethod
    def tensor_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch")
        output: Dict[str, Any] = {}
        tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
        for key in tensor_keys:
            output[key] = torch.stack([sample[key] for sample in batch], dim=0)
        passthrough_keys = [key for key in batch[0].keys() if key not in tensor_keys]
        for key in passthrough_keys:
            output[key] = [sample[key] for sample in batch]
        return output

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = self._make_iteration_rng()
        source_iters: Dict[str, Optional[Iterator[Dict[str, Any]]]] = {
            "takano": self._iter_records_by_source("takano", rng),
            "yubari": self._iter_records_by_source("yubari", rng),
            "image": self._iter_records_by_source("image", rng),
        }

        while True:
            source_name = self._sample_source_name(rng)
            source_iter = source_iters.get(source_name)
            if source_iter is None:
                available = [key for key, iterator in source_iters.items() if iterator is not None]
                if not available:
                    raise RuntimeError("All dataset source iterators are unavailable")
                source_name = available[0]
                source_iter = source_iters[source_name]
            sample = next(source_iter)
            if sample is not None:
                yield sample


def create_parquet_tar_dataloader_v2(
    metadata_url: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    batch_size: int,
    stride: int = 1,
    max_source_frames: int = 160,
    metadata_source: str = "takano",
    image_metadata_url: Optional[str] = None,
    image_internal_url: Optional[str] = None,
    image_dataset_prob: float = 0.0,
    takano_dataset_prob: Optional[float] = None,
    yubari_dataset_prob: Optional[float] = None,
    image_as_single_frame: bool = True,
    yubari_video_metadata_url: Optional[str] = None,
    yubari_sidecar_metadata_url: Optional[str] = None,
    yubari_video_tar_url: Optional[str] = None,
    yubari_sidecar_tar_url: Optional[str] = None,
    yubari_shard_start: Optional[int] = None,
    yubari_shard_end: Optional[int] = None,
    enable_degradation: bool = False,
    degradation_config_path: Optional[str] = None,
    global_seed: Optional[int] = None,
    num_workers: int = 0,
    output_tensors: bool = True,
    max_parquet_records: Optional[int] = None,
    max_yubari_records: Optional[int] = None,
    media_cache_dir: Optional[str] = None,
) -> Tuple[FlashVSRParquetTarDatasetV2, DataLoader]:
    dataset = FlashVSRParquetTarDatasetV2(
        metadata_url=metadata_url,
        height=height,
        width=width,
        num_frames=num_frames,
        stride=stride,
        max_source_frames=max_source_frames,
        metadata_source=metadata_source,
        image_metadata_url=image_metadata_url,
        image_internal_url=image_internal_url,
        image_dataset_prob=image_dataset_prob,
        takano_dataset_prob=takano_dataset_prob,
        yubari_dataset_prob=yubari_dataset_prob,
        image_as_single_frame=image_as_single_frame,
        yubari_video_metadata_url=yubari_video_metadata_url,
        yubari_sidecar_metadata_url=yubari_sidecar_metadata_url,
        yubari_video_tar_url=yubari_video_tar_url,
        yubari_sidecar_tar_url=yubari_sidecar_tar_url,
        yubari_shard_start=yubari_shard_start,
        yubari_shard_end=yubari_shard_end,
        enable_degradation=enable_degradation,
        degradation_config_path=degradation_config_path,
        global_seed=global_seed,
        output_tensors=output_tensors,
        max_parquet_records=max_parquet_records,
        max_yubari_records=max_yubari_records,
        media_cache_dir=media_cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=FlashVSRParquetTarDatasetV2.tensor_collate_fn,
    )
    return dataset, dataloader
