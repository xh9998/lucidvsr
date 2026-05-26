import io
import hashlib
import json
import math
import os
import random
import tarfile
import tempfile
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None

try:
    import webdataset as wds
except ImportError:
    wds = None

try:
    from torchdata.datapipes.iter import IterableWrapper
except ImportError:
    try:
        from torch.utils.data.datapipes.iter import IterableWrapper
    except ImportError:
        IterableWrapper = None

from diffsynth.core.data.operators import ImageCropAndResize
from wanvideo.data.flashvsr.degradation import build_degradation_model
from .conductor_bridge_v2 import list_remote_files_with_suffixes
from .parquet_index import FlashVSRParquetRecord, load_parquet_records, normalize_remote_url

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
REMOTE_DISCOVERY_PREFIXES = ("conductor://", "s3://", "blobby://")
REMOTE_DISCOVERY_CACHE_DIR = os.environ.get("FLASHVSR_REMOTE_DISCOVERY_CACHE_DIR", "/tmp/flashvsr_remote_discovery")
REMOTE_DISCOVERY_STALE_LOCK_SECONDS = int(os.environ.get("FLASHVSR_REMOTE_DISCOVERY_STALE_LOCK_SECONDS", "600"))


def _dataset_debug_log(message: str):
    debug_dir = os.environ.get("FLASHVSR_DEBUG_DIR")
    if not debug_dir:
        return
    rank = 0
    local_rank = os.environ.get("LOCAL_RANK", "0")
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, f"dataset_branches_rank{rank}.log")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"rank={rank} local_rank={local_rank} {message}\n")


def _train_debug_print(message: str):
    if not _TRAIN_DEBUG_ENABLED:
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[dataset_debug {timestamp} rank={rank} local_rank={local_rank}] {message}", flush=True)


_CONVERT_DEBUG_COUNT = 0
_COLLATE_DEBUG_COUNT = 0
_TRAIN_DEBUG_ENABLED = os.environ.get("FLASHVSR_TRAIN_DEBUG", "").lower() in ("1", "true", "yes", "y")
_DATA_TIMING_DEBUG_ENABLED = os.environ.get("FLASHVSR_DATA_TIMING_DEBUG", "").lower() in ("1", "true", "yes", "y")
_DATA_TIMING_DEBUG_COUNT = 0
_DATA_TIMING_DEBUG_MAX = int(os.environ.get("FLASHVSR_DATA_TIMING_DEBUG_MAX", "12"))


def _data_timing_log(message: str):
    if not _DATA_TIMING_DEBUG_ENABLED:
        return
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    worker = get_worker_info()
    worker_id = "main" if worker is None else str(worker.id)
    print(f"[flashvsr_data_timing rank={rank} local_rank={local_rank} worker={worker_id}] {message}", flush=True)


def _distributed_rank_world_size() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _expand_input_urls(base_url: Optional[str]) -> List[str]:
    if base_url is None:
        return []
    candidates = [item.strip() for item in str(base_url).replace("\n", ",").split(",")]
    return [candidate for candidate in candidates if candidate]


def _looks_like_manifest(path: str) -> bool:
    lowered = str(path).lower()
    return lowered.endswith(".txt") or lowered.endswith(".jsonl") or lowered.endswith(".manifest")


def _load_manifest_entries(path: str) -> List[str]:
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
                    media_url = payload.get("media_url")
                    if media_url:
                        entries.append(normalize_remote_url(str(media_url)))
                elif isinstance(payload, str):
                    entries.append(normalize_remote_url(payload))
            else:
                entries.append(normalize_remote_url(line))
    return entries


def _discovery_cache_paths(base_url: str, suffixes: Sequence[str]) -> Tuple[str, str]:
    os.makedirs(REMOTE_DISCOVERY_CACHE_DIR, exist_ok=True)
    payload = json.dumps(
        {
            "base_url": normalize_remote_url(base_url),
            "suffixes": list(suffixes),
        },
        sort_keys=True,
    )
    cache_key = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    cache_path = os.path.join(REMOTE_DISCOVERY_CACHE_DIR, f"{cache_key}.json")
    lock_path = os.path.join(REMOTE_DISCOVERY_CACHE_DIR, f"{cache_key}.lock")
    return cache_path, lock_path


def _read_discovery_cache(cache_path: str) -> Optional[List[str]]:
    if not os.path.isfile(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    urls = payload.get("urls")
    if not isinstance(urls, list):
        return None
    normalized_urls = [normalize_remote_url(str(url)) for url in urls]
    # Empty discovery results are not a valid cache for remote datasets. A
    # transient conductor/env failure can otherwise poison future distributed
    # launches and make every rank immediately see an empty source list.
    if not normalized_urls:
        return None
    return normalized_urls


def _write_discovery_cache(cache_path: str, urls: Sequence[str]) -> None:
    if not urls:
        return
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump({"urls": list(urls)}, file, ensure_ascii=False)
    os.replace(tmp_path, cache_path)


def _try_remove_stale_discovery_lock(lock_path: str) -> bool:
    try:
        age = time.time() - os.path.getmtime(lock_path)
    except FileNotFoundError:
        return True
    except Exception:
        return False
    if age < REMOTE_DISCOVERY_STALE_LOCK_SECONDS:
        return False
    try:
        os.remove(lock_path)
        if _TRAIN_DEBUG_ENABLED:
            print(
                f"[dataset_discovery] removed stale lock path={lock_path} age_seconds={age:.1f}",
                flush=True,
            )
        return True
    except FileNotFoundError:
        return True
    except Exception as error:
        if _TRAIN_DEBUG_ENABLED:
            print(
                f"[dataset_discovery] failed to remove stale lock path={lock_path} error={type(error).__name__}:{error}",
                flush=True,
            )
        return False


def _node_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _degradation_cuda_device() -> str:
    # Keep online degradation outside CUDA. Degradation runs in DataLoader
    # workers; using CUDA here creates extra contexts and large temporary
    # tensors that compete with model training memory.
    return "cpu"


class ConsistentClipDegradation:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model = None
        self.model_pid = None

    def _get_model(self):
        current_pid = os.getpid()
        if self.model is None or self.model_pid != current_pid:
            device = _degradation_cuda_device()
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


class PseudoVideoGenerator:
    def __init__(self, target_height: int, target_width: int, num_frames: int):
        self.target_height = target_height
        self.target_width = target_width
        self.num_frames = num_frames

    def _ensure_large_enough(self, image: Image.Image) -> Image.Image:
        img_w, img_h = image.size
        if img_w >= self.target_width * 2 and img_h >= self.target_height * 2:
            return image
        scale = max(2.0 * self.target_width / img_w, 2.0 * self.target_height / img_h)
        resized = image.resize((int(round(img_w * scale)), int(round(img_h * scale))), Image.LANCZOS)
        return resized

    def _pan(self, image: Image.Image, rng: random.Random) -> List[Image.Image]:
        image = self._ensure_large_enough(image)
        img_w, img_h = image.size
        max_x = max(0, img_w - self.target_width)
        max_y = max(0, img_h - self.target_height)
        start_x = rng.randint(0, max_x) if max_x > 0 else 0
        start_y = rng.randint(0, max_y) if max_y > 0 else 0
        end_x = rng.randint(0, max_x) if max_x > 0 else 0
        end_y = rng.randint(0, max_y) if max_y > 0 else 0
        frames = []
        for idx in range(self.num_frames):
            alpha = idx / max(1, self.num_frames - 1)
            cur_x = int(round((1 - alpha) * start_x + alpha * end_x))
            cur_y = int(round((1 - alpha) * start_y + alpha * end_y))
            frame = image.crop((cur_x, cur_y, cur_x + self.target_width, cur_y + self.target_height))
            frames.append(frame)
        return frames

    def _zoom(self, image: Image.Image, rng: random.Random) -> List[Image.Image]:
        image = self._ensure_large_enough(image)
        img_w, img_h = image.size
        max_zoom = min(img_w / self.target_width, img_h / self.target_height, 2.5)
        zoom = rng.uniform(1.1, max_zoom)
        center_x = img_w // 2
        center_y = img_h // 2
        frames = []
        for idx in range(self.num_frames):
            alpha = idx / max(1, self.num_frames - 1)
            current_zoom = zoom - (zoom - 1.0) * alpha
            crop_w = int(round(self.target_width * current_zoom))
            crop_h = int(round(self.target_height * current_zoom))
            left = max(0, min(img_w - crop_w, center_x - crop_w // 2))
            top = max(0, min(img_h - crop_h, center_y - crop_h // 2))
            frame = image.crop((left, top, left + crop_w, top + crop_h))
            frame = frame.resize((self.target_width, self.target_height), Image.LANCZOS)
            frames.append(frame)
        if rng.random() < 0.5:
            frames.reverse()
        return frames

    def generate(self, image: Image.Image, seed: Optional[int] = None, rng: Optional[random.Random] = None) -> List[Image.Image]:
        if rng is None:
            rng = random.Random(seed) if seed is not None else random.Random()
        if rng.random() < 0.5:
            return self._pan(image, rng)
        return self._zoom(image, rng)


class FlashVSRStreamingDataset(IterableDataset):
    """
    Streaming raw-video dataset for FlashVSR Stage 1 baseline.

    Supported modes:
    - TAR-sharded videos via WebDataset
    - Direct video file lists discovered under a prefix
    - Optional image dataset mixed in as pseudo videos

    Output format:
    - video: List[PIL.Image]
    - lq_video: List[PIL.Image]
    """

    load_from_cache = False
    custom_collate_fn = None

    def __init__(
        self,
        internal_url: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        stride: int = 1,
        max_source_frames: int = 160,
        image_internal_url: Optional[str] = None,
        image_dataset_prob: float = 0.0,
        enable_degradation: bool = True,
        degradation_seed: Optional[int] = None,
        hq_prefix_frames: int = 0,
        control_dropout_prob: float = 0.0,
        shuffle_buffer: int = 100,
        max_raw_image_bytes: int = 50_000_000,
        global_seed: Optional[int] = None,
        metadata_url: Optional[str] = None,
        metadata_source: str = "auto",
        max_parquet_records: Optional[int] = None,
        min_overall_score: Optional[float] = None,
        require_qwen35_parse_success: bool = False,
        degradation_config_path: Optional[str] = None,
        output_tensors: bool = False,
    ):
        super().__init__()
        self.internal_url = internal_url
        self.image_internal_url = image_internal_url
        self.metadata_url = metadata_url
        self.metadata_source = metadata_source
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.stride = stride
        self.max_source_frames = max_source_frames
        self.image_dataset_prob = image_dataset_prob
        self.enable_degradation = enable_degradation
        self.degradation_seed = degradation_seed
        self.hq_prefix_frames = hq_prefix_frames
        self.control_dropout_prob = control_dropout_prob
        self.shuffle_buffer = shuffle_buffer
        self.max_raw_image_bytes = max_raw_image_bytes
        self.global_seed = global_seed
        self.max_parquet_records = max_parquet_records
        self.min_overall_score = min_overall_score
        self.require_qwen35_parse_success = require_qwen35_parse_success
        self.degradation_config_path = degradation_config_path
        self.output_tensors = output_tensors

        self.frame_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=height * width,
            height_division_factor=16,
            width_division_factor=16,
        )
        self.pseudo_video_generator = PseudoVideoGenerator(height, width, num_frames)
        self.degradation_model = ConsistentClipDegradation(config_path=degradation_config_path) if enable_degradation else None

        self.parquet_records: List[FlashVSRParquetRecord] = []
        if metadata_url is not None:
            self.parquet_records = load_parquet_records(
                metadata_url=metadata_url,
                dataset_source=metadata_source,
                max_records=max_parquet_records,
                min_overall_score=min_overall_score,
                require_qwen35_parse_success=require_qwen35_parse_success,
            )

        self.video_manifest_urls, self.video_urls = self._discover_sample_sources(self.internal_url, VIDEO_EXTENSIONS + (".tar",))
        self.image_manifest_urls, self.image_urls = self._discover_sample_sources(self.image_internal_url, IMAGE_EXTENSIONS + (".tar",))

        self.video_tar_urls = [url for url in self.video_urls if str(url).endswith(".tar")]
        self.video_file_urls = [url for url in self.video_urls if not str(url).endswith(".tar")]
        self.image_tar_urls = [url for url in self.image_urls if str(url).endswith(".tar")]
        self.image_file_urls = [url for url in self.image_urls if not str(url).endswith(".tar")]

        if not self.video_urls and not self.image_urls and not self.parquet_records and not self.video_manifest_urls and not self.image_manifest_urls:
            raise ValueError("No video or image samples were discovered for FlashVSR streaming dataset.")

        if num_frames % 4 != 1:
            raise ValueError(f"num_frames must follow 4n+1 pattern, got {num_frames}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if self.output_tensors:
            self.custom_collate_fn = self.tensor_collate_fn

    def _meets_min_resolution(self, width: int, height: int) -> bool:
        return int(width) >= int(self.width) and int(height) >= int(self.height)

    def _discover_urls(self, base_url: Optional[str], suffixes: Sequence[str]) -> List[str]:
        base_urls = [normalize_remote_url(url) for url in _expand_input_urls(base_url)]
        if not base_urls:
            return []

        merged_urls: List[str] = []
        for one_base_url in base_urls:
            _train_debug_print(f"discover_begin base={one_base_url} suffixes={','.join(suffixes)}")
            urls: List[str] = []
            is_remote = one_base_url.startswith(REMOTE_DISCOVERY_PREFIXES)
            if os.path.isfile(one_base_url) and _looks_like_manifest(one_base_url):
                _train_debug_print(f"discover_manifest_read_begin path={one_base_url}")
                urls = _load_manifest_entries(one_base_url)
                _train_debug_print(f"discover_manifest_read_end path={one_base_url} count={len(urls)}")

            if not urls and is_remote and str(one_base_url).endswith(tuple(suffixes)):
                urls = [one_base_url]
                _train_debug_print(f"discover_direct_remote_file base={one_base_url}")

            if not urls and is_remote:
                cache_path, lock_path = _discovery_cache_paths(one_base_url, suffixes)
                _train_debug_print(f"discover_remote_cache_check base={one_base_url} cache={cache_path} lock={lock_path}")
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    rank = dist.get_rank()
                    discovery_payload: List[Optional[Dict[str, Any]]] = [None]
                    if rank == 0:
                        try:
                            _train_debug_print(f"discover_dist_rank0_begin base={one_base_url}")
                            rank0_urls = _read_discovery_cache(cache_path)
                            if rank0_urls is not None:
                                _train_debug_print(f"discover_dist_rank0_cache_hit base={one_base_url} count={len(rank0_urls)}")
                            if rank0_urls is None:
                                while True:
                                    try:
                                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                        os.close(fd)
                                        _train_debug_print(f"discover_dist_rank0_lock_acquired base={one_base_url}")
                                        break
                                    except FileExistsError:
                                        _try_remove_stale_discovery_lock(lock_path)
                                        rank0_urls = _read_discovery_cache(cache_path)
                                        if rank0_urls is not None:
                                            _train_debug_print(f"discover_dist_rank0_cache_after_lock base={one_base_url} count={len(rank0_urls)}")
                                            break
                                        time.sleep(1.0)
                            if rank0_urls is None:
                                try:
                                    _train_debug_print(f"discover_dist_rank0_remote_list_begin base={one_base_url}")
                                    rank0_urls = [
                                        normalize_remote_url(item)
                                        for item in list_remote_files_with_suffixes(one_base_url, suffixes)
                                    ]
                                    _train_debug_print(f"discover_dist_rank0_remote_list_end base={one_base_url} count={len(rank0_urls)}")
                                    _write_discovery_cache(cache_path, rank0_urls)
                                finally:
                                    try:
                                        os.remove(lock_path)
                                    except FileNotFoundError:
                                        pass
                            discovery_payload[0] = {"ok": True, "urls": rank0_urls}
                        except Exception as error:
                            discovery_payload[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    _train_debug_print(f"discover_dist_broadcast_begin base={one_base_url}")
                    dist.broadcast_object_list(discovery_payload, src=0)
                    _train_debug_print(f"discover_dist_broadcast_end base={one_base_url}")
                    payload = discovery_payload[0] or {"ok": False, "error": "empty discovery payload"}
                    if not payload.get("ok", False):
                        raise RuntimeError(
                            f"rank0 remote discovery failed for {one_base_url}: {payload.get('error')}"
                        )
                    urls = [normalize_remote_url(item) for item in payload.get("urls", [])]
                    if rank != 0 and urls:
                        try:
                            _write_discovery_cache(cache_path, urls)
                        except Exception:
                            pass
                else:
                    cached_urls = _read_discovery_cache(cache_path)
                    if cached_urls is not None:
                        urls = cached_urls
                        _train_debug_print(f"discover_remote_cache_hit base={one_base_url} count={len(urls)}")
                        # Skip node-local lock/list path when this node already has a valid cache.
                        filtered = [url for url in urls if str(url).endswith(tuple(suffixes))]
                        if filtered:
                            merged_urls.extend(filtered)
                        elif not is_remote:
                            merged_urls.extend(urls)
                        _train_debug_print(f"discover_end base={one_base_url} urls={len(urls)} filtered={len(filtered)} merged={len(merged_urls)}")
                        continue
                    local_rank = _node_local_rank()
                    if local_rank == 0:
                        _train_debug_print(f"discover_node_local0_begin base={one_base_url}")
                        while True:
                            try:
                                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                                os.close(fd)
                                _train_debug_print(f"discover_node_local0_lock_acquired base={one_base_url}")
                                break
                            except FileExistsError:
                                _try_remove_stale_discovery_lock(lock_path)
                                cached_urls = _read_discovery_cache(cache_path)
                                if cached_urls is not None:
                                    urls = cached_urls
                                    _train_debug_print(f"discover_node_local0_cache_after_lock base={one_base_url} count={len(urls)}")
                                    break
                                time.sleep(1.0)
                        if not urls:
                            try:
                                _train_debug_print(f"discover_node_local0_remote_list_begin base={one_base_url}")
                                urls = [
                                    normalize_remote_url(item)
                                    for item in list_remote_files_with_suffixes(one_base_url, suffixes)
                                ]
                                _train_debug_print(f"discover_node_local0_remote_list_end base={one_base_url} count={len(urls)}")
                                _write_discovery_cache(cache_path, urls)
                            finally:
                                try:
                                    os.remove(lock_path)
                                except FileNotFoundError:
                                    pass
                    else:
                        _train_debug_print(f"discover_node_wait_cache_begin base={one_base_url}")
                        wait_started_at = time.time()
                        while True:
                            cached_urls = _read_discovery_cache(cache_path)
                            if cached_urls is not None:
                                urls = cached_urls
                                _train_debug_print(f"discover_node_wait_cache_end base={one_base_url} count={len(urls)}")
                                break
                            if time.time() - wait_started_at > 1800:
                                raise TimeoutError(
                                    f"timed out waiting for remote discovery cache for {one_base_url}"
                                )
                            time.sleep(1.0)

            if not urls and parabolt is not None and not is_remote:
                try:
                    urls = list(parabolt.io.find_files(one_base_url))
                except Exception:
                    urls = []

            if not urls and os.path.exists(one_base_url):
                if os.path.isdir(one_base_url):
                    for root, _, files in os.walk(one_base_url):
                        for file_name in files:
                            urls.append(os.path.join(root, file_name))
                else:
                    urls = [one_base_url]

            if not urls:
                if is_remote:
                    raise RuntimeError(f"failed to discover remote urls for {one_base_url} via unified conductor bridge")
                urls = [one_base_url]

            filtered = [url for url in urls if str(url).endswith(tuple(suffixes))]
            if filtered:
                merged_urls.extend(filtered)
            elif not is_remote:
                merged_urls.extend(urls)
            _train_debug_print(f"discover_end base={one_base_url} urls={len(urls)} filtered={len(filtered)} merged={len(merged_urls)}")

        return sorted(set(merged_urls))

    def _discover_sample_sources(self, base_url: Optional[str], suffixes: Sequence[str]) -> Tuple[List[str], List[str]]:
        base_urls = [normalize_remote_url(url) for url in _expand_input_urls(base_url)]
        if not base_urls:
            return [], []
        manifest_urls = [url for url in base_urls if os.path.isfile(url) and _looks_like_manifest(url)]
        non_manifest_urls = [url for url in base_urls if url not in manifest_urls]
        discovered_urls = self._discover_urls(",".join(non_manifest_urls), suffixes) if non_manifest_urls else []
        return manifest_urls, discovered_urls

    def _split_for_worker(self, urls: List[str]) -> List[str]:
        worker = get_worker_info()
        if worker is None:
            return urls
        return urls[worker.id :: worker.num_workers]

    def _split_for_process_and_worker(self, items: List[Any]) -> List[Any]:
        rank, world_size = _distributed_rank_world_size()
        sharded = items[rank::world_size]
        worker = get_worker_info()
        if worker is None:
            return sharded
        return sharded[worker.id :: worker.num_workers]

    def _split_for_process(self, items: List[Any]) -> List[Any]:
        rank, world_size = _distributed_rank_world_size()
        return items[rank::world_size]

    def _make_iteration_rng(self) -> random.Random:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rank, _ = _distributed_rank_world_size()
        if self.global_seed is not None:
            seed = int(self.global_seed) + rank * 10000019 + worker_id * 1000003
        elif self.degradation_seed is not None:
            seed = int(self.degradation_seed) + rank * 10000019 + worker_id * 1000003
        else:
            seed = random.SystemRandom().randint(0, 2**31 - 1)
        return random.Random(seed)

    def _rank_worker_shard(self) -> Tuple[int, int]:
        rank, world_size = _distributed_rank_world_size()
        worker = get_worker_info()
        if worker is None:
            return rank, world_size
        shard_id = rank * worker.num_workers + worker.id
        num_shards = world_size * worker.num_workers
        return shard_id, num_shards

    def _iter_manifest_entries(self, manifest_paths: Sequence[str]) -> Iterator[str]:
        shard_id, num_shards = self._rank_worker_shard()
        while True:
            for manifest_path in manifest_paths:
                is_jsonl = str(manifest_path).lower().endswith(".jsonl")
                with open(manifest_path, "r", encoding="utf-8") as file:
                    for line_index, line in enumerate(file):
                        if line_index % num_shards != shard_id:
                            continue
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if is_jsonl:
                            payload = json.loads(line)
                            if isinstance(payload, dict):
                                media_url = payload.get("media_url")
                                if media_url:
                                    yield normalize_remote_url(str(media_url))
                            elif isinstance(payload, str):
                                yield normalize_remote_url(payload)
                        else:
                            yield normalize_remote_url(line)

    def _next_sample_seed(self, rng: random.Random) -> int:
        return rng.randint(0, 2**31 - 1)

    @staticmethod
    def _coprime_step(length: int, rng: random.Random) -> int:
        if length <= 1:
            return 1
        step = rng.randrange(1, length)
        while math.gcd(step, length) != 1:
            step = rng.randrange(1, length)
        return step

    def _iter_deterministic_permutation(self, items: Sequence[Any], rng: random.Random) -> Iterator[Any]:
        length = len(items)
        if length == 0:
            return
        if length == 1:
            while True:
                yield items[0]
        while True:
            start = rng.randrange(length)
            step = self._coprime_step(length, rng)
            for offset in range(length):
                yield items[(start + offset * step) % length]

    @staticmethod
    def _pil_to_tensor(frame: Image.Image) -> torch.Tensor:
        frame = frame.convert("RGB")
        array = np.asarray(frame, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(dtype=torch.bfloat16)

    def _maybe_convert_output(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if not self.output_tensors:
            return sample
        global _CONVERT_DEBUG_COUNT
        output = {
            "video": torch.stack([self._pil_to_tensor(frame) for frame in sample["video"]], dim=0),
            "lq_video": torch.stack([self._pil_to_tensor(frame) for frame in sample["lq_video"]], dim=0),
            "sample_seed": torch.tensor(sample["sample_seed"], dtype=torch.long),
        }
        if _CONVERT_DEBUG_COUNT < 4:
            _dataset_debug_log(
                "convert_output "
                f"video_shape={tuple(output['video'].shape)} video_dtype={output['video'].dtype} "
                f"lq_shape={tuple(output['lq_video'].shape)} lq_dtype={output['lq_video'].dtype} "
                f"sample_seed={int(output['sample_seed'])}"
            )
            _CONVERT_DEBUG_COUNT += 1
        return output

    @staticmethod
    def tensor_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch is not supported.")
        global _COLLATE_DEBUG_COUNT
        if _COLLATE_DEBUG_COUNT < 4:
            keys = list(batch[0].keys())
            tensor_keys = [key for key, value in batch[0].items() if torch.is_tensor(value)]
            if _TRAIN_DEBUG_ENABLED:
                _dataset_debug_log(
                    f"tensor_collate_fn begin batch_len={len(batch)} keys={keys} tensor_keys={tensor_keys}"
                )
                print(
                    f"[tensor_collate_fn] batch_len={len(batch)} tensor_keys={tensor_keys}",
                    flush=True,
                )
        output = {}
        for key in batch[0].keys():
            value = batch[0][key]
            if torch.is_tensor(value):
                output[key] = torch.stack([sample[key] for sample in batch], dim=0)
        if _COLLATE_DEBUG_COUNT < 4:
            shape_info = {
                key: (tuple(value.shape), str(value.dtype))
                for key, value in output.items()
                if torch.is_tensor(value)
            }
            if _TRAIN_DEBUG_ENABLED:
                _dataset_debug_log(f"tensor_collate_fn end shape_info={shape_info}")
                print(f"[tensor_collate_fn] end shape_info={shape_info}", flush=True)
            _COLLATE_DEBUG_COUNT += 1
        return output

    def _open_binary(self, url: str) -> bytes:
        url = normalize_remote_url(url)
        timing_start = time.perf_counter() if _DATA_TIMING_DEBUG_ENABLED else None
        data = None
        backend = None
        if fsspec is not None:
            try:
                with fsspec.open(url, "rb").open() as file:
                    data = file.read()
                backend = "fsspec"
                if timing_start is not None:
                    _data_timing_log(
                        f"open_binary backend={backend} bytes={len(data)} elapsed={time.perf_counter() - timing_start:.3f}s url={url}"
                    )
                return data
            except Exception:
                pass
        if parabolt is not None and hasattr(parabolt, "io") and hasattr(parabolt.io, "open"):
            with parabolt.io.open(url, "rb") as file:
                data = file.read()
            backend = "parabolt"
            if timing_start is not None:
                _data_timing_log(
                    f"open_binary backend={backend} bytes={len(data)} elapsed={time.perf_counter() - timing_start:.3f}s url={url}"
                )
            return data
        with open(url, "rb") as file:
            data = file.read()
        backend = "local"
        if timing_start is not None:
            _data_timing_log(
                f"open_binary backend={backend} bytes={len(data)} elapsed={time.perf_counter() - timing_start:.3f}s url={url}"
            )
        return data

    @contextmanager
    def _open_stream(self, url: str):
        url = normalize_remote_url(url)
        if fsspec is not None:
            try:
                with fsspec.open(url, "rb").open() as file:
                    yield file
                    return
            except Exception:
                pass
        if parabolt is not None and hasattr(parabolt, "io") and hasattr(parabolt.io, "open"):
            with parabolt.io.open(url, "rb") as file:
                yield file
                return
        with open(url, "rb") as file:
            yield file

    def _extract_frames(self, video_bytes: bytes) -> Optional[List[Image.Image]]:
        temp_path = None
        frames: List[Image.Image] = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_bytes)
                temp_path = tmp.name
            cap = cv2.VideoCapture(temp_path)
            while len(frames) < self.max_source_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_h, frame_w = frame_bgr.shape[:2]
                if not self._meets_min_resolution(frame_w, frame_h):
                    cap.release()
                    return None
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_rgb)
                frame = self.frame_processor(frame)
                frames.append(frame)
            cap.release()
            if len(frames) < (self.num_frames - 1) * self.stride + 1:
                return None
            return frames
        except Exception as error:
            warnings.warn(f"Failed to decode video sample: {error}")
            return None
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _select_clip(self, frames: List[Image.Image], rng: random.Random) -> Optional[List[Image.Image]]:
        needed = (self.num_frames - 1) * self.stride + 1
        if len(frames) < needed:
            return None
        max_start = len(frames) - needed
        if max_start < 1:
            return None
        start = rng.randint(1, max_start)
        return [frames[start + idx * self.stride] for idx in range(self.num_frames)]

    def _build_lq_clip(self, hr_frames: List[Image.Image], rng: random.Random, sample_seed: int) -> List[Image.Image]:
        if self.control_dropout_prob > 0.0 and rng.random() < self.control_dropout_prob:
            return [Image.new("RGB", frame.size, (0, 0, 0)) for frame in hr_frames]

        if self.degradation_model is None:
            lq_frames = [frame.copy() for frame in hr_frames]
        else:
            lq_frames = self.degradation_model.degrade_batch_consistent(hr_frames, seed=sample_seed)

        if self.hq_prefix_frames > 0:
            for idx in range(min(self.hq_prefix_frames, len(hr_frames))):
                lq_frames[idx] = hr_frames[idx].copy()
        return lq_frames

    def _process_video_bytes(self, video_bytes: bytes, sample_id: str, rng: random.Random) -> Optional[Dict[str, Any]]:
        global _DATA_TIMING_DEBUG_COUNT
        timing_enabled = _DATA_TIMING_DEBUG_ENABLED and _DATA_TIMING_DEBUG_COUNT < _DATA_TIMING_DEBUG_MAX
        timing_start = time.perf_counter() if timing_enabled else None
        frames = self._extract_frames(video_bytes)
        timing_after_decode = time.perf_counter() if timing_enabled else None
        if frames is None:
            if timing_enabled:
                _data_timing_log(
                    f"process_video sample_id={sample_id} bytes={len(video_bytes)} decode={timing_after_decode - timing_start:.3f}s rejected=decode"
                )
                _DATA_TIMING_DEBUG_COUNT += 1
            return None
        clip = self._select_clip(frames, rng=rng)
        timing_after_select = time.perf_counter() if timing_enabled else None
        if clip is None:
            if timing_enabled:
                _data_timing_log(
                    f"process_video sample_id={sample_id} bytes={len(video_bytes)} frames={len(frames)} "
                    f"decode={timing_after_decode - timing_start:.3f}s select={timing_after_select - timing_after_decode:.3f}s rejected=clip"
                )
                _DATA_TIMING_DEBUG_COUNT += 1
            return None
        sample_seed = self._next_sample_seed(rng)
        lq_video = self._build_lq_clip(clip, rng=rng, sample_seed=sample_seed)
        timing_after_degrade = time.perf_counter() if timing_enabled else None
        output = self._maybe_convert_output({
            "video": clip,
            "lq_video": lq_video,
            "sample_id": sample_id,
            "source_type": "video",
            "sample_seed": sample_seed,
        })
        if timing_enabled:
            timing_after_convert = time.perf_counter()
            _data_timing_log(
                f"process_video sample_id={sample_id} bytes={len(video_bytes)} frames={len(frames)} "
                f"decode={timing_after_decode - timing_start:.3f}s "
                f"select={timing_after_select - timing_after_decode:.3f}s "
                f"degrade={timing_after_degrade - timing_after_select:.3f}s "
                f"convert={timing_after_convert - timing_after_degrade:.3f}s "
                f"total={timing_after_convert - timing_start:.3f}s"
            )
            _DATA_TIMING_DEBUG_COUNT += 1
        return output

    def _process_image(self, image: Image.Image, sample_id: str, rng: random.Random) -> Optional[Dict[str, Any]]:
        try:
            image = image.convert("RGB")
            if not self._meets_min_resolution(*image.size):
                return None
            sample_seed = self._next_sample_seed(rng)
            pseudo_rng = random.Random(sample_seed)
            frames = self.pseudo_video_generator.generate(image, seed=sample_seed, rng=pseudo_rng)
            frames = [self.frame_processor(frame) for frame in frames]
            return self._maybe_convert_output({
                "video": frames,
                "lq_video": self._build_lq_clip(frames, rng=rng, sample_seed=sample_seed),
                "sample_id": sample_id,
                "source_type": "image",
                "sample_seed": sample_seed,
            })
        except Exception as error:
            warnings.warn(f"Failed to process image sample: {error}")
            return None

    def _extract_tar_member_bytes(self, shard_url: str, member_path: str) -> Optional[bytes]:
        try:
            with self._open_stream(shard_url) as file:
                with tarfile.open(fileobj=file, mode="r|*") as tar:
                    for member in tar:
                        if member.name == member_path or member.name.endswith(member_path):
                            extracted = tar.extractfile(member)
                            if extracted is None:
                                return None
                            return extracted.read()
        except Exception as error:
            warnings.warn(f"Failed to extract {member_path} from shard {shard_url}: {error}")
        return None

    def _process_parquet_record(self, record: FlashVSRParquetRecord, rng: random.Random) -> Optional[Dict[str, Any]]:
        if record.tar_member_path is None:
            video_bytes = self._open_binary(record.media_path)
        else:
            video_bytes = self._extract_tar_member_bytes(record.media_path, record.tar_member_path)
            if video_bytes is None:
                return None
        sample = self._process_video_bytes(video_bytes, sample_id=record.sample_id, rng=rng)
        if sample is None:
            return None
        if self.output_tensors:
            return sample
        sample["source_dataset"] = record.dataset_source
        sample["caption_text"] = record.caption_text
        sample["metadata"] = record.metadata
        sample["media_path"] = record.media_path
        if record.tar_member_path is not None:
            sample["tar_member_path"] = record.tar_member_path
        return sample

    def _make_torchdata_tar_pipe(self, urls: List[str], rng: Optional[random.Random] = None):
        if IterableWrapper is None:
            raise ImportError("torchdata is required for TAR-based FlashVSR streaming datasets.")
        ordered_urls = self._split_for_process(list(urls))
        if not ordered_urls:
            ordered_urls = list(urls)
        if rng is not None:
            rng.shuffle(ordered_urls)
        datapipe = IterableWrapper(ordered_urls)
        if rng is None:
            datapipe = datapipe.shuffle(buffer_size=max(self.shuffle_buffer, 1))
        if not hasattr(datapipe, "sharding_filter"):
            raise AttributeError("Current datapipe implementation does not provide sharding_filter.")
        sharded = datapipe.sharding_filter()
        if not hasattr(sharded, "open_files_by_fsspec"):
            raise AttributeError("Current datapipe implementation does not provide open_files_by_fsspec.")
        if not hasattr(sharded.open_files_by_fsspec(mode='rb'), "load_from_tar"):
            raise AttributeError("Current datapipe implementation does not provide load_from_tar.")
        datapipe = sharded.open_files_by_fsspec(mode="rb")
        datapipe = datapipe.load_from_tar("r:")
        return datapipe

    def _raw_size_prefilter(self, sample: Dict[str, bytes]) -> bool:
        for key, value in sample.items():
            if any(key.endswith(ext) for ext in IMAGE_EXTENSIONS) and len(value) > self.max_raw_image_bytes:
                return False
        return True

    def _iterate_direct_videos(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        urls = self._split_for_process_and_worker(self.video_file_urls)
        logged_first = False
        for url in self._iter_deterministic_permutation(urls, rng) if urls else []:
            if not logged_first:
                _dataset_debug_log(f"direct_video first_url={url}")
                logged_first = True
            started_at = time.time()
            sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
            elapsed = time.time() - started_at
            if sample is not None:
                if logged_first:
                    _dataset_debug_log(
                        f"direct_video yielded sample_id={sample.get('sample_id')} elapsed_sec={elapsed:.3f}"
                    )
                    logged_first = False
                yield sample
        if self.video_manifest_urls:
            for url in self._iter_manifest_entries(self.video_manifest_urls):
                if not logged_first:
                    _dataset_debug_log(f"manifest_video first_url={url}")
                    logged_first = True
                started_at = time.time()
                sample = self._process_video_bytes(self._open_binary(url), sample_id=os.path.basename(url), rng=rng)
                elapsed = time.time() - started_at
                if sample is not None:
                    if logged_first:
                        _dataset_debug_log(
                            f"manifest_video yielded sample_id={sample.get('sample_id')} elapsed_sec={elapsed:.3f}"
                        )
                        logged_first = False
                    yield sample

    def _iterate_direct_images(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        urls = self._split_for_process_and_worker(self.image_file_urls)
        for url in self._iter_deterministic_permutation(urls, rng) if urls else []:
            try:
                image = Image.open(io.BytesIO(self._open_binary(url))).convert("RGB")
            except Exception as error:
                warnings.warn(f"Failed to open image sample {url}: {error}")
                continue
            sample = self._process_image(image, sample_id=os.path.basename(url), rng=rng)
            if sample is not None:
                yield sample
        if self.image_manifest_urls:
            for url in self._iter_manifest_entries(self.image_manifest_urls):
                try:
                    image = Image.open(io.BytesIO(self._open_binary(url))).convert("RGB")
                except Exception as error:
                    warnings.warn(f"Failed to open manifest image sample {url}: {error}")
                    continue
                sample = self._process_image(image, sample_id=os.path.basename(url), rng=rng)
                if sample is not None:
                    yield sample

    def _iterate_tar_videos(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        if not self.video_tar_urls:
            return
        _dataset_debug_log(f"tar_video build_pipe num_shards={len(self.video_tar_urls)}")
        datapipe = self._make_torchdata_tar_pipe(self.video_tar_urls, rng=rng if self.global_seed is not None else None)
        logged_first = False
        while True:
            for file_name, stream_item in datapipe:
                if not str(file_name).endswith(VIDEO_EXTENSIONS):
                    continue
                if not logged_first:
                    _dataset_debug_log(f"tar_video first_member={file_name}")
                    logged_first = True
                video_bytes = stream_item.read()
                started_at = time.time()
                processed = self._process_video_bytes(
                    video_bytes,
                    sample_id=os.path.basename(str(file_name)),
                    rng=rng,
                )
                elapsed = time.time() - started_at
                if processed is not None:
                    if logged_first:
                        _dataset_debug_log(
                            f"tar_video yielded sample_id={processed.get('sample_id')} elapsed_sec={elapsed:.3f}"
                        )
                        logged_first = False
                    if not self.output_tensors:
                        processed["tar_member_path"] = str(file_name)
                    yield processed

    def _iterate_tar_images(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        if not self.image_tar_urls:
            return
        datapipe = self._make_torchdata_tar_pipe(self.image_tar_urls, rng=rng if self.global_seed is not None else None)
        while True:
            for file_name, stream_item in datapipe:
                if not str(file_name).endswith(IMAGE_EXTENSIONS):
                    continue
                image = Image.open(io.BytesIO(stream_item.read())).convert("RGB")
                processed = self._process_image(
                    image,
                    sample_id=os.path.basename(str(file_name)),
                    rng=rng,
                )
                if processed is not None:
                    if not self.output_tensors:
                        processed["tar_member_path"] = str(file_name)
                    yield processed

    def _iterate_parquet_videos(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        records = self._split_for_process_and_worker(self.parquet_records)
        if not records:
            return
        for record in self._iter_deterministic_permutation(records, rng):
            sample = self._process_parquet_record(record, rng=rng)
            if sample is not None:
                yield sample

    def _video_iterator(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        iterators: List[Iterator[Dict[str, Any]]] = []
        if self.parquet_records:
            iterators.append(self._iterate_parquet_videos(rng=rng))
        if self.video_tar_urls:
            iterators.append(self._iterate_tar_videos(rng=rng))
        if self.video_file_urls:
            iterators.append(self._iterate_direct_videos(rng=rng))
        if not iterators:
            return
        if len(iterators) == 1:
            yield from iterators[0]
            return
        while True:
            yield next(iterators[rng.randrange(len(iterators))])

    def _image_iterator(self, rng: random.Random) -> Iterator[Dict[str, Any]]:
        iterators: List[Iterator[Dict[str, Any]]] = []
        if self.image_tar_urls:
            iterators.append(self._iterate_tar_images(rng=rng))
        if self.image_file_urls or self.image_manifest_urls:
            iterators.append(self._iterate_direct_images(rng=rng))
        if not iterators:
            return
        if len(iterators) == 1:
            yield from iterators[0]
            return
        while True:
            yield next(iterators[rng.randrange(len(iterators))])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = self._make_iteration_rng()
        video_iter = self._video_iterator(rng=rng) if (self.video_urls or self.parquet_records) else None
        image_iter = self._image_iterator(rng=rng) if self.image_urls else None

        while True:
            use_image = image_iter is not None and (
                video_iter is None or rng.random() < self.image_dataset_prob
            )
            if use_image:
                yield next(image_iter)
            else:
                yield next(video_iter)
