import io
import os
import random
import tarfile
import tempfile
import warnings
import hashlib
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence

import cv2
from PIL import Image

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None

from .parquet_index import normalize_remote_url
from .conductor_bridge_v2 import (
    get_conductor_local_path,
    has_ref_big_conductor,
    list_remote_files_with_suffixes,
    open_conductor_stream,
    read_conductor_bytes,
)


REMOTE_DISCOVERY_PREFIXES = ("conductor://", "s3://", "blobby://")


class FlashVSRMediaReaderV2:
    """
    Media access layer for FlashVSR parquet-driven datasets.

    This keeps remote file discovery / remote file access / tar extraction
    separate from sample indexing so the dataset class stays focused on
    sampling logic.
    """

    def __init__(
        self,
        frame_processor,
        num_frames: int,
        stride: int,
        max_source_frames: int,
        media_cache_dir: Optional[str] = None,
    ):
        self.frame_processor = frame_processor
        self.num_frames = num_frames
        self.stride = stride
        self.max_source_frames = max_source_frames
        self.media_cache_dir = media_cache_dir

    def _cached_media_path(self, normalized_url: str, suffix: str = "") -> Optional[str]:
        if not self.media_cache_dir:
            return None
        os.makedirs(self.media_cache_dir, exist_ok=True)
        digest = hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()
        base_name = os.path.basename(normalized_url) or "cached_media"
        return os.path.join(self.media_cache_dir, f"{digest}_{base_name}{suffix}")

    @contextmanager
    def open_stream(self, url: str):
        url = normalize_remote_url(url)
        if str(url).startswith("conductor://"):
            if not has_ref_big_conductor():
                raise RuntimeError(f"ref_big conductor client unavailable for {url}")
            with open_conductor_stream(url, "rb") as file:
                yield file
                return
        if fsspec is not None:
            try:
                with fsspec.open(url, "rb").open() as file:
                    yield file
                    return
            except Exception:
                pass
        if parabolt is not None and hasattr(parabolt, "io") and hasattr(parabolt.io, "open"):
            try:
                with parabolt.io.open(url, "rb") as file:
                    yield file
                    return
            except Exception:
                pass
        with open(url, "rb") as file:
            yield file

    def discover_urls(self, base_url: Optional[str], suffixes: Sequence[str]) -> List[str]:
        base_urls = [normalize_remote_url(item) for item in str(base_url or "").replace("\n", ",").split(",") if item.strip()]
        if not base_urls:
            return []

        merged_urls: List[str] = []
        for one_base_url in base_urls:
            if str(one_base_url).endswith(tuple(suffixes)):
                merged_urls.append(one_base_url)
                continue
            urls: List[str] = []
            if parabolt is not None:
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
            if not urls and one_base_url.startswith(REMOTE_DISCOVERY_PREFIXES):
                urls = [normalize_remote_url(item) for item in list_remote_files_with_suffixes(one_base_url, suffixes)]
            if not urls:
                if one_base_url.startswith(REMOTE_DISCOVERY_PREFIXES):
                    raise RuntimeError(f"failed to discover remote urls for {one_base_url} via unified conductor bridge")
                urls = [one_base_url]
            merged_urls.extend(url for url in urls if str(url).endswith(tuple(suffixes)))
        return sorted(set(merged_urls))

    def extract_tar_member_bytes(self, shard_url: str, member_path: str) -> Optional[bytes]:
        try:
            with self.open_stream(shard_url) as file:
                with tarfile.open(fileobj=file, mode="r|*") as tar:
                    for member in tar:
                        if member.name == member_path or member.name.endswith(member_path):
                            extracted = tar.extractfile(member)
                            if extracted is None:
                                return None
                            return extracted.read()
        except Exception as error:
            warnings.warn(f"Failed to extract {member_path} from {shard_url}: {error}")

        local_tar = get_conductor_local_path(normalize_remote_url(shard_url)) if str(normalize_remote_url(shard_url)).startswith("conductor://") else None
        if local_tar is not None:
            try:
                with tarfile.open(local_tar, mode="r") as tar:
                    extracted = tar.extractfile(member_path)
                    if extracted is not None:
                        return extracted.read()
                    for member in tar.getmembers():
                        if member.name.endswith(member_path):
                            extracted = tar.extractfile(member)
                            if extracted is not None:
                                return extracted.read()
            except Exception as error:
                warnings.warn(f"Failed local cached tar read for {member_path} from {shard_url}: {error}")
        return None

    def list_tar_members(self, shard_url: str, suffixes: Optional[Sequence[str]] = None) -> List[str]:
        suffixes = tuple(suffixes or ())
        members: List[str] = []
        try:
            with self.open_stream(shard_url) as file:
                with tarfile.open(fileobj=file, mode="r|*") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        if suffixes and not str(member.name).endswith(suffixes):
                            continue
                        members.append(member.name)
            return members
        except Exception as error:
            warnings.warn(f"Failed to list tar members from {shard_url}: {error}")

        local_tar = get_conductor_local_path(normalize_remote_url(shard_url)) if str(normalize_remote_url(shard_url)).startswith("conductor://") else None
        if local_tar is None:
            return members
        try:
            with tarfile.open(local_tar, mode="r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if suffixes and not str(member.name).endswith(suffixes):
                        continue
                    members.append(member.name)
        except Exception as error:
            warnings.warn(f"Failed local tar listing for {shard_url}: {error}")
        return members

    def iter_tar_members(self, shard_url: str, suffixes: Optional[Sequence[str]] = None) -> Iterator[str]:
        suffixes = tuple(suffixes or ())
        try:
            with self.open_stream(shard_url) as file:
                with tarfile.open(fileobj=file, mode="r|*") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        if suffixes and not str(member.name).endswith(suffixes):
                            continue
                        yield member.name
            return
        except Exception as error:
            warnings.warn(f"Failed to iterate tar members from {shard_url}: {error}")

        local_tar = get_conductor_local_path(normalize_remote_url(shard_url)) if str(normalize_remote_url(shard_url)).startswith("conductor://") else None
        if local_tar is None:
            return
        try:
            with tarfile.open(local_tar, mode="r") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    if suffixes and not str(member.name).endswith(suffixes):
                        continue
                    yield member.name
        except Exception as error:
            warnings.warn(f"Failed local tar iteration for {shard_url}: {error}")

    def extract_indexed_tar_member_bytes(
        self,
        shard_url: str,
        suffixes: Sequence[str],
        member_index: int = 0,
    ) -> Optional[tuple[str, bytes]]:
        suffixes = tuple(suffixes)
        target_index = max(0, int(member_index))
        seen = 0
        try:
            with self.open_stream(shard_url) as file:
                with tarfile.open(fileobj=file, mode="r|*") as tar:
                    for member in tar:
                        if not member.isfile() or not str(member.name).endswith(suffixes):
                            continue
                        if seen == target_index:
                            extracted = tar.extractfile(member)
                            if extracted is None:
                                return None
                            return member.name, extracted.read()
                        seen += 1
        except Exception as error:
            warnings.warn(f"Failed to extract indexed tar member from {shard_url}: {error}")

        local_tar = get_conductor_local_path(normalize_remote_url(shard_url)) if str(normalize_remote_url(shard_url)).startswith("conductor://") else None
        if local_tar is None:
            return None
        try:
            with tarfile.open(local_tar, mode="r") as tar:
                for member in tar:
                    if not member.isfile() or not str(member.name).endswith(suffixes):
                        continue
                    if seen == target_index:
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            return None
                        return member.name, extracted.read()
                    seen += 1
        except Exception as error:
            warnings.warn(f"Failed local indexed tar extraction for {shard_url}: {error}")
        return None

    def read_media_bytes(self, media_url: str) -> Optional[bytes]:
        normalized = normalize_remote_url(media_url)
        if str(normalized).startswith("conductor://"):
            if not has_ref_big_conductor():
                raise RuntimeError(f"ref_big conductor client unavailable for {normalized}")
            try:
                return read_conductor_bytes(normalized)
            except Exception as error:
                warnings.warn(f"Failed to read media bytes via ref_big conductor from {media_url}: {error}")
        try:
            with self.open_stream(media_url) as file:
                return file.read()
        except Exception as error:
            warnings.warn(f"Failed to read media bytes from {media_url}: {error}")

        return None

    def read_media_range(self, media_url: str, offset: int, size: int) -> Optional[bytes]:
        normalized = normalize_remote_url(media_url)
        if normalized.startswith("conductor://"):
            if not has_ref_big_conductor():
                raise RuntimeError(f"ref_big conductor client unavailable for {normalized}")
            local_path = get_conductor_local_path(normalized)
            if local_path is not None:
                try:
                    with open(local_path, "rb") as file:
                        file.seek(int(offset))
                        return file.read(int(size))
                except Exception as error:
                    warnings.warn(f"Failed local cached byte range read from {media_url}: {error}")
        try:
            with self.open_stream(normalized) as file:
                file.seek(int(offset))
                return file.read(int(size))
        except Exception as error:
            warnings.warn(f"Failed to read byte range from {media_url}: {error}")
            return None

    def extract_frames(self, video_bytes: bytes) -> Optional[List[Image.Image]]:
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
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_rgb)
                frame = self.frame_processor(frame)
                frames.append(frame)
            cap.release()
            needed = (self.num_frames - 1) * self.stride + 1
            if len(frames) < needed:
                return None
            return frames
        finally:
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def select_clip(self, frames: List[Image.Image], rng: random.Random) -> Optional[List[Image.Image]]:
        needed = (self.num_frames - 1) * self.stride + 1
        if len(frames) < needed:
            return None
        max_start = len(frames) - needed
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return [frames[start + idx * self.stride] for idx in range(self.num_frames)]

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[Image.Image]:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as error:
            warnings.warn(f"Failed to decode image bytes: {error}")
            return None
