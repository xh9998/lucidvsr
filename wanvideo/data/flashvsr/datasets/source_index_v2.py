import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .media_reader_v2 import FlashVSRMediaReaderV2
from .parquet_index import (
    FlashVSRParquetRecord,
    _discover_parquet_urls,
    _read_parquet_frame,
    load_image_records,
    load_parquet_records,
    normalize_remote_url,
)
from .conductor_bridge_v2 import list_remote_files_with_suffixes


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class FlashVSRSourceIndexV2:
    records: List[FlashVSRParquetRecord]
    image_urls: List[str]
    takano_parquet_urls: List[str]
    image_parquet_urls: List[str]
    yubari_video_root_pairs: List[Dict[str, str]]

    @property
    def has_records(self) -> bool:
        return bool(self.records)

    @property
    def has_images(self) -> bool:
        return bool(self.image_urls)


def build_source_index_v2(
    media_reader: FlashVSRMediaReaderV2,
    metadata_url: Optional[str],
    metadata_source: str,
    image_metadata_url: Optional[str],
    image_internal_url: Optional[str],
    yubari_video_metadata_url: Optional[str],
    yubari_sidecar_metadata_url: Optional[str],
    yubari_video_tar_url: Optional[str] = None,
    yubari_sidecar_tar_url: Optional[str] = None,
    yubari_shard_start: Optional[int] = None,
    yubari_shard_end: Optional[int] = None,
    max_parquet_records: Optional[int] = None,
    max_yubari_records: Optional[int] = None,
    image_suffixes: Sequence[str] = IMAGE_EXTENSIONS,
) -> FlashVSRSourceIndexV2:
    records: List[FlashVSRParquetRecord] = []
    takano_parquet_urls: List[str] = []
    image_parquet_urls: List[str] = []
    yubari_video_root_pairs: List[Dict[str, str]] = []
    if metadata_url:
        if metadata_source == "takano" and max_parquet_records is None:
            takano_parquet_urls = _discover_parquet_urls(metadata_url)
        else:
            records.extend(
                load_parquet_records(
                    metadata_url=metadata_url,
                    dataset_source=metadata_source,
                    max_records=max_parquet_records,
                )
            )
    if image_metadata_url:
        if max_parquet_records is None:
            image_parquet_urls = _discover_parquet_urls(image_metadata_url)
        else:
            records.extend(
                load_image_records(
                    metadata_url=image_metadata_url,
                    max_records=max_parquet_records,
                )
            )
    if yubari_video_metadata_url or yubari_sidecar_metadata_url:
        warnings.warn(
            "Yubari metadata-root inputs are deprecated. Use yubari_video_tar_url only."
        )
    if yubari_sidecar_tar_url:
        warnings.warn(
            "Yubari sidecar tar input is deprecated for current training. Only yubari_video_tar_url is used."
        )
    if yubari_video_tar_url:
        if max_yubari_records is None:
            yubari_video_root_pairs = _discover_yubari_video_root_pairs(
                media_reader=media_reader,
                video_url=yubari_video_tar_url,
                shard_start=yubari_shard_start,
                shard_end=yubari_shard_end,
            )
        else:
            records.extend(
                _load_yubari_video_root_records(
                    media_reader=media_reader,
                    video_url=yubari_video_tar_url,
                    shard_start=yubari_shard_start,
                    shard_end=yubari_shard_end,
                    max_records=max_yubari_records,
                )
            )
    image_urls = []
    if not image_metadata_url:
        image_urls = media_reader.discover_urls(image_internal_url, image_suffixes)
    return FlashVSRSourceIndexV2(
        records=records,
        image_urls=image_urls,
        takano_parquet_urls=takano_parquet_urls,
        image_parquet_urls=image_parquet_urls,
        yubari_video_root_pairs=yubari_video_root_pairs,
    )


def _discover_yubari_video_root_pairs(
    media_reader: FlashVSRMediaReaderV2,
    video_url: str,
    shard_start: Optional[int] = None,
    shard_end: Optional[int] = None,
) -> List[Dict[str, str]]:
    parquet_urls = _discover_yubari_video_root_urls(
        media_reader,
        video_url,
        (".parquet",),
        shard_start=shard_start,
        shard_end=shard_end,
        max_files=None,
    )
    tar_urls = _discover_yubari_video_root_urls(
        media_reader,
        video_url,
        (".tar",),
        shard_start=shard_start,
        shard_end=shard_end,
        max_files=None,
    )
    if any(os.path.basename(url).startswith("part-") for url in parquet_urls):
        parquet_urls = [url for url in parquet_urls if os.path.basename(url).startswith("part-")]
    if any(os.path.basename(url).startswith("part-") for url in tar_urls):
        tar_urls = [url for url in tar_urls if os.path.basename(url).startswith("part-")]
    tar_by_stem: Dict[str, str] = {
        os.path.splitext(os.path.basename(url))[0]: normalize_remote_url(url) for url in tar_urls
    }
    pairs: List[Dict[str, str]] = []
    for parquet_url in parquet_urls:
        stem = os.path.splitext(os.path.basename(parquet_url))[0]
        video_tar_url = tar_by_stem.get(stem)
        if video_tar_url is None:
            continue
        pairs.append(
            {
                "parquet_url": normalize_remote_url(parquet_url),
                "video_tar_url": normalize_remote_url(video_tar_url),
            }
        )
    return pairs


def _load_yubari_video_root_records(
    media_reader: FlashVSRMediaReaderV2,
    video_url: str,
    shard_start: Optional[int] = None,
    shard_end: Optional[int] = None,
    max_records: Optional[int] = None,
) -> List[FlashVSRParquetRecord]:
    max_files = None
    if max_records is not None:
        max_files = max(1, (max_records + 199) // 200)
    parquet_urls = _discover_yubari_video_root_urls(
        media_reader,
        video_url,
        (".parquet",),
        shard_start=shard_start,
        shard_end=shard_end,
        max_files=max_files,
    )
    tar_urls = _discover_yubari_video_root_urls(
        media_reader,
        video_url,
        (".tar",),
        shard_start=shard_start,
        shard_end=shard_end,
        max_files=max_files,
    )
    if any(os.path.basename(url).startswith("part-") for url in parquet_urls):
        parquet_urls = [url for url in parquet_urls if os.path.basename(url).startswith("part-")]
    if any(os.path.basename(url).startswith("part-") for url in tar_urls):
        tar_urls = [url for url in tar_urls if os.path.basename(url).startswith("part-")]
    tar_by_stem: Dict[str, str] = {
        os.path.splitext(os.path.basename(url))[0]: normalize_remote_url(url) for url in tar_urls
    }

    records: List[FlashVSRParquetRecord] = []
    for parquet_url in parquet_urls:
        stem = os.path.splitext(os.path.basename(parquet_url))[0]
        video_tar_url = tar_by_stem.get(stem)
        if video_tar_url is None:
            continue
        frame = _read_parquet_frame(parquet_url)
        if "file_name" not in frame.columns:
            continue
        for row in frame.itertuples(index=False):
            file_name = getattr(row, "file_name")
            if not str(file_name).endswith(".mp4"):
                continue
            sample_id = os.path.basename(str(file_name))
            sample_key = os.path.splitext(sample_id)[0]
            data_offset = getattr(row, "data_offset", None)
            data_size = getattr(row, "data_size", None)
            records.append(
                FlashVSRParquetRecord(
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
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def _discover_yubari_video_root_urls(
    media_reader: FlashVSRMediaReaderV2,
    video_url: str,
    suffixes: Sequence[str],
    shard_start: Optional[int] = None,
    shard_end: Optional[int] = None,
    max_files: Optional[int] = None,
) -> List[str]:
    normalized = normalize_remote_url(video_url)
    if normalized.endswith(tuple(suffixes)):
        return [normalized]
    if shard_start is not None and shard_end is not None:
        urls = [
            normalize_remote_url(
                normalized.rstrip("/") + f"/part-{shard_id:06d}{suffixes[0]}"
            )
            for shard_id in range(int(shard_start), int(shard_end) + 1)
        ]
        if max_files is not None:
            urls = urls[:max_files]
        return urls
    if normalized.startswith("conductor://") or normalized.startswith("s3://"):
        try:
            line_limit = max(8, max_files * 4 + 4) if max_files is not None else None
            urls: List[str] = [
                normalize_remote_url(url)
                for url in list_remote_files_with_suffixes(normalized, suffixes, recursive=False, line_limit=line_limit)
            ]
            if urls:
                urls = sorted(set(urls))
                if max_files is not None:
                    urls = urls[:max_files]
                return urls
        except Exception:
            try:
                urls = [normalize_remote_url(url) for url in list_remote_files_with_suffixes(normalized, suffixes)]
                if urls:
                    urls = sorted(set(urls))
                    if max_files is not None:
                        urls = urls[:max_files]
                    return urls
            except Exception:
                pass
    return media_reader.discover_urls(video_url, suffixes)


def _load_yubari_tar_records(
    media_reader: FlashVSRMediaReaderV2,
    video_tar_url: str,
    sidecar_tar_url: Optional[str] = None,
    max_records: Optional[int] = None,
) -> List[FlashVSRParquetRecord]:
    video_tar_urls = media_reader.discover_urls(video_tar_url, (".tar",))
    sidecar_tar_urls = media_reader.discover_urls(sidecar_tar_url, (".tar",)) if sidecar_tar_url else []
    sidecar_tar_by_basename: Dict[str, str] = {
        os.path.basename(url): normalize_remote_url(url) for url in sidecar_tar_urls
    }

    records: List[FlashVSRParquetRecord] = []
    for one_video_tar_url in video_tar_urls:
        video_tar_basename = os.path.basename(one_video_tar_url)
        metadata_tar = sidecar_tar_by_basename.get(video_tar_basename)
        if metadata_tar is None:
            records.append(
                FlashVSRParquetRecord(
                    dataset_source="yubari",
                    sample_id=video_tar_basename,
                    media_path=normalize_remote_url(one_video_tar_url),
                    tar_member_path=None,
                    caption_text=None,
                    metadata={
                        "sample_key": os.path.splitext(video_tar_basename)[0],
                        "video_tar_url": normalize_remote_url(one_video_tar_url),
                        "yubari_tar_shard_only": True,
                    },
                )
            )
            if max_records is not None and len(records) >= max_records:
                return records
            continue
        for member_name in media_reader.iter_tar_members(one_video_tar_url, suffixes=(".mp4",)):
            sample_id = os.path.basename(member_name)
            sample_key = os.path.splitext(sample_id)[0]
            metadata = {
                "sample_key": sample_key,
                "video_tar_url": normalize_remote_url(one_video_tar_url),
                "video_file_name": member_name,
            }
            if metadata_tar:
                metadata.update(
                    {
                        "metadata_tar_url": metadata_tar,
                        "metadata_file_name": f"{sample_key}.json",
                    }
                )
            records.append(
                FlashVSRParquetRecord(
                    dataset_source="yubari",
                    sample_id=sample_id,
                    media_path=normalize_remote_url(one_video_tar_url),
                    tar_member_path=member_name,
                    caption_text=None,
                    metadata=metadata,
                )
            )
            if max_records is not None and len(records) >= max_records:
                return records
    return records
