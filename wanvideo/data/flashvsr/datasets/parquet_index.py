import hashlib
import io
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None

from .conductor_bridge_v2 import (
    get_conductor_local_path,
    has_ref_big_conductor,
    list_remote_files_with_suffixes,
    read_conductor_bytes,
)


@dataclass
class FlashVSRParquetRecord:
    dataset_source: str
    sample_id: str
    media_path: str
    tar_member_path: Optional[str]
    caption_text: Optional[str]
    metadata: Dict[str, Any]


def normalize_remote_url(url: Optional[str]) -> Optional[str]:
    if url is None:
        return None
    if url.startswith("s3://"):
        return "conductor://" + url[len("s3://") :]
    return url


REMOTE_DISCOVERY_PREFIXES = ("conductor://", "s3://", "blobby://")


def _expand_input_urls(base_url: Optional[str]) -> List[str]:
    if base_url is None:
        return []
    candidates = [item.strip() for item in str(base_url).replace("\n", ",").split(",")]
    return [candidate for candidate in candidates if candidate]


def _discover_parquet_urls(base_url: Optional[str]) -> List[str]:
    base_urls = [normalize_remote_url(url) for url in _expand_input_urls(base_url)]
    if not base_urls:
        return []

    merged_urls: List[str] = []
    for one_base_url in base_urls:
        if str(one_base_url).endswith(".parquet"):
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
            urls = [normalize_remote_url(url) for url in list_remote_files_with_suffixes(one_base_url, (".parquet",))]
        if not urls:
            if one_base_url.startswith(REMOTE_DISCOVERY_PREFIXES):
                raise RuntimeError(f"failed to discover parquet urls for {one_base_url} via unified conductor bridge")
            urls = [one_base_url]
        merged_urls.extend(url for url in urls if str(url).endswith(".parquet"))
    return sorted(set(merged_urls))


def _open_binary(url: str) -> bytes:
    url = normalize_remote_url(url)
    if str(url).startswith("conductor://"):
        if not has_ref_big_conductor():
            raise RuntimeError(f"ref_big conductor client unavailable for {url}")
        return read_conductor_bytes(url)
    if fsspec is not None:
        try:
            with fsspec.open(url, "rb").open() as file:
                return file.read()
        except Exception:
            pass
    if parabolt is not None and hasattr(parabolt, "io") and hasattr(parabolt.io, "open"):
        try:
            with parabolt.io.open(url, "rb") as file:
                return file.read()
        except Exception:
            pass
    if os.path.exists(url):
        with open(url, "rb") as file:
            return file.read()
    raise FileNotFoundError(f"Unable to open parquet url: {url}")


def _resolve_cached_path(cache_dir: str, source_url: str) -> str:
    digest = hashlib.sha1(source_url.encode("utf-8")).hexdigest()
    base_name = os.path.basename(source_url) or "cached.parquet"
    return os.path.join(cache_dir, f"{digest}_{base_name}")


def _open_parquet_for_arrow(parquet_url: str):
    normalized_url = normalize_remote_url(parquet_url)
    if normalized_url.startswith("conductor://"):
        if not has_ref_big_conductor():
            raise RuntimeError("ref_big conductor client unavailable for parquet access")
        local_path = get_conductor_local_path(normalized_url)
        if local_path:
            return pq.ParquetFile(local_path)
        parquet_bytes = read_conductor_bytes(normalized_url)
        return pq.ParquetFile(pa.BufferReader(parquet_bytes))
    parquet_bytes = _open_binary(normalized_url)
    return pq.ParquetFile(pa.BufferReader(parquet_bytes))


@lru_cache(maxsize=64)
def _read_parquet_frame(parquet_url: str):
    normalized_url = normalize_remote_url(parquet_url)
    if parquet_url.startswith("s3://") or normalized_url.startswith("conductor://"):
        if normalized_url.startswith("conductor://") and not has_ref_big_conductor():
            raise RuntimeError("ref_big conductor client unavailable for parquet access")
        if normalized_url.startswith("conductor://") and has_ref_big_conductor():
            local_path = get_conductor_local_path(normalized_url)
            if local_path:
                return pd.read_parquet(local_path)
            parquet_bytes = read_conductor_bytes(normalized_url)
            return pd.read_parquet(io.BytesIO(parquet_bytes))
        parquet_bytes = _open_binary(normalized_url)
        return pd.read_parquet(io.BytesIO(parquet_bytes))
    try:
        parquet_bytes = _open_binary(normalized_url)
        return pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception:
        pass

    return pd.read_parquet(normalized_url)


def iter_parquet_row_dicts_from_url(
    parquet_url: str,
    columns: Optional[List[str]] = None,
    batch_size: int = 4096,
) -> Iterable[Dict[str, Any]]:
    if pq is None or pa is None:
        frame = _read_parquet_frame(parquet_url)
        for row in frame.to_dict(orient="records"):
            yield row
        return

    parquet_file = _open_parquet_for_arrow(parquet_url)
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        payload = table.to_pydict()
        if not payload:
            continue
        keys = list(payload.keys())
        if not keys:
            continue
        length = len(payload[keys[0]])
        for index in range(length):
            yield {key: payload[key][index] for key in keys}


def _normalize_source(row: Dict[str, Any], dataset_source: str) -> Optional[str]:
    if dataset_source != "auto":
        return dataset_source
    if "video_path" in row and "HIGH_RES_TARGET_S3_PATH" in row:
        return "takano"
    if "path_lucid" in row and "overall_score" in row:
        return "storymotion"
    return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if text.lower() == "nan":
        return None
    return text


def _parse_takano_member_path(path_value: Any) -> Optional[str]:
    path_text = _safe_str(path_value)
    if path_text is None or ".tar/" not in path_text:
        return None
    return path_text.split(".tar/", 1)[1]


def _build_storymotion_record(row: Dict[str, Any]) -> Optional[FlashVSRParquetRecord]:
    media_path = normalize_remote_url(_safe_str(row.get("path_lucid")))
    if not media_path:
        return None
    return FlashVSRParquetRecord(
        dataset_source="storymotion",
        sample_id=os.path.basename(media_path),
        media_path=media_path,
        tar_member_path=None,
        caption_text=_safe_str(row.get("qwen35_output")),
        metadata={
            "width": row.get("width"),
            "height": row.get("height"),
            "duration": row.get("duration"),
            "frame_rate": row.get("frame_rate"),
            "orientation": _safe_str(row.get("orientation")),
            "overall_score": row.get("overall_score"),
            "semantic_score": row.get("semantic_score"),
            "technical_score": row.get("technical_score"),
            "overlay_ratio": row.get("overlay_ratio"),
            "qwen35_parse_success": row.get("qwen35_parse_success"),
        },
    )


def _build_takano_record(row: Dict[str, Any]) -> Optional[FlashVSRParquetRecord]:
    shard_path = normalize_remote_url(_safe_str(row.get("path_lucid")))
    raw_path = _safe_str(row.get("path"))
    member_path = _parse_takano_member_path(raw_path)
    direct_clip_path = normalize_remote_url(raw_path)

    # New Takano parquet roots point directly at clip mp4 objects.
    if not shard_path and direct_clip_path and direct_clip_path.endswith(".mp4"):
        return FlashVSRParquetRecord(
            dataset_source="takano",
            sample_id=os.path.basename(direct_clip_path),
            media_path=direct_clip_path,
            tar_member_path=None,
            caption_text=_safe_str(row.get("qwen35_output")),
            metadata={
                "max_width": row.get("MAX_WIDTH"),
                "max_height": row.get("MAX_HEIGHT"),
                "video_path": _safe_str(row.get("video_path")),
                "org_path": _safe_str(row.get("org_path")),
                "metadata_path": _safe_str(row.get("metadata_path")),
                "high_res_target_s3_path": _safe_str(row.get("HIGH_RES_TARGET_S3_PATH")),
                "qwen35_parse_success": row.get("qwen35_parse_success"),
                "resolution": _safe_str(row.get("resolution")),
                "frame_rate": row.get("frame_rate"),
                "duration": row.get("duration"),
                "width": row.get("width"),
                "height": row.get("height"),
                "source": _safe_str(row.get("source")),
            },
        )

    if not shard_path or not member_path:
        return None
    return FlashVSRParquetRecord(
        dataset_source="takano",
        sample_id=os.path.basename(member_path),
        media_path=shard_path,
        tar_member_path=member_path,
        caption_text=_safe_str(row.get("qwen35_output")),
        metadata={
            "max_width": row.get("MAX_WIDTH"),
            "max_height": row.get("MAX_HEIGHT"),
            "video_path": _safe_str(row.get("video_path")),
            "metadata_path": _safe_str(row.get("metadata_path")),
            "high_res_target_s3_path": _safe_str(row.get("HIGH_RES_TARGET_S3_PATH")),
            "qwen35_parse_success": row.get("qwen35_parse_success"),
        },
    )


def _build_yubari_record(
    video_row: Dict[str, Any],
    metadata_row: Optional[Dict[str, Any]],
    video_tar_url: str,
    metadata_tar_url: Optional[str],
) -> Optional[FlashVSRParquetRecord]:
    file_name = _safe_str(video_row.get("file_name"))
    if not file_name:
        return None
    sample_id = os.path.basename(file_name)
    sample_key = os.path.splitext(sample_id)[0]
    metadata_json_name = None
    if metadata_row is not None:
        metadata_json_name = _safe_str(metadata_row.get("file_name"))
    if metadata_json_name is None:
        metadata_json_name = f"{sample_key}.json"
    return FlashVSRParquetRecord(
        dataset_source="yubari",
        sample_id=sample_id,
        media_path=normalize_remote_url(video_tar_url),
        tar_member_path=file_name,
        caption_text=None,
        metadata={
            "sample_key": sample_key,
            "video_tar_url": normalize_remote_url(video_tar_url),
            "video_file_name": file_name,
            "video_header_offset": video_row.get("header_offset"),
            "video_data_offset": video_row.get("data_offset"),
            "video_data_size": video_row.get("data_size"),
            "video_flags": video_row.get("flags"),
            "metadata_tar_url": normalize_remote_url(metadata_tar_url) if metadata_tar_url else None,
            "metadata_file_name": metadata_json_name,
            "metadata_header_offset": None if metadata_row is None else metadata_row.get("header_offset"),
            "metadata_data_offset": None if metadata_row is None else metadata_row.get("data_offset"),
            "metadata_data_size": None if metadata_row is None else metadata_row.get("data_size"),
            "metadata_flags": None if metadata_row is None else metadata_row.get("flags"),
        },
    )


def _build_image_record(row: Dict[str, Any]) -> Optional[FlashVSRParquetRecord]:
    target_s3_path = _safe_str(row.get("TARGET_S3_PATH"))
    if not target_s3_path:
        return None
    media_path = normalize_remote_url(target_s3_path)
    sample_id = _safe_str(row.get("ASSET_NAME")) or os.path.basename(media_path)
    caption = (
        _safe_str(row.get("DESCRIPTION"))
        or _safe_str(row.get("JAPANESE_DESCRIPTION"))
        or _safe_str(row.get("KOREAN_DESCRIPTION"))
        or _safe_str(row.get("FRENCH_DESCRIPTION"))
        or _safe_str(row.get("SPANISH_DESCRIPTION"))
        or _safe_str(row.get("GERMAN_DESCRIPTION"))
    )
    return FlashVSRParquetRecord(
        dataset_source="image",
        sample_id=sample_id,
        media_path=media_path,
        tar_member_path=None,
        caption_text=caption,
        metadata={
            "asset_id": _safe_str(row.get("ASSET_ID")),
            "catalog": _safe_str(row.get("CATALOG")),
            "batch_suffix": _safe_str(row.get("BATCH_SUFFIX")),
            "categories": _safe_str(row.get("CATEGORIES")),
            "subcategories": _safe_str(row.get("SUBCATEGORIES")),
            "keywords": _safe_str(row.get("KEYWORDS")),
            "location": _safe_str(row.get("LOCATION")),
            "type": _safe_str(row.get("TYPE")),
            "md5": _safe_str(row.get("MD5")),
            "target_s3_path": target_s3_path,
            "max_width": row.get("MAX_WIDTH"),
            "max_height": row.get("MAX_HEIGHT"),
            "size": row.get("SIZE"),
            "has_people": row.get("HAS_PEOPLE"),
            "full_biometric_consent": row.get("FULL_BIOMETRIC_CONSENT"),
            "is_adult": row.get("IS_ADULT"),
        },
    )


def load_yubari_records(
    video_metadata_url: str,
    sidecar_metadata_url: str,
    max_records: Optional[int] = None,
) -> List[FlashVSRParquetRecord]:
    if pd is None:
        raise ImportError("pandas is required for parquet-driven FlashVSR dataset indexing.")

    video_parquet_urls = _discover_parquet_urls(video_metadata_url)
    sidecar_parquet_urls = _discover_parquet_urls(sidecar_metadata_url)
    if not video_parquet_urls:
        raise ValueError(f"No video parquet files found under video_metadata_url={video_metadata_url}")
    if not sidecar_parquet_urls:
        raise ValueError(f"No metadata parquet files found under sidecar_metadata_url={sidecar_metadata_url}")

    sidecar_rows_by_key: Dict[str, Dict[str, Any]] = {}
    sidecar_tar_by_key: Dict[str, str] = {}
    for parquet_url in sidecar_parquet_urls:
        frame = _read_parquet_frame(parquet_url)
        tar_url = normalize_remote_url(parquet_url[:-len(".parquet")] + ".tar")
        for row in frame.to_dict(orient="records"):
            file_name = _safe_str(row.get("file_name"))
            if not file_name:
                continue
            sidecar_rows_by_key[os.path.splitext(os.path.basename(file_name))[0]] = row
            sidecar_tar_by_key[os.path.splitext(os.path.basename(file_name))[0]] = tar_url

    records: List[FlashVSRParquetRecord] = []
    for parquet_url in video_parquet_urls:
        frame = _read_parquet_frame(parquet_url)
        tar_url = normalize_remote_url(parquet_url[:-len(".parquet")] + ".tar")
        for row in frame.to_dict(orient="records"):
            file_name = _safe_str(row.get("file_name"))
            if not file_name:
                continue
            sample_key = os.path.splitext(os.path.basename(file_name))[0]
            record = _build_yubari_record(
                video_row=row,
                metadata_row=sidecar_rows_by_key.get(sample_key),
                video_tar_url=tar_url,
                metadata_tar_url=sidecar_tar_by_key.get(sample_key),
            )
            if record is None:
                continue
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def load_image_records(
    metadata_url: str,
    max_records: Optional[int] = None,
) -> List[FlashVSRParquetRecord]:
    if pd is None:
        raise ImportError("pandas is required for parquet-driven FlashVSR dataset indexing.")

    parquet_urls = _discover_parquet_urls(metadata_url)
    if not parquet_urls:
        raise ValueError(f"No image parquet files found under metadata_url={metadata_url}")

    records: List[FlashVSRParquetRecord] = []
    for parquet_url in parquet_urls:
        frame = _read_parquet_frame(parquet_url)
        for row in frame.to_dict(orient="records"):
            record = _build_image_record(row)
            if record is None:
                continue
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def load_parquet_records(
    metadata_url: Optional[str],
    dataset_source: str = "auto",
    max_records: Optional[int] = None,
    min_overall_score: Optional[float] = None,
    require_qwen35_parse_success: bool = False,
) -> List[FlashVSRParquetRecord]:
    if pd is None:
        raise ImportError("pandas is required for parquet-driven FlashVSR dataset indexing.")

    parquet_urls = _discover_parquet_urls(metadata_url)
    if not parquet_urls:
        raise ValueError(f"No parquet files found under metadata_url={metadata_url}")

    records: List[FlashVSRParquetRecord] = []
    for parquet_url in parquet_urls:
        frame = _read_parquet_frame(parquet_url)
        for row in frame.to_dict(orient="records"):
            source = _normalize_source(row, dataset_source)
            if source is None:
                continue
            if require_qwen35_parse_success and not bool(row.get("qwen35_parse_success", False)):
                continue
            if source == "storymotion" and min_overall_score is not None:
                score = row.get("overall_score")
                if score is None or float(score) < min_overall_score:
                    continue
            if source == "storymotion":
                record = _build_storymotion_record(row)
            elif source == "takano":
                record = _build_takano_record(row)
            else:
                continue
            if record is None:
                continue
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def iter_parquet_records_from_frame(
    frame,
    dataset_source: str = "auto",
    min_overall_score: Optional[float] = None,
    require_qwen35_parse_success: bool = False,
) -> Iterable[FlashVSRParquetRecord]:
    for row in frame.to_dict(orient="records"):
        source = _normalize_source(row, dataset_source)
        if source is None:
            continue
        if require_qwen35_parse_success and not bool(row.get("qwen35_parse_success", False)):
            continue
        if source == "storymotion" and min_overall_score is not None:
            score = row.get("overall_score")
            if score is None or float(score) < min_overall_score:
                continue
        if source == "storymotion":
            record = _build_storymotion_record(row)
        elif source == "takano":
            record = _build_takano_record(row)
        else:
            continue
        if record is not None:
            yield record


def load_parquet_records_from_url(
    parquet_url: str,
    dataset_source: str = "auto",
    max_records: Optional[int] = None,
    min_overall_score: Optional[float] = None,
    require_qwen35_parse_success: bool = False,
) -> List[FlashVSRParquetRecord]:
    frame = _read_parquet_frame(parquet_url)
    records: List[FlashVSRParquetRecord] = []
    for record in iter_parquet_records_from_frame(
        frame,
        dataset_source=dataset_source,
        min_overall_score=min_overall_score,
        require_qwen35_parse_success=require_qwen35_parse_success,
    ):
        records.append(record)
        if max_records is not None and len(records) >= max_records:
            break
    return records
