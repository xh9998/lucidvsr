import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None


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

        if not urls:
            urls = [one_base_url]
        merged_urls.extend(url for url in urls if str(url).endswith(".parquet"))
    return sorted(set(merged_urls))


def _open_binary(url: str) -> bytes:
    url = normalize_remote_url(url)
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


def _read_parquet_frame(parquet_url: str):
    normalized_url = normalize_remote_url(parquet_url)
    try:
        parquet_bytes = _open_binary(normalized_url)
        return pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception:
        pass

    if parquet_url.startswith("s3://") or normalized_url.startswith("conductor://"):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            temp_path = tmp.name
        try:
            source_url = parquet_url
            if normalized_url.startswith("conductor://"):
                source_url = "s3://" + normalized_url[len("conductor://") :]
            subprocess.check_call(["conductor", "s3", "cp", source_url, temp_path])
            return pd.read_parquet(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return pd.read_parquet(normalized_url)


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
    member_path = _parse_takano_member_path(row.get("path"))
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
            else:
                record = _build_takano_record(row)
            if record is None:
                continue
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                return records
    return records
