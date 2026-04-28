import argparse
import json
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None

from wanvideo.data.flashvsr.datasets.parquet_index import _discover_parquet_urls, _read_parquet_frame, normalize_remote_url


def _count_rows_from_metadata(url: str) -> int:
    if pq is None:
        frame = _read_parquet_frame(url)
        return int(len(frame))

    normalized_url = normalize_remote_url(url)
    if fsspec is not None:
        try:
            with fsspec.open(normalized_url, "rb").open() as file:
                return int(pq.ParquetFile(file).metadata.num_rows)
        except Exception:
            pass
    if parabolt is not None and hasattr(parabolt, "io") and hasattr(parabolt.io, "open"):
        try:
            with parabolt.io.open(normalized_url, "rb") as file:
                return int(pq.ParquetFile(file).metadata.num_rows)
        except Exception:
            pass
    parquet_file = Path(normalized_url)
    if parquet_file.exists():
        return int(pq.ParquetFile(parquet_file).metadata.num_rows)
    frame = _read_parquet_frame(url)
    return int(len(frame))


def count_rows(metadata_url: str, max_files: int | None = None) -> dict:
    parquet_urls = _discover_parquet_urls(metadata_url)
    if max_files is not None:
        parquet_urls = parquet_urls[: max(0, int(max_files))]

    total_rows = 0
    details = []
    for url in parquet_urls:
        row_count = _count_rows_from_metadata(url)
        total_rows += int(row_count)
        details.append({"url": url, "rows": int(row_count)})

    return {
        "metadata_url": metadata_url,
        "num_parquet_files": len(parquet_urls),
        "total_rows": total_rows,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_url", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    result = count_rows(args.metadata_url, max_files=args.max_files)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
