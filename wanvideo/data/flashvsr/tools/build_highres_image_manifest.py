import argparse
from pathlib import Path

from wanvideo.data.flashvsr.datasets.parquet_index import _discover_parquet_urls, _read_parquet_frame


PREFERRED_PATH_COLUMNS = (
    "TARGET_S3_PATH",
    "path",
    "media_url",
    "url",
)


def _pick_path_column(columns):
    for column in PREFERRED_PATH_COLUMNS:
        if column in columns:
            return column
    raise ValueError(f"Could not find any image path column in parquet columns: {list(columns)}")


def build_manifest(metadata_url: str, output_path: str, max_files: int | None = None) -> dict:
    parquet_urls = _discover_parquet_urls(metadata_url)
    if max_files is not None:
        parquet_urls = parquet_urls[: max(0, int(max_files))]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    seen = set()
    path_column = None

    with output.open("w", encoding="utf-8") as fout:
        for parquet_url in parquet_urls:
            frame = _read_parquet_frame(parquet_url)
            if path_column is None:
                path_column = _pick_path_column(frame.columns)
            for value in frame[path_column].tolist():
                total_rows += 1
                if value is None:
                    continue
                path = str(value).strip()
                if not path:
                    continue
                if path in seen:
                    continue
                seen.add(path)
                fout.write(path)
                fout.write("\n")
                kept_rows += 1

    return {
        "metadata_url": metadata_url,
        "output_path": str(output),
        "num_parquet_files": len(parquet_urls),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "path_column": path_column,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_url", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    result = build_manifest(
        metadata_url=args.metadata_url,
        output_path=args.output_path,
        max_files=args.max_files,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
