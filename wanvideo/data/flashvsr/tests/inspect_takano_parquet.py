import argparse
import io
import json
from typing import Any, Dict

import pandas as pd

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import parabolt
except ImportError:
    parabolt = None

from wanvideo.data.flashvsr.datasets.parquet_index import normalize_remote_url


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
    with open(url, "rb") as file:
        return file.read()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_url", type=str, required=True)
    parser.add_argument("--row_index", type=int, default=0)
    args = parser.parse_args()

    data = _open_binary(args.parquet_url)
    df = pd.read_parquet(io.BytesIO(data))

    print(f"Number of rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    row: Dict[str, Any] = {key: _jsonable(value) for key, value in df.iloc[args.row_index].to_dict().items()}
    print("First row:")
    print(json.dumps(row, ensure_ascii=False, indent=2, default=str))
    if "path" in row:
        print("row.path =", row["path"])
    if "path_lucid" in row:
        print("row.path_lucid =", row["path_lucid"])


if __name__ == "__main__":
    main()
