import argparse
from pathlib import Path

from wanvideo.data.flashvsr.datasets.conductor_bridge_v2 import list_remote_files_with_suffixes
from wanvideo.data.flashvsr.datasets.parquet_index import normalize_remote_url


def build_manifest(root_url: str, output_path: str, recursive: bool = True, limit: int | None = None) -> dict:
    urls = [
        normalize_remote_url(url)
        for url in list_remote_files_with_suffixes(root_url, (".tar",), recursive=recursive, line_limit=limit)
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for url in urls:
            fout.write(url)
            fout.write("\n")

    return {
        "root_url": normalize_remote_url(root_url),
        "output_path": str(output),
        "recursive": bool(recursive),
        "num_tar_files": len(urls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_url", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--non_recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = build_manifest(
        root_url=args.root_url,
        output_path=args.output_path,
        recursive=not args.non_recursive,
        limit=args.limit,
    )
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
