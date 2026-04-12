import argparse
import json
from pathlib import Path
from typing import Optional

from wanvideo.data.flashvsr.datasets.parquet_index import load_parquet_records


def build_manifest(metadata_url: str, output_path: str, max_records: Optional[int] = None) -> int:
    records = load_parquet_records(
        metadata_url=metadata_url,
        dataset_source="storymotion",
        max_records=max_records,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            fps = record.metadata.get("frame_rate")
            duration = record.metadata.get("duration")
            num_frames_est = None
            if fps is not None and duration is not None:
                try:
                    num_frames_est = int(round(float(fps) * float(duration)))
                except Exception:
                    num_frames_est = None
            payload = {
                "media_url": record.media_path,
                "caption": record.caption_text,
                "width": record.metadata.get("width"),
                "height": record.metadata.get("height"),
                "fps": fps,
                "duration": duration,
                "num_frames_est": num_frames_est,
                "source_dataset": record.dataset_source,
                "sample_id": record.sample_id,
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_url", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_records", type=int, default=None)
    args = parser.parse_args()

    count = build_manifest(args.metadata_url, args.output_path, args.max_records)
    print(f"manifest 已生成: {args.output_path}")
    print(f"总条目数: {count}")


if __name__ == "__main__":
    main()
