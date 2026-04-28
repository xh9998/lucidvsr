import argparse
import json
import os
from typing import Any, Dict, List


def _load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _compare_samples(lhs: List[Dict[str, Any]], rhs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    count = min(len(lhs), len(rhs))
    rows: List[Dict[str, Any]] = []
    for idx in range(count):
        rows.append(
            {
                "sample_index": idx,
                "lhs_sample_id": lhs[idx].get("sample_id"),
                "rhs_sample_id": rhs[idx].get("sample_id"),
                "same_sample_id": lhs[idx].get("sample_id") == rhs[idx].get("sample_id"),
                "same_sample_seed": lhs[idx].get("sample_seed") == rhs[idx].get("sample_seed"),
                "same_video_hash": lhs[idx].get("video_hash") == rhs[idx].get("video_hash"),
                "same_lq_video_hash": lhs[idx].get("lq_video_hash") == rhs[idx].get("lq_video_hash"),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lhs", required=True)
    parser.add_argument("--rhs", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    lhs = _load_summary(args.lhs)
    rhs = _load_summary(args.rhs)
    result = {
        "lhs": os.path.abspath(args.lhs),
        "rhs": os.path.abspath(args.rhs),
        "comparison": _compare_samples(lhs.get("samples", []), rhs.get("samples", [])),
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
