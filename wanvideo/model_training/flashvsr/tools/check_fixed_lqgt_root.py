#!/usr/bin/env python3
"""Checksum a fixed LQ/GT overfit root for Stage3 DMD debugging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    value_f = value.float()
    # bfloat16 cannot be converted directly to numpy on this torch build.
    digest = hashlib.sha256(value_f.numpy().tobytes()).hexdigest()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": digest,
        "min": float(value_f.min()),
        "max": float(value_f.max()),
        "mean": float(value_f.mean()),
        "std": float(value_f.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    sample_paths = sorted(root.glob("sample_*.pt"))
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*.pt under {root}")

    rows: list[dict[str, Any]] = []
    for path in sample_paths:
        payload = torch.load(path, map_location="cpu")
        video = payload.get("video")
        lq_video = payload.get("lq_video")
        if not (torch.is_tensor(video) and torch.is_tensor(lq_video)):
            raise TypeError(f"{path} does not contain tensor video/lq_video")
        rows.append(
            {
                "file": str(path),
                "sample_id": payload.get("sample_id", path.stem),
                "sample_seed": int(payload.get("sample_seed", -1)),
                "video": _tensor_summary(video),
                "lq_video": _tensor_summary(lq_video),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump({"root": str(root), "num_samples": len(rows), "samples": rows}, file, ensure_ascii=False, indent=2)
    print(f"[fixed_lqgt_checksum] wrote {output} samples={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
