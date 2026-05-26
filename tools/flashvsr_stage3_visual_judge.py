#!/usr/bin/env python3
"""Create visual montages and simple video-quality flags for Stage3 debug outputs.

Expected input layout:
  ROOT/CASE/gt/*.mp4
  ROOT/CASE/lq/*.mp4
  ROOT/CASE/sr/*.mp4

The script is intentionally standalone so DMD debug probes do not modify training code.
"""

from __future__ import annotations

import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_frame(video: Path, frame_index: int | None) -> np.ndarray | None:
    frame_count = probe_frame_count(video)
    if frame_count <= 0:
        return None
    if frame_index is None:
        frame_index = frame_count // 2
    frame_index = max(0, min(frame_count - 1, frame_index))
    try:
        return np.asarray(iio.imread(video, index=frame_index, plugin="pyav"))[:, :, :3]
    except Exception:
        try:
            for idx, frame in enumerate(iio.imiter(video)):
                if idx == frame_index:
                    return np.asarray(frame)[:, :, :3]
        except Exception:
            return None
    return None


@lru_cache(maxsize=512)
def probe_frame_count(video: Path) -> int:
    try:
        props = iio.improps(video)
    except Exception:
        return 0
    try:
        n_images = int(props.n_images or 0)
        if n_images > 0:
            return n_images
    except OverflowError:
        pass
    try:
        return sum(1 for _ in iio.imiter(video))
    except Exception:
        return 0


def sample_frames(video: Path) -> dict[str, np.ndarray]:
    frame_count = probe_frame_count(video)
    indices = {
        "first": 0,
        "mid": frame_count // 2 if frame_count > 0 else 0,
        "last": max(0, frame_count - 1),
    }
    out = {}
    for key, idx in indices.items():
        frame = read_frame(video, idx)
        if frame is not None:
            out[key] = frame
    return out


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = max(1, int(round(w * height / h)))
    return np.asarray(Image.fromarray(img).resize((width, height), Image.Resampling.LANCZOS))


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 34), fill=(0, 0, 0))
    draw.text((8, 9), text, fill=(255, 255, 255), font=ImageFont.load_default())
    return np.asarray(pil)


def metrics(img: np.ndarray, ref: np.ndarray | None = None) -> dict[str, float]:
    x = img.astype(np.float32) / 255.0
    mean_rgb = x.reshape(-1, 3).mean(axis=0)
    std_rgb = x.reshape(-1, 3).std(axis=0)
    gray = (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2])
    gy, gx = np.gradient(gray)
    blur = float((gx * gx + gy * gy).mean() * 1_000_000.0)
    out = {
        "brightness": float(gray.mean()),
        "r_mean": float(mean_rgb[0]),
        "g_mean": float(mean_rgb[1]),
        "b_mean": float(mean_rgb[2]),
        "r_std": float(std_rgb[0]),
        "g_std": float(std_rgb[1]),
        "b_std": float(std_rgb[2]),
        "rgb_cast": float(mean_rgb.max() - mean_rgb.min()),
        "blur_lap_var": blur,
    }
    if ref is not None:
        ref_r = np.asarray(Image.fromarray(ref).resize((img.shape[1], img.shape[0]), Image.Resampling.LANCZOS))
        diff = (x - ref_r.astype(np.float32) / 255.0) ** 2
        out["mse_to_gt"] = float(diff.mean())
    else:
        out["mse_to_gt"] = math.nan
    return out


def flags(m: dict[str, float]) -> list[str]:
    out: list[str] = []
    if m["brightness"] < 0.08:
        out.append("black/dark")
    if m["brightness"] > 0.92:
        out.append("over-bright")
    if m["rgb_cast"] > 0.12:
        if m["g_mean"] > m["r_mean"] + 0.08 and m["g_mean"] > m["b_mean"] + 0.08:
            out.append("green-cast")
        elif m["r_mean"] > m["b_mean"] + 0.08 and m["g_mean"] > m["b_mean"] + 0.08:
            out.append("yellow/warm-cast")
        else:
            out.append("color-cast")
    if max(m["r_std"], m["g_std"], m["b_std"]) < 0.045:
        out.append("gray/low-contrast")
    if m["blur_lap_var"] < 20.0:
        out.append("very-blurry")
    if not out:
        out.append("ok-by-proxy")
    return out


def discover_cases(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "sr").is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    montage_tiles: list[np.ndarray] = []

    for case_dir in discover_cases(args.root):
        case = case_dir.name
        gt_files = sorted((case_dir / "gt").glob("*.mp4"))
        lq_files = sorted((case_dir / "lq").glob("*.mp4"))
        sr_files = sorted((case_dir / "sr").glob("*.mp4"))
        gt = gt_files[0] if gt_files else None
        lq = lq_files[0] if lq_files else None
        gt_first = read_frame(gt, 0) if gt else None

        for video in sr_files:
            step = video.stem
            frames = sample_frames(video)
            for frame_name, frame in frames.items():
                met = metrics(frame, gt_first if frame_name == "first" else None)
                fl = ",".join(flags(met))
                rows.append({"case": case, "video": video.name, "frame": frame_name, "flags": fl, **met})
            first = frames.get("first")
            if first is not None:
                tiles = []
                if gt is not None:
                    gt_img = read_frame(gt, 0)
                    if gt_img is not None:
                        tiles.append(put_label(resize_to_height(gt_img, args.height), f"{case} GT"))
                if lq is not None:
                    lq_img = read_frame(lq, 0)
                    if lq_img is not None:
                        tiles.append(put_label(resize_to_height(lq_img, args.height), f"{case} LQ"))
                tiles.append(put_label(resize_to_height(first, args.height), step))
                montage_tiles.append(np.concatenate(tiles, axis=1))

    if montage_tiles:
        width = max(t.shape[1] for t in montage_tiles)
        padded = []
        for tile in montage_tiles:
            if tile.shape[1] < width:
                pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), dtype=tile.dtype)
                tile = np.concatenate([tile, pad], axis=1)
            padded.append(tile)
        montage = np.concatenate(padded, axis=0)
        Image.fromarray(montage).save(args.out / "first_frame_compare.png")

    csv_path = args.out / "metrics.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_path = args.out / "visual_judge.md"
    with md_path.open("w") as f:
        f.write(f"# Visual Judge Report\n\nRoot: `{args.root}`\n\n")
        f.write("| case | video | first-frame flags | brightness | rgb_cast | blur |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: |\n")
        for row in rows:
            if row["frame"] != "first":
                continue
            f.write(
                f"| {row['case']} | {row['video']} | {row['flags']} | "
                f"{float(row['brightness']):.3f} | {float(row['rgb_cast']):.3f} | "
                f"{float(row['blur_lap_var']):.1f} |\n"
            )


if __name__ == "__main__":
    main()
