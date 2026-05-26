#!/usr/bin/env python3
"""Build Stage3 visual review reports for clean results and probe dumps.

Supported layouts:
  1. Clean results:
       ROOT/CASE/{gt,lq,sr}/*.mp4
  2. Probe dumps / decoded outputs:
       ROOT/CASE/{HR,LQ,SR,real_x0,fake_x0,student_z_pred}/*.{mp4,png,...}

The script is standalone on purpose: it must not be imported by training code
or require modifying any Stage3 entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

STREAM_ALIASES: dict[str, tuple[str, ...]] = {
    "HR": ("HR", "hr", "GT", "gt", "target", "hq"),
    "LQ": ("LQ", "lq", "LR", "lr", "input", "low"),
    "SR": ("SR", "sr", "pred", "output", "result"),
    "real_x0": ("real_x0", "real", "G_real", "greal", "teacher_real"),
    "fake_x0": ("fake_x0", "fake", "G_fake", "gfake", "teacher_fake"),
    "student_z_pred": (
        "student_z_pred",
        "student_x0",
        "z_pred",
        "pred_x0",
        "student",
        "student_pred",
    ),
}
STREAM_ORDER = ("HR", "LQ", "SR", "real_x0", "fake_x0", "student_z_pred")
ANCHOR_STREAMS = ("SR", "student_z_pred", "real_x0", "fake_x0")


@dataclass(frozen=True)
class MediaItem:
    stream: str
    path: Path
    sample_key: str


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "sample"


def _canonical_stream(name: str) -> str | None:
    for canonical, aliases in STREAM_ALIASES.items():
        if name in aliases:
            return canonical
    return None


def _iter_media_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTS)


def _stream_dirs(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.exists() or not folder.is_dir():
        return out
    for child in folder.iterdir():
        if not child.is_dir():
            continue
        canonical = _canonical_stream(child.name)
        if canonical is not None and _iter_media_files(child):
            out[canonical] = child
    return out


def discover_case_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Find directories that directly contain stream subdirectories."""
    root = root.resolve()
    if _stream_dirs(root):
        return [root]

    cases: list[Path] = []
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        try:
            rel_depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if rel_depth > max_depth:
            continue
        streams = _stream_dirs(path)
        if streams:
            cases.append(path)

    # Avoid returning both parent and nested child when both look like cases.
    filtered: list[Path] = []
    for case in cases:
        if not any(parent in filtered for parent in case.parents):
            filtered.append(case)
    return filtered


def _normalize_sample_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^(HR|GT|LQ|LR|SR|real_x0|fake_x0|student_z_pred)[_-]*", "", stem, flags=re.I)
    stem = re.sub(r"_(HR|GT|LQ|LR|SR|real_x0|fake_x0|student_z_pred)$", "", stem, flags=re.I)
    stem = re.sub(r"^(sample[_-]?\d+).*", r"\1", stem, flags=re.I)
    return stem or path.stem


def collect_media(case_dir: Path) -> dict[str, list[MediaItem]]:
    streams = _stream_dirs(case_dir)
    out: dict[str, list[MediaItem]] = {}
    for stream, folder in streams.items():
        out[stream] = [
            MediaItem(stream=stream, path=media, sample_key=_normalize_sample_key(media))
            for media in _iter_media_files(folder)
        ]
    return out


def _media_count(path: Path) -> int:
    if path.suffix.lower() in IMAGE_EXTS:
        return 1
    return probe_frame_count(path)


@lru_cache(maxsize=2048)
def probe_frame_count(path: Path) -> int:
    try:
        props = iio.improps(path)
        count = int(props.n_images or 0)
        if count > 0:
            return count
    except Exception:
        pass
    # Do not fall back to iterating the whole video. These reports are often
    # built over many 85/89-frame videos, and exact frame count is not worth
    # making the visual-review tool slow.
    return 0


def read_frame(path: Path, frame_index: int | None = 0) -> np.ndarray | None:
    if path.suffix.lower() in IMAGE_EXTS:
        try:
            return np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            return None

    frame_count = probe_frame_count(path)
    if frame_index is None:
        frame_index = frame_count // 2 if frame_count > 0 else 0
    if frame_count > 0:
        frame_index = max(0, min(frame_count - 1, int(frame_index)))
    else:
        frame_index = max(0, int(frame_index))
    try:
        frame = iio.imread(path, index=frame_index, plugin="pyav")
        return np.asarray(frame)[:, :, :3]
    except Exception:
        try:
            for idx, frame in enumerate(iio.imiter(path)):
                if idx == frame_index:
                    return np.asarray(frame)[:, :, :3]
        except Exception:
            return None
    return None


@lru_cache(maxsize=2048)
def sample_frames(path: Path, fallback_mid: int = 44, fallback_last: int = 88) -> dict[str, np.ndarray]:
    if path.suffix.lower() in IMAGE_EXTS:
        frame = read_frame(path, 0)
        return {"first": frame, "mid": frame, "last": frame} if frame is not None else {}

    count = _media_count(path)
    if count > 1:
        indices = {"first": 0, "mid": count // 2, "last": max(0, count - 1)}
    else:
        # Stage2/3 debug videos are usually 85 or 89 frames. If metadata cannot
        # provide frame count, sample representative positions without scanning
        # the full stream just to locate the last frame.
        indices = {"first": 0, "mid": fallback_mid, "last": fallback_last}
    out: dict[str, np.ndarray] = {}
    wanted = {idx: name for name, idx in indices.items()}
    max_idx = max(wanted)
    try:
        for idx, frame in enumerate(iio.imiter(path)):
            if idx in wanted:
                out[wanted[idx]] = np.asarray(frame)[:, :, :3]
            if idx >= max_idx:
                break
    except Exception:
        for name, index in indices.items():
            frame = read_frame(path, index)
            if frame is not None:
                out[name] = frame
    return out


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    width = max(1, int(round(w * height / max(1, h))))
    return np.asarray(Image.fromarray(img).resize((width, height), Image.Resampling.LANCZOS))


def resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    return np.asarray(Image.fromarray(img).resize((ref.shape[1], ref.shape[0]), Image.Resampling.LANCZOS))


def put_label(img: np.ndarray, text: str, subtext: str | None = None) -> np.ndarray:
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    label_h = 42 if subtext else 28
    draw.rectangle((0, 0, pil.width, label_h), fill=(0, 0, 0))
    draw.text((8, 6), text[:70], fill=(255, 255, 255), font=font)
    if subtext:
        draw.text((8, 24), subtext[:80], fill=(230, 230, 230), font=font)
    return np.asarray(pil)


def _saturation_mean(x: np.ndarray) -> float:
    maxc = x.max(axis=2)
    minc = x.min(axis=2)
    return float(((maxc - minc) / np.maximum(maxc, 1e-6)).mean())


def compute_metrics(img: np.ndarray, ref: np.ndarray | None = None) -> dict[str, float]:
    x = img.astype(np.float32) / 255.0
    flat = x.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    gray = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
    gy, gx = np.gradient(gray)
    edge_energy = float((gx * gx + gy * gy).mean() * 1_000_000.0)
    out = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "r_mean": float(mean_rgb[0]),
        "g_mean": float(mean_rgb[1]),
        "b_mean": float(mean_rgb[2]),
        "r_std": float(std_rgb[0]),
        "g_std": float(std_rgb[1]),
        "b_std": float(std_rgb[2]),
        "rgb_cast": float(mean_rgb.max() - mean_rgb.min()),
        "saturation": _saturation_mean(x),
        "edge_energy": edge_energy,
        "mse_to_hr": math.nan,
        "edge_ratio_to_hr": math.nan,
        "brightness_delta_to_hr": math.nan,
    }
    if ref is not None:
        ref_r = resize_like(ref, img)
        r = ref_r.astype(np.float32) / 255.0
        ref_gray = 0.299 * r[:, :, 0] + 0.587 * r[:, :, 1] + 0.114 * r[:, :, 2]
        rgy, rgx = np.gradient(ref_gray)
        ref_edge = float((rgx * rgx + rgy * rgy).mean() * 1_000_000.0)
        out["mse_to_hr"] = float(((x - r) ** 2).mean())
        out["edge_ratio_to_hr"] = float(edge_energy / max(ref_edge, 1e-6))
        out["brightness_delta_to_hr"] = float(out["brightness"] - ref_gray.mean())
    return out


def classify_flags(m: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if m["brightness"] < 0.07:
        flags.append("black/dark")
    if m["brightness"] > 0.93:
        flags.append("over-bright")
    if m["contrast"] < 0.035 or max(m["r_std"], m["g_std"], m["b_std"]) < 0.045:
        flags.append("gray/low-contrast")

    r, g, b = m["r_mean"], m["g_mean"], m["b_mean"]
    if g > r + 0.055 and g > b + 0.055:
        flags.append("green-cast")
    elif r > b + 0.055 and g > b + 0.045:
        flags.append("yellow/warm-cast")
    elif m["rgb_cast"] > 0.14:
        flags.append("color-cast")

    if m["edge_energy"] < 18.0:
        flags.append("very-blurry")
    if not math.isnan(m["edge_ratio_to_hr"]):
        if m["edge_ratio_to_hr"] < 0.35:
            flags.append("blurrier-than-HR")
        elif m["edge_ratio_to_hr"] > 2.7 and m.get("mse_to_hr", 0.0) > 0.018:
            flags.append("over-sharp/structure-risk")
    if not math.isnan(m["mse_to_hr"]) and m["mse_to_hr"] > 0.055:
        flags.append("structure/color-bad")
    if not flags:
        flags.append("ok-by-proxy")
    return flags


def _find_match(items: list[MediaItem], key: str) -> MediaItem | None:
    if not items:
        return None
    for item in items:
        if item.sample_key == key:
            return item
    for item in items:
        if key in item.sample_key or item.sample_key in key:
            return item
    return items[0]


def build_sample_records(media: dict[str, list[MediaItem]]) -> list[tuple[str, dict[str, MediaItem]]]:
    """Build review records.

    Clean validation directories usually have one HR/LQ and many SR step files.
    In that case each SR file must become its own row. Probe dumps usually have
    one file per semantic stream and should be grouped by sample key.
    """
    sr_items = media.get("SR", [])
    sr_has_steps = any("step" in item.path.stem.lower() for item in sr_items)
    if len(sr_items) > 1 or sr_has_steps:
        records: list[tuple[str, dict[str, MediaItem]]] = []
        for anchor in sr_items:
            record: dict[str, MediaItem] = {"SR": anchor}
            for stream in STREAM_ORDER:
                if stream == "SR":
                    continue
                match = _find_match(media.get(stream, []), anchor.sample_key)
                if match is not None:
                    record[stream] = match
            records.append((anchor.path.stem, record))
        return records

    anchor_keys: list[str] = []
    for stream in ANCHOR_STREAMS:
        for item in media.get(stream, []):
            if item.sample_key not in anchor_keys:
                anchor_keys.append(item.sample_key)
    if not anchor_keys:
        for items in media.values():
            for item in items:
                if item.sample_key not in anchor_keys:
                    anchor_keys.append(item.sample_key)

    grouped: list[tuple[str, dict[str, MediaItem]]] = []
    for key in anchor_keys:
        record = {}
        for stream in STREAM_ORDER:
            match = _find_match(media.get(stream, []), key)
            if match is not None:
                record[stream] = match
        grouped.append((key, record))
    return grouped


def _concat_horizontal(images: list[np.ndarray], pad: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("no images to concatenate")
    height = max(img.shape[0] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[0] < height:
            bottom = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img = np.concatenate([img, bottom], axis=0)
        padded.append(img)
    sep = np.zeros((height, pad, 3), dtype=images[0].dtype)
    out = padded[0]
    for img in padded[1:]:
        out = np.concatenate([out, sep, img], axis=1)
    return out


def _concat_vertical(images: list[np.ndarray], pad: int = 10) -> np.ndarray:
    if not images:
        raise ValueError("no images to concatenate")
    width = max(img.shape[1] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[1] < width:
            right = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=img.dtype)
            img = np.concatenate([img, right], axis=1)
        padded.append(img)
    sep = np.zeros((pad, width, 3), dtype=images[0].dtype)
    out = padded[0]
    for img in padded[1:]:
        out = np.concatenate([out, sep, img], axis=0)
    return out


def unique_output_dir(out: Path, overwrite: bool = False) -> Path:
    if overwrite or not out.exists() or not any(out.iterdir()):
        out.mkdir(parents=True, exist_ok=True)
        return out
    base = out
    for idx in range(2, 1000):
        candidate = base.with_name(f"{base.name}_v{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"could not allocate non-overwriting output directory for {out}")


def write_report(root: Path, out_dir: Path, height: int, max_rows: int) -> None:
    cases = discover_case_dirs(root)
    rows: list[dict[str, str | float | int]] = []
    first_rows: list[np.ndarray] = []
    multi_rows: list[np.ndarray] = []

    for case_dir in cases:
        case_name = case_dir.name if case_dir != root else root.name
        media = collect_media(case_dir)
        sample_records = build_sample_records(media)
        for record_idx, (sample_name, record) in enumerate(sample_records):
            if len(first_rows) >= max_rows:
                break
            sample_name = sample_name or f"sample_{record_idx:03d}"
            hr_item = record.get("HR")
            hr_first = read_frame(hr_item.path, 0) if hr_item is not None else None

            first_tiles: list[np.ndarray] = []
            multi_tiles: list[np.ndarray] = []
            for stream in STREAM_ORDER:
                item = record.get(stream)
                if item is None:
                    continue
                frames = sample_frames(item.path)
                for frame_name, frame in frames.items():
                    met = compute_metrics(frame, hr_first if stream != "HR" else None)
                    fl = classify_flags(met)
                    rows.append(
                        {
                            "case": case_name,
                            "sample": sample_name,
                            "stream": stream,
                            "file": item.path.name,
                            "frame": frame_name,
                            "flags": ",".join(fl),
                            "frame_count": _media_count(item.path),
                            **met,
                        }
                    )

                first = frames.get("first")
                if first is not None:
                    met = compute_metrics(first, hr_first if stream != "HR" else None)
                    sub = ",".join(classify_flags(met))
                    first_tiles.append(
                        put_label(
                            resize_to_height(first, height),
                            f"{case_name}/{sample_name} {stream}",
                            sub,
                        )
                    )
                multi_parts: list[np.ndarray] = []
                for frame_name in ("first", "mid", "last"):
                    frame = frames.get(frame_name)
                    if frame is None:
                        continue
                    multi_parts.append(
                        put_label(
                            resize_to_height(frame, height),
                            f"{stream}:{frame_name}",
                        )
                    )
                if multi_parts:
                    multi_tiles.append(_concat_horizontal(multi_parts, pad=4))

            if first_tiles:
                first_rows.append(_concat_horizontal(first_tiles))
            if multi_tiles:
                multi_rows.append(_concat_horizontal(multi_tiles))

    if first_rows:
        Image.fromarray(_concat_vertical(first_rows)).save(out_dir / "first_frame_compare.png")
    if multi_rows:
        Image.fromarray(_concat_vertical(multi_rows)).save(out_dir / "multiframe_compare.png")

    if rows:
        csv_path = out_dir / "metrics.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with (out_dir / "visual_judge.md").open("w") as f:
        f.write("# Stage3 Probe Visual Report\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write(f"Cases discovered: `{len(cases)}`\n\n")
        f.write("## Outputs\n\n")
        f.write("- `first_frame_compare.png`: first-frame review montage.\n")
        f.write("- `multiframe_compare.png`: first/mid/last temporal review montage.\n")
        f.write("- `metrics.csv`: per-stream, per-frame proxy metrics and flags.\n\n")
        f.write("## First-Frame Flags\n\n")
        f.write("| case | sample | stream | file | flags | brightness | contrast | rgb_cast | edge | mse_to_hr |\n")
        f.write("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            if row["frame"] != "first":
                continue
            f.write(
                f"| {row['case']} | {row['sample']} | {row['stream']} | {row['file']} | "
                f"{row['flags']} | {float(row['brightness']):.3f} | "
                f"{float(row['contrast']):.3f} | {float(row['rgb_cast']):.3f} | "
                f"{float(row['edge_energy']):.1f} | {float(row['mse_to_hr']):.4f} |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- Flags are proxy heuristics only. Use the montage for final visual judgment.\n")
        f.write("- `over-sharp/structure-risk` means edge energy is much higher than HR while MSE is also high.\n")
        f.write("- Probe streams are compared in a fixed order: HR, LQ, SR, real_x0, fake_x0, student_z_pred.\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Input clean/probe root directory.")
    parser.add_argument("--out", required=True, type=Path, help="Output report directory.")
    parser.add_argument("--height", type=int, default=180, help="Tile height in montage images.")
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum sample rows in montage.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    out_dir = unique_output_dir(args.out.expanduser().resolve(), overwrite=args.overwrite)
    write_report(root=root, out_dir=out_dir, height=args.height, max_rows=args.max_rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
