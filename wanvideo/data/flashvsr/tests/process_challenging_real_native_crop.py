#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

import av
import imageio
import numpy as np
from PIL import Image


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_videos(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTS)


def resize_cover(frame: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = frame.size
    scale = max(width / src_w, height / src_h)
    resized_w = max(width, int(round(src_w * scale)))
    resized_h = max(height, int(round(src_h * scale)))
    return frame.resize((resized_w, resized_h), Image.BICUBIC)


def center_crop(frame: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = frame.size
    if src_w < width or src_h < height:
        raise ValueError(f"source frame too small for native crop: got {src_w}x{src_h}, need {width}x{height}")
    left = (src_w - width) // 2
    top = (src_h - height) // 2
    return frame.crop((left, top, left + width, top + height))


def read_first_frames(video_path: Path, num_frames: int, width: int, height: int) -> List[Image.Image]:
    frames: List[Image.Image] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            image = resize_cover(frame.to_image().convert("RGB"), width, height)
            frames.append(center_crop(image, width, height))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise RuntimeError(f"{video_path} decoded no frames")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames


def save_video_exact(frames: List[Image.Image], save_path: Path, fps: int) -> None:
    writer = imageio.get_writer(
        str(save_path),
        fps=fps,
        quality=5,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize-cover and center-crop real videos to the fixed VSR LQ size.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list_videos(input_dir)
    if not videos:
        raise FileNotFoundError(f"no videos found in {input_dir}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "rule": "resize-cover then center crop; first N decoded frames; repeat last frame if source is short; output fps fixed",
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "videos": [],
    }

    for video_path in videos:
        frames = read_first_frames(video_path, args.num_frames, args.width, args.height)
        out_path = output_dir / f"{video_path.stem}_{args.num_frames}f_{args.width}x{args.height}.mp4"
        save_video_exact(frames, out_path, args.fps)
        item = {"input": str(video_path), "output": str(out_path), "frames": len(frames), "fps": args.fps}
        summary["videos"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"[done] processed {len(summary['videos'])} videos -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
