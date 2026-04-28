#!/usr/bin/env python3
import argparse
import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def count_frames_and_fps(video_path: str):
    reader = imageio.get_reader(video_path)
    try:
        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            meta = {}
        fps_val = meta.get("fps", 8)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 8
        frame_count = 0
        while True:
            try:
                reader.get_data(frame_count)
                frame_count += 1
            except Exception:
                break
        return frame_count, fps
    finally:
        try:
            reader.close()
        except Exception:
            pass


def rewrite_downsampled(video_path: str, factor: int):
    frame_count, fps = count_frames_and_fps(video_path)
    reader = imageio.get_reader(video_path)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="down4x_", suffix=".mp4")
    os.close(tmp_fd)
    writer = imageio.get_writer(
        tmp_path,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for idx in range(frame_count):
            frame = reader.get_data(idx)
            image = Image.fromarray(frame)
            width, height = image.size
            image = image.resize((max(1, width // factor), max(1, height // factor)), Image.BICUBIC)
            writer.append_data(np.asarray(image))
    finally:
        try:
            reader.close()
        except Exception:
            pass
        writer.close()
    os.replace(tmp_path, video_path)
    print(f"downsampled={video_path} frames={frame_count} fps={fps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_root", required=True)
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.testset_root)
    video_paths = sorted(root.glob("testset10_*/lq/*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"no lq videos found under {root}")
    for path in video_paths:
        rewrite_downsampled(str(path), args.factor)


if __name__ == "__main__":
    main()
