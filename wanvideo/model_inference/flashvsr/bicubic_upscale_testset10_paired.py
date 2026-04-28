#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def read_meta(video_path: str):
    reader = imageio.get_reader(video_path)
    try:
        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            meta = {}
        fps_val = meta.get("fps", 8)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 8
        frames = []
        idx = 0
        while True:
            try:
                frames.append(reader.get_data(idx))
                idx += 1
            except Exception:
                break
        return frames, fps
    finally:
        try:
            reader.close()
        except Exception:
            pass


def upscale_video(input_video: str, output_video: str, scale: int):
    frames, fps = read_meta(input_video)
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            image = Image.fromarray(frame)
            width, height = image.size
            image = image.resize((width * scale, height * scale), Image.BICUBIC)
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
    print(f"saved={output_video}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    for variant_dir in sorted(Path(args.testset_root).glob("testset10_*")):
        input_dir = variant_dir / "lq"
        output_dir = Path(args.output_root) / variant_dir.name
        os.makedirs(output_dir, exist_ok=True)
        for input_video in sorted(input_dir.glob("*.mp4")):
            sample_name = input_video.stem
            output_video = output_dir / f"{sample_name}_sr.mp4"
            upscale_video(str(input_video), str(output_video), args.scale)


if __name__ == "__main__":
    main()
