#!/usr/bin/env python3
import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio
import numpy as np


def count_frames_and_fps(video_path: str):
    reader = imageio.get_reader(video_path)
    try:
        meta = {}
        try:
            meta = reader.get_meta_data()
        except Exception:
            meta = {}
        fps_val = meta.get("fps", 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        try:
            nframes = meta.get("nframes", None)
            if isinstance(nframes, int) and nframes > 0:
                return nframes, fps
        except Exception:
            pass

        count = 0
        while True:
            try:
                reader.get_data(count)
                count += 1
            except Exception:
                break
        return count, fps
    finally:
        try:
            reader.close()
        except Exception:
            pass


def smallest_8n_minus_3_geq(n: int) -> int:
    if n <= 0:
        raise ValueError(f"invalid frame count: {n}")
    k = max(1, math.ceil((n + 3) / 8))
    return 8 * k - 3


def pad_video_to_length(input_video: str, padded_video: str, target_frames: int, fps: int):
    reader = imageio.get_reader(input_video)
    writer = imageio.get_writer(
        padded_video,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        frames = []
        idx = 0
        while True:
            try:
                frame = reader.get_data(idx)
                frames.append(frame)
                writer.append_data(frame)
                idx += 1
            except Exception:
                break
        if not frames:
            raise RuntimeError(f"no frames found in {input_video}")
        last_frame = frames[-1]
        for _ in range(max(0, target_frames - len(frames))):
            writer.append_data(last_frame)
    finally:
        try:
            reader.close()
        except Exception:
            pass
        writer.close()


def trim_video(input_video: str, output_video: str, keep_frames: int, fps: int):
    reader = imageio.get_reader(input_video)
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    writer = imageio.get_writer(
        output_video,
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for idx in range(keep_frames):
            writer.append_data(reader.get_data(idx))
    finally:
        try:
            reader.close()
        except Exception:
            pass
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashvsr_repo", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sparse_ratio", type=float, default=2.0)
    parser.add_argument("--kv_ratio", type=float, default=3.0)
    parser.add_argument("--local_range", type=int, default=11)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--tiled", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    original_frames, fps = count_frames_and_fps(args.input_video)
    target_frames = smallest_8n_minus_3_geq(original_frames)

    repo = Path(args.flashvsr_repo).resolve()
    infer_py = repo / "examples" / "WanVSR" / "infer_flashvsr_full_cloud.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"missing infer script: {infer_py}")

    with tempfile.TemporaryDirectory(prefix="flashvsr_pad_") as tmpdir:
        padded_input = os.path.join(tmpdir, "input_padded.mp4")
        temp_output_dir = os.path.join(tmpdir, "out")
        os.makedirs(temp_output_dir, exist_ok=True)
        pad_video_to_length(args.input_video, padded_input, target_frames, fps)

        cmd = [
            os.environ.get("PYTHON_BIN", "python"),
            str(infer_py),
            "--input_path",
            padded_input,
            "--output_path",
            temp_output_dir,
            "--model_dir",
            args.model_dir,
            "--seed",
            str(args.seed),
            "--scale",
            str(args.scale),
            "--sparse_ratio",
            str(args.sparse_ratio),
            "--kv_ratio",
            str(args.kv_ratio),
            "--local_range",
            str(args.local_range),
            "--quality",
            str(args.quality),
        ]
        if args.tiled:
            cmd.append("--tiled")
        subprocess.run(cmd, check=True, cwd=str(repo / "examples" / "WanVSR"))

        produced = sorted(Path(temp_output_dir).glob("*.mp4"))
        if len(produced) != 1:
            raise RuntimeError(f"expected one output mp4, got {len(produced)} in {temp_output_dir}")

        os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
        trim_video(str(produced[0]), args.output_video, original_frames, fps)
        print(
            f"saved={args.output_video} input_frames={original_frames} "
            f"padded_frames={target_frames} fps={fps}"
        )


if __name__ == "__main__":
    main()
