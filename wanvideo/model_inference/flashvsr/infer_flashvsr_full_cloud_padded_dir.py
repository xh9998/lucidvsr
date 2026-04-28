#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from wanvideo.model_inference.flashvsr.infer_flashvsr_full_cloud_padded import (
    count_frames_and_fps,
    pad_video_to_length,
    smallest_8n_minus_3_geq,
    trim_video,
)


def list_videos(input_dir: str):
    exts = (".mp4", ".mov", ".avi", ".mkv")
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(exts)
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flashvsr_repo", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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
    inputs = list_videos(args.input_dir)
    if not inputs:
        raise ValueError(f"No supported videos found in input_dir={args.input_dir}")

    repo = Path(args.flashvsr_repo).resolve()
    infer_py = repo / "examples" / "WanVSR" / "infer_flashvsr_full_cloud.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"missing infer script: {infer_py}")

    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flashvsr_pad_dir_") as tmpdir:
        padded_dir = os.path.join(tmpdir, "inputs")
        temp_output_dir = os.path.join(tmpdir, "out")
        os.makedirs(padded_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        metadata = []
        for input_video in inputs:
            original_frames, fps = count_frames_and_fps(input_video)
            target_frames = smallest_8n_minus_3_geq(original_frames)
            stem = os.path.splitext(os.path.basename(input_video))[0]
            padded_input = os.path.join(padded_dir, f"{stem}.mp4")
            pad_video_to_length(input_video, padded_input, target_frames, fps)
            metadata.append((stem, original_frames, fps))
            print(f"[pad] {stem}: frames={original_frames} padded={target_frames} fps={fps}")

        cmd = [
            os.environ.get("PYTHON_BIN", "python"),
            str(infer_py),
            "--input_path",
            padded_dir,
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

        for stem, original_frames, fps in metadata:
            produced = os.path.join(temp_output_dir, f"FlashVSR_v1.1_Full_{stem}_seed{args.seed}.mp4")
            if not os.path.exists(produced):
                raise FileNotFoundError(f"missing FlashVSR output for {stem}: {produced}")
            output_video = os.path.join(args.output_dir, f"{stem}_sr.mp4")
            trim_video(produced, output_video, original_frames, fps)
            print(f"[trim] {stem}: saved={output_video} frames={original_frames} fps={fps}")


if __name__ == "__main__":
    main()
