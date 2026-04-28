import argparse
import os
from pathlib import Path

from diffsynth.utils.data import VideoData, save_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim every mp4 in a directory to the first N frames.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.mp4")):
        frames = VideoData(str(path), height=args.height, width=args.width).raw_data()
        frames = frames[: args.num_frames]
        if len(frames) == 0:
            continue
        save_video(
            frames,
            str(output_dir / path.name),
            fps=args.fps,
            quality=5,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        print(f"saved={output_dir / path.name}")


if __name__ == "__main__":
    main()
