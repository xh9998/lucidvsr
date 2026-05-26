import argparse
import os
from types import SimpleNamespace

from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1 import add_common_args, build_stage2_pipe
from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_2 import run_single_video


def list_input_videos(input_dir: str):
    return [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(".mp4")
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for Stage2 v6.2 full-sequence masked DiT checkpoints. Loads Wan/VAE/projector once per checkpoint."
    )
    add_common_args(parser)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_suffix", type=str, default="_sr")
    parser.add_argument("--print_debug", action="store_true", default=False)
    args = parser.parse_args()

    input_videos = list_input_videos(args.input_dir)
    if not input_videos:
        raise ValueError(f"No mp4 inputs found in input_dir={args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"input_dir={args.input_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"num_inputs={len(input_videos)}")
    print("v6_2_mode=full_sequence_masked_dit_no_kvcache")

    pipe = build_stage2_pipe(args)
    for index, input_video in enumerate(input_videos, start=1):
        sample = os.path.splitext(os.path.basename(input_video))[0]
        output_video = os.path.join(args.output_dir, f"{sample}{args.output_suffix}.mp4")
        print(f"[{index}/{len(input_videos)}] input={input_video}")
        print(f"[{index}/{len(input_videos)}] output={output_video}")
        video_args = SimpleNamespace(**vars(args))
        video_args.input_video = input_video
        video_args.output_video = output_video
        run_single_video(video_args, pipe)


if __name__ == "__main__":
    main()
