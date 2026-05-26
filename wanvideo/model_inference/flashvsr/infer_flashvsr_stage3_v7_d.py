import argparse

from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1 import (
    add_common_args,
    build_stage2_pipe,
    run_single_video,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage3 v7-D one-step inference using the Stage2 official-style "
            "streaming/KV-cache path. This matches v7-D validation semantics."
        )
    )
    add_common_args(parser)
    parser.set_defaults(num_inference_steps=1, tiled=False)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    args = parser.parse_args()
    print("stage3_v7_d_inference=streaming_kvcache_one_step")
    pipe = build_stage2_pipe(args)
    run_single_video(args, pipe)


if __name__ == "__main__":
    main()
