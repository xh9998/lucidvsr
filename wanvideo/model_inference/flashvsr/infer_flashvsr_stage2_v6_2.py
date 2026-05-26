import argparse
import os
from typing import Any

from diffsynth.utils.data import save_video
from wanvideo.model_inference.flashvsr.color_fix import apply_color_fix
from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1 import (
    add_common_args,
    build_stage2_pipe,
    load_lq_video_frames,
)


def run_single_video(args: Any, pipe):
    """Stage2 v6.2 inference: training-aligned full DiT pass.

    Difference from v6.1:
    - LQ projector is still causal/streaming, producing the full post-first-frame latent sequence.
    - DiT receives the full latent sequence in one call.
    - DiT self-attention uses the same chunk-causal/block-sparse mask as training.
    - No inference-time K/V cache and no 6+2+2 chunk loop.
    """
    lq_video, effective_height, effective_width = load_lq_video_frames(
        args.input_video,
        args.height,
        args.width,
        args.input_bicubic_upscale,
    )
    lq_video = lq_video[: args.num_frames]
    if not lq_video:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")

    if args.print_debug:
        raw_frames = len(lq_video)
        expected_latents = max(1, (raw_frames - 1) // 4)
        print("[v6.2-debug] mode=full_dit_mask_no_kvcache")
        print(f"[v6.2-debug] input_frames={raw_frames}")
        print(f"[v6.2-debug] expected_stage2_latent_time={expected_latents}")
        print(f"[v6.2-debug] effective_size={effective_width}x{effective_height}")
        print(f"[v6.2-debug] stage2_attention_mode={args.stage2_attention_mode}")
        print(f"[v6.2-debug] stage2_topk_ratio={args.stage2_topk_ratio}")
        print(f"[v6.2-debug] stage2_local_num={args.stage2_local_num}")

    sr_video = pipe.infer_from_lq(
        lq_video=lq_video,
        height=effective_height,
        width=effective_width,
        num_frames=len(lq_video),
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        tiled=args.tiled,
        output_type="quantized",
    )

    if not args.disable_color_fix:
        if len(sr_video) != len(lq_video):
            aligned = min(len(sr_video), len(lq_video))
            print(f"[color_fix] frame_count_mismatch sr={len(sr_video)} lq={len(lq_video)} using={aligned}")
            sr_video = sr_video[:aligned]
            lq_for_color = lq_video[:aligned]
        else:
            lq_for_color = lq_video
        sr_video = apply_color_fix(sr_video, lq_for_color, method=args.color_fix_method)

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    save_video(sr_video, args.output_video, fps=args.fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"saved_sr={args.output_video}")


def main():
    parser = argparse.ArgumentParser(description="Stage2 v6.2 training-aligned full-sequence inference.")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, required=True)
    parser.add_argument("--print_debug", action="store_true", default=False)
    args = parser.parse_args()

    pipe = build_stage2_pipe(args)
    run_single_video(args, pipe)


if __name__ == "__main__":
    main()
