import argparse
import csv
from pathlib import Path

import torch

from wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1 import (
    add_common_args,
    build_stage2_pipe,
    load_lq_video_frames,
)


def _flatten_stats(tensor: torch.Tensor):
    x = tensor.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "l2": float(torch.linalg.vector_norm(x).item()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Dump Stage2 v6 LQ projector chunk statistics.")
    add_common_args(parser)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    pipe = build_stage2_pipe(args)
    lq_video, effective_height, effective_width = load_lq_video_frames(
        args.input_video,
        args.height,
        args.width,
        args.input_bicubic_upscale,
    )
    lq_video = lq_video[: args.num_frames]
    if not lq_video:
        raise ValueError(f"No frames loaded from input_video={args.input_video}")
    lq_tensor = pipe.preprocess_video(lq_video).to(device=pipe.device, dtype=pipe.torch_dtype)

    pipe.lq_proj_in.clear_cache()
    rows = []
    chunk_outputs = []
    # Match FlashVSRLQProjIn.forward(): prepend first-frame warmup, then call
    # stream_forward in 4-frame units.
    raw_frames = int(lq_tensor.shape[2])
    iterations = 1 + (raw_frames - 1) // 4
    first_frame = lq_tensor[:, :, :1].repeat(1, 1, 3, 1, 1)
    projector_input = torch.cat([first_frame, lq_tensor], dim=2)
    for chunk_id in range(iterations):
        start = chunk_id * 4
        end = min((chunk_id + 1) * 4, int(projector_input.shape[2]))
        clip = projector_input[:, :, start:end]
        cur = pipe.lq_proj_in.stream_forward(clip)
        if cur is None:
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "raw_start": max(0, start - 3),
                    "raw_end_exclusive": max(0, end - 3),
                    "layer_id": -1,
                    "tokens": 0,
                    "status": "warmup",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "l2": "",
                }
            )
            continue
        chunk_outputs.append(cur)
        for layer_id, layer_tensor in enumerate(cur):
            stats = _flatten_stats(layer_tensor)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "raw_start": max(0, start - 3),
                    "raw_end_exclusive": max(0, end - 3),
                    "layer_id": layer_id,
                    "tokens": int(layer_tensor.shape[1]),
                    "status": "output",
                    **stats,
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        fieldnames = [
            "chunk_id",
            "raw_start",
            "raw_end_exclusive",
            "layer_id",
            "tokens",
            "status",
            "mean",
            "std",
            "min",
            "max",
            "l2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"input_video={args.input_video}")
    print(f"effective_size={effective_width}x{effective_height}")
    print(f"frames={len(lq_video)} projector_output_chunks={len(chunk_outputs)}")
    print(f"saved_csv={output_csv}")


if __name__ == "__main__":
    main()
