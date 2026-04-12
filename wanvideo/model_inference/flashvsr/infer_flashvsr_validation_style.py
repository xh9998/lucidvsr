import argparse
import json
import os
from copy import deepcopy

import yaml
import torch

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.utils.data import save_video
from wanvideo.data.flashvsr.datasets.streaming_dataset import FlashVSRStreamingDataset
from wanvideo.model_training.flashvsr.train_flashvsr_stage1 import (
    FlashVSRStage1Pipeline,
    collect_fixed_validation_samples,
    _tensor_video_to_pil_frames,
)


def split_flashvsr_ckpt(state_dict):
    lq_proj_state = {}
    lora_state = {}
    other_state = {}
    for key, value in state_dict.items():
        if key.startswith("lq_proj_in."):
            lq_proj_state[key[len("lq_proj_in."):]] = value
        elif "lora_" in key:
            lora_state[key] = value
        else:
            other_state[key] = value
    return lq_proj_state, lora_state, other_state


def build_model_configs(model_paths):
    return [ModelConfig(path=path) for path in model_paths]


def parse_model_paths(raw_model_paths):
    if isinstance(raw_model_paths, list):
        return raw_model_paths
    if isinstance(raw_model_paths, str):
        return json.loads(raw_model_paths)
    raise TypeError(f"Unsupported model_paths type: {type(raw_model_paths)}")


def build_streaming_dataset(args_dict):
    return FlashVSRStreamingDataset(
        internal_url=args_dict.get("internal_url"),
        metadata_url=args_dict.get("metadata_url"),
        metadata_source=args_dict.get("metadata_source"),
        max_parquet_records=args_dict.get("max_parquet_records"),
        min_overall_score=args_dict.get("min_overall_score", -1.0),
        require_qwen35_parse_success=args_dict.get("require_qwen35_parse_success", False),
        image_internal_url=args_dict.get("image_internal_url", ""),
        image_dataset_prob=args_dict.get("image_dataset_prob", 0.0),
        height=args_dict["height"],
        width=args_dict["width"],
        num_frames=args_dict["num_frames"],
        stride=args_dict.get("stride", 1),
        max_source_frames=args_dict.get("max_source_frames"),
        enable_degradation=args_dict.get("enable_degradation", True),
        degradation_config_path=args_dict.get("degradation_config_path"),
        degradation_seed=args_dict.get("degradation_seed"),
        hq_prefix_frames=args_dict.get("hq_prefix_frames", 0),
        control_dropout_prob=args_dict.get("control_dropout_prob", 0.0),
        shuffle_buffer=args_dict.get("shuffle_buffer", 128),
        global_seed=args_dict.get("global_seed"),
        output_tensors=True,
    )


def main():
    parser = argparse.ArgumentParser(description="按训练内 validation 的同源路径做 FlashVSR 推理。")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()

    resolved_args_path = os.path.join(args.experiment_dir, "output", "resolved_args.yaml")
    with open(resolved_args_path, "r", encoding="utf-8") as file:
        resolved_args = yaml.safe_load(file)

    dataset = build_streaming_dataset(resolved_args)
    validation_samples = collect_fixed_validation_samples(dataset, args.num_samples)
    sample = deepcopy(validation_samples[args.sample_index])

    hr_tensor = sample["video"]
    lq_tensor = sample["lq_video"]
    sample_seed = resolved_args.get("global_seed", 20260407) + args.sample_index
    fps = args.fps if args.fps is not None else resolved_args.get("validation_fps", 8)
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else resolved_args.get("validation_num_inference_steps", 10)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    ckpt = load_state_dict(args.checkpoint_path, device="cpu")
    lq_proj_state, lora_state, other_state = split_flashvsr_ckpt(ckpt)
    print(f"checkpoint={args.checkpoint_path}")
    print(f"lq_proj_keys={len(lq_proj_state)} lora_keys={len(lora_state)} other_keys={len(other_state)}")

    model_paths = parse_model_paths(resolved_args["model_paths"])
    pipe = FlashVSRStage1Pipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=args.device,
        model_configs=build_model_configs(model_paths),
        prompt_tensor_path=resolved_args["prompt_tensor_path"],
    )
    pipe.lq_proj_in.load_state_dict(lq_proj_state, strict=False)
    if lora_state:
        pipe.load_lora(pipe.dit, state_dict=lora_state, verbose=1)

    os.makedirs(args.output_dir, exist_ok=True)
    hr_frames = _tensor_video_to_pil_frames(hr_tensor)
    lq_frames = _tensor_video_to_pil_frames(lq_tensor)
    save_video(hr_frames, os.path.join(args.output_dir, "hr.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])
    save_video(lq_frames, os.path.join(args.output_dir, "lq.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

    sr_frames = pipe.infer_from_lq(
        lq_video=lq_tensor.unsqueeze(0),
        height=int(hr_tensor.shape[-2]),
        width=int(hr_tensor.shape[-1]),
        num_frames=int(hr_tensor.shape[0]),
        seed=sample_seed,
        rand_device="cpu",
        num_inference_steps=num_inference_steps,
        tiled=True,
        output_type="quantized",
    )
    save_video(sr_frames, os.path.join(args.output_dir, "sr.mp4"), fps=fps, quality=5, ffmpeg_params=["-pix_fmt", "yuv420p"])

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment_dir": args.experiment_dir,
                "checkpoint_path": args.checkpoint_path,
                "sample_index": args.sample_index,
                "sample_seed": sample_seed,
                "num_inference_steps": num_inference_steps,
                "fps": fps,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved_dir={args.output_dir}")


if __name__ == "__main__":
    main()
