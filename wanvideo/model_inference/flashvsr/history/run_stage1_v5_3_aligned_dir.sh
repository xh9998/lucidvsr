#!/usr/bin/env bash
set -euo pipefail

# Batch inference for v5.3 / v5.3.5 / v5.3.6 Stage-1 checkpoints.
# This path is for non-streaming aligned projector experiments:
#   - lq_proj_temporal_mode=nonstreaming_aligned
#   - input video is bicubic-upscaled by 4x before entering the model
#   - color fix is enabled by default
#   - Wan/VAE/projector are loaded once for the whole input directory

export PYTHONPATH="/mnt/task_runtime/lucidvsr:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a v5.3 aligned checkpoint}"
INPUT_DIR="${INPUT_DIR:?Set INPUT_DIR to a directory containing mp4 inputs}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for SR outputs}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
INPUT_BICUBIC_UPSCALE="${INPUT_BICUBIC_UPSCALE:-4.0}"
LQ_PROJ_TEMPORAL_MODE="${LQ_PROJ_TEMPORAL_MODE:-nonstreaming_aligned}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -u wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned_batch.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --input_bicubic_upscale "${INPUT_BICUBIC_UPSCALE}" \
  --lq_proj_temporal_mode "${LQ_PROJ_TEMPORAL_MODE}" \
  --lq_proj_scale "${LQ_PROJ_SCALE}" \
  --color_fix_method "${COLOR_FIX_METHOD}" \
  "$@"
