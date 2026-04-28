#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
INPUT_DIR="${INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/input/inference_input5_first17_20260421}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/compare_triplet_input5_first17_20260421/flashvsr_official_scale1_native}"
CUDA_DEVICE="${CUDA_DEVICE:-2}"
SEED="${SEED:-0}"

cd "${ROOT_DIR}"
INPUT_DIR="${INPUT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CUDA_DEVICE="${CUDA_DEVICE}" \
SEED="${SEED}" \
SCALE=1 \
bash wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh
