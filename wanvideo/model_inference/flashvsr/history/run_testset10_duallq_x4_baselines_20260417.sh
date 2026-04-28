#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

export TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_duallq_20260417}"

cd /mnt/task_runtime/lucidvsr

CUDA_DEVICE="${CUDA_DEVICE_FLASHVSR:-1}" \
OUTPUT_ROOT="${OUTPUT_ROOT_FLASHVSR:-/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_full_testset10_duallq_x4_20260417}" \
INPUT_SUBDIR="lq_x4" \
bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_testset10_paired_20260417.sh

CUDA_DEVICE="${CUDA_DEVICE_SEEDVR3B:-1}" \
OUTPUT_ROOT="${OUTPUT_ROOT_SEEDVR3B:-/mnt/task_wrapper/user_output/artifacts/inference/seedvr_3b_testset10_duallq_x4_20260417}" \
INPUT_SUBDIR="lq_x4" \
bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_seedvr_3b_testset10_paired_20260417.sh

CUDA_DEVICE="${CUDA_DEVICE_SEEDVR7B:-1}" \
OUTPUT_ROOT="${OUTPUT_ROOT_SEEDVR7B:-/mnt/task_wrapper/user_output/artifacts/inference/seedvr_7b_testset10_duallq_x4_20260417}" \
INPUT_SUBDIR="lq_x4" \
bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_seedvr_7b_testset10_paired_20260417.sh
