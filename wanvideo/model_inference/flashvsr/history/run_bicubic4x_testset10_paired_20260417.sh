#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_paired}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/bicubic4x_testset10_paired_20260417}"
ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" wanvideo/model_inference/flashvsr/bicubic_upscale_testset10_paired.py \
  --testset_root "${TESTSET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --scale 4
