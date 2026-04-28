#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

FLASHVSR_REPO="${FLASHVSR_REPO:-/mnt/task_runtime/FlashVSR}"
WORK_DIR="${FLASHVSR_REPO}/examples/WanVSR"
INPUT_DIR="${INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/input/testset10/lq}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_full_testset10_20260417}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SEED="${SEED:-0}"
SCALE="${SCALE:-4}"
SPARSE_RATIO="${SPARSE_RATIO:-2.0}"
KV_RATIO="${KV_RATIO:-3.0}"
LOCAL_RANGE="${LOCAL_RANGE:-11}"
QUALITY="${QUALITY:-6}"
TILED_FLAG="${TILED_FLAG:-0}"
DEBUG_DUMP_DIR="${DEBUG_DUMP_DIR:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${WORK_DIR}"

ARGS=(
  --input_path "${INPUT_DIR}"
  --output_path "${OUTPUT_DIR}"
  --model_dir "${MODEL_DIR}"
  --seed "${SEED}"
  --scale "${SCALE}"
  --sparse_ratio "${SPARSE_RATIO}"
  --kv_ratio "${KV_RATIO}"
  --local_range "${LOCAL_RANGE}"
  --quality "${QUALITY}"
)

if [[ "${TILED_FLAG}" == "1" ]]; then
  ARGS+=(--tiled)
fi

if [[ -n "${DEBUG_DUMP_DIR}" ]]; then
  ARGS+=(--debug_dump_dir "${DEBUG_DUMP_DIR}")
fi

printf '%q ' "${PYTHON_BIN}" infer_flashvsr_full_cloud.py "${ARGS[@]}" > "${OUTPUT_DIR}/launch_command.sh"
printf '\n' >> "${OUTPUT_DIR}/launch_command.sh"
cp "${OUTPUT_DIR}/launch_command.sh" "${OUTPUT_DIR}/launch_command.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" infer_flashvsr_full_cloud.py "${ARGS[@]}"
