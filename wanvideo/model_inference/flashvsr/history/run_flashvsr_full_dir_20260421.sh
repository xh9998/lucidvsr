#!/usr/bin/env bash
set -euo pipefail

FLASHVSR_PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

FLASHVSR_REPO="${FLASHVSR_REPO:-/mnt/task_runtime/FlashVSR}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
INPUT_DIR="${INPUT_DIR:?need INPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?need OUTPUT_DIR}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEED="${SEED:-0}"
SCALE="${SCALE:-4}"
SPARSE_RATIO="${SPARSE_RATIO:-2.0}"
KV_RATIO="${KV_RATIO:-3.0}"
LOCAL_RANGE="${LOCAL_RANGE:-11}"
QUALITY="${QUALITY:-6}"
TILED_FLAG="${TILED_FLAG:-1}"

mkdir -p "${OUTPUT_DIR}"
cd /mnt/task_runtime/lucidvsr

args=(
  "wanvideo/model_inference/flashvsr/infer_flashvsr_full_cloud_padded_dir.py"
  --flashvsr_repo "${FLASHVSR_REPO}"
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --model_dir "${MODEL_DIR}"
  --seed "${SEED}"
  --scale "${SCALE}"
  --sparse_ratio "${SPARSE_RATIO}"
  --kv_ratio "${KV_RATIO}"
  --local_range "${LOCAL_RANGE}"
  --quality "${QUALITY}"
)
if [[ "${TILED_FLAG}" == "1" ]]; then
  args+=(--tiled)
fi
PYTHON_BIN="${FLASHVSR_PYTHON_BIN}" CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${FLASHVSR_PYTHON_BIN}" "${args[@]}"
