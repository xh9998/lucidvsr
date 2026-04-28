#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

FLASHVSR_REPO="${FLASHVSR_REPO:-/mnt/task_runtime/FlashVSR}"
WORK_DIR="${FLASHVSR_REPO}/examples/WanVSR"
TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_paired}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_full_testset10_paired_20260417}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
INPUT_SUBDIR="${INPUT_SUBDIR:-lq}"
SEED="${SEED:-0}"
SCALE="${SCALE:-4}"
SPARSE_RATIO="${SPARSE_RATIO:-2.0}"
KV_RATIO="${KV_RATIO:-3.0}"
LOCAL_RANGE="${LOCAL_RANGE:-11}"
QUALITY="${QUALITY:-6}"
TILED_FLAG="${TILED_FLAG:-1}"

mkdir -p "${OUTPUT_ROOT}"
cd "${WORK_DIR}"

run_variant() {
  local variant="$1"
  local input_dir="${TESTSET_ROOT}/${variant}/${INPUT_SUBDIR}"
  local output_dir="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${output_dir}"

  : > "${output_dir}/run.log"
  for input_video in "${input_dir}"/*.mp4; do
    local sample_name
    sample_name="$(basename "${input_video}" .mp4)"
    local output_video="${output_dir}/${sample_name}_sr.mp4"

    local -a args=(
      "${ROOT_DIR:-/mnt/task_runtime/lucidvsr}/wanvideo/model_inference/flashvsr/infer_flashvsr_full_cloud_padded.py"
      --flashvsr_repo "${FLASHVSR_REPO}"
      --input_video "${input_video}"
      --output_video "${output_video}"
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

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${args[@]}" \
      2>&1 | tee -a "${output_dir}/run.log"
  done
}

run_variant "testset10_17f"
run_variant "testset10_89f"
