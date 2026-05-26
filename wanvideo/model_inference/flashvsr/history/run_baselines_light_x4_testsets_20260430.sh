#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"

SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_20260430/lq}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x109_nativecrop_20260430}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/compare_light_x4_flash_seedvr3b_20260430_by_dataset}"

FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
SEEDVR3B_MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

run_flashvsr_dataset() {
  local gpu="$1"
  local dataset="$2"
  local input_dir="$3"
  local out_dir="${OUTPUT_ROOT}/${dataset}/flashvsr_official"
  local log_file="${OUTPUT_ROOT}/logs/flashvsr_official_${dataset}.log"
  mkdir -p "${out_dir}"
  CUDA_DEVICE="${gpu}" \
  INPUT_DIR="${input_dir}" \
  OUTPUT_DIR="${out_dir}" \
  MODEL_DIR="${FLASHVSR_MODEL_DIR}" \
  SCALE=4 \
  SEED="${SEED}" \
  bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh" \
    2>&1 | tee "${log_file}"
}

run_seedvr_dataset() {
  local gpu="$1"
  local dataset="$2"
  local input_dir="$3"
  local res_h="$4"
  local res_w="$5"
  local out_dir="${OUTPUT_ROOT}/${dataset}/seedvr3b"
  local log_file="${OUTPUT_ROOT}/logs/seedvr3b_${dataset}.log"
  mkdir -p "${out_dir}"
  CUDA_DEVICE="${gpu}" \
  MODEL_KIND=seedvr1 \
  MODEL_DIR="${SEEDVR3B_MODEL_DIR}" \
  INPUT_DIR="${input_dir}" \
  OUTPUT_DIR="${out_dir}" \
  SEEDVR_PYTHON="${SEEDVR_PYTHON}" \
  RES_H="${res_h}" \
  RES_W="${res_w}" \
  SEED="${SEED}" \
  LOG_FILE="${log_file}" \
  bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/seedvr3b_${dataset}_launcher.log"
}

{
  echo "SYNTHETIC_INPUT_DIR=${SYNTHETIC_INPUT_DIR}"
  echo "REAL_INPUT_DIR=${REAL_INPUT_DIR}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "FPS=${FPS}"
  echo "FlashVSR keeps source fps through wrapper trim; test inputs are expected to be ${FPS} fps."
} | tee "${OUTPUT_ROOT}/settings.txt"

run_flashvsr_dataset 0 synthetic "${SYNTHETIC_INPUT_DIR}" &
pid_flash_synth=$!
run_seedvr_dataset 1 synthetic "${SYNTHETIC_INPUT_DIR}" 768 1280 &
pid_seed_synth=$!
run_flashvsr_dataset 2 real "${REAL_INPUT_DIR}" &
pid_flash_real=$!
run_seedvr_dataset 3 real "${REAL_INPUT_DIR}" 436 1280 &
pid_seed_real=$!

wait "${pid_flash_synth}"
wait "${pid_seed_synth}"
wait "${pid_flash_real}"
wait "${pid_seed_real}"

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
conductor s3 sync "${OUTPUT_ROOT}" "s3://lxh/data/test/$(basename "${OUTPUT_ROOT}")"
echo "[done] ${OUTPUT_ROOT}"
