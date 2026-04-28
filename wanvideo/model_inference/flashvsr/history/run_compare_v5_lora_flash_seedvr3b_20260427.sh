#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
LQ_PROJ_CKPT="${LQ_PROJ_CKPT:-/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
SEEDVR3B_MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}"

SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq_20260427/lq}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/challenging_test_lxh_17f_320x192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset}"

NUM_FRAMES="${NUM_FRAMES:-17}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
INPUT_BICUBIC_UPSCALE="${INPUT_BICUBIC_UPSCALE:-4.0}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"

CKPT_V53="${CKPT_V53:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200/output/step-1300.safetensors}"
CKPT_V532="${CKPT_V532:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600/output/step-2000.safetensors}"
CKPT_V531="${CKPT_V531:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100/output/step-1300.safetensors}"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT_DIR}"

run_lora_set() {
  local gpu="$1"
  local label="$2"
  local ckpt="$3"
  shift 3

  if [[ ! -f "${ckpt}" ]]; then
    echo "[missing] ${label}: ${ckpt}" >&2
    return 1
  fi

  for dataset_spec in "synthetic:${SYNTHETIC_INPUT_DIR}" "real:${REAL_INPUT_DIR}"; do
    local dataset="${dataset_spec%%:*}"
    local input_dir="${dataset_spec#*:}"
    local out_dir="${OUTPUT_ROOT}/${dataset}/${label}"
    local log_file="${OUTPUT_ROOT}/logs/${label}_${dataset}.log"
    mkdir -p "${out_dir}"

    shopt -s nullglob
    local inputs=("${input_dir}"/*.mp4)
    shopt -u nullglob
    if [[ "${#inputs[@]}" -eq 0 ]]; then
      echo "[missing-input] ${label}/${dataset}: ${input_dir}" >&2
      return 1
    fi

    echo "[lora-batch] gpu=${gpu} label=${label} dataset=${dataset} inputs=${#inputs[@]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2_batch.py \
      --checkpoint_path "${ckpt}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${input_dir}" \
      --output_dir "${out_dir}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --device cuda \
      --torch_dtype "${TORCH_DTYPE}" \
      --lq_proj_checkpoint "${LQ_PROJ_CKPT}" \
      --lq_proj_layer_num 1 \
      --projection_scale 1.0 \
      --input_bicubic_upscale "${INPUT_BICUBIC_UPSCALE}" \
      --color_fix_method "${COLOR_FIX_METHOD}" \
      2>&1 | tee "${log_file}"
  done
}

run_flashvsr_official() {
  local gpu="$1"
  for dataset_spec in "synthetic:${SYNTHETIC_INPUT_DIR}" "real:${REAL_INPUT_DIR}"; do
    local dataset="${dataset_spec%%:*}"
    local input_dir="${dataset_spec#*:}"
    local out_dir="${OUTPUT_ROOT}/${dataset}/flashvsr_official"
    local log_file="${OUTPUT_ROOT}/logs/flashvsr_official_${dataset}.log"
    mkdir -p "${out_dir}"
    echo "[flashvsr] gpu=${gpu} dataset=${dataset}"
    CUDA_DEVICE="${gpu}" \
    INPUT_DIR="${input_dir}" \
    OUTPUT_DIR="${out_dir}" \
    MODEL_DIR="${FLASHVSR_MODEL_DIR}" \
    SCALE=4 \
    SEED="${SEED}" \
    bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh" \
      2>&1 | tee "${log_file}"
  done
}

run_seedvr3b() {
  local gpu="$1"
  for dataset_spec in "synthetic:${SYNTHETIC_INPUT_DIR}" "real:${REAL_INPUT_DIR}"; do
    local dataset="${dataset_spec%%:*}"
    local input_dir="${dataset_spec#*:}"
    local out_dir="${OUTPUT_ROOT}/${dataset}/seedvr3b"
    local log_file="${OUTPUT_ROOT}/logs/seedvr3b_${dataset}.log"
    mkdir -p "${out_dir}"
    echo "[seedvr3b] gpu=${gpu} dataset=${dataset}"
    CUDA_DEVICE="${gpu}" \
    MODEL_KIND=seedvr1 \
    MODEL_DIR="${SEEDVR3B_MODEL_DIR}" \
    INPUT_DIR="${input_dir}" \
    OUTPUT_DIR="${out_dir}" \
    SEEDVR_PYTHON="${SEEDVR_PYTHON}" \
    RES_H="${HEIGHT}" \
    RES_W="${WIDTH}" \
    SEED="${SEED}" \
    LOG_FILE="${log_file}" \
    bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/seedvr3b_${dataset}_launcher.log"
  done
}

{
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "SYNTHETIC_INPUT_DIR=${SYNTHETIC_INPUT_DIR}"
  echo "REAL_INPUT_DIR=${REAL_INPUT_DIR}"
  echo "LQ_PROJ_CKPT=${LQ_PROJ_CKPT}"
  echo "INPUT_BICUBIC_UPSCALE=${INPUT_BICUBIC_UPSCALE}"
  echo "COLOR_FIX_METHOD=${COLOR_FIX_METHOD}"
  echo "CKPT_V53=${CKPT_V53}"
  echo "CKPT_V532=${CKPT_V532}"
  echo "CKPT_V531=${CKPT_V531}"
} | tee "${OUTPUT_ROOT}/settings.txt"

run_lora_set 0 "v5_3_step1300_flashproj" "${CKPT_V53}" >"${OUTPUT_ROOT}/logs/v5_3.log" 2>&1 &
pid_v53=$!
run_lora_set 1 "v5_3_2_step2000_flashproj" "${CKPT_V532}" >"${OUTPUT_ROOT}/logs/v5_3_2.log" 2>&1 &
pid_v532=$!
run_lora_set 2 "v5_3_1_step1300_flashproj" "${CKPT_V531}" >"${OUTPUT_ROOT}/logs/v5_3_1.log" 2>&1 &
pid_v531=$!
(
  run_flashvsr_official 3
  run_seedvr3b 3
) >"${OUTPUT_ROOT}/logs/baselines.log" 2>&1 &
pid_base=$!

wait "${pid_v53}"
wait "${pid_v532}"
wait "${pid_v531}"
wait "${pid_base}"

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
echo "[done] ${OUTPUT_ROOT}"
