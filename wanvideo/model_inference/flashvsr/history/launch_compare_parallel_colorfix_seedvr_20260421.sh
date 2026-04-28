#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
FLASHVSR_PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
LORA_CKPT_PATH="${LORA_CKPT_PATH:?need LORA_CKPT_PATH}"
FULLFT_CKPT_PATH="${FULLFT_CKPT_PATH:?need FULLFT_CKPT_PATH}"
INPUT_DIR_TESTSET="${INPUT_DIR_TESTSET:?need INPUT_DIR_TESTSET}"
INPUT_DIR_INPUT5="${INPUT_DIR_INPUT5:?need INPUT_DIR_INPUT5}"
OUTPUT_ROOT_TESTSET="${OUTPUT_ROOT_TESTSET:?need OUTPUT_ROOT_TESTSET}"
OUTPUT_ROOT_INPUT5="${OUTPUT_ROOT_INPUT5:?need OUTPUT_ROOT_INPUT5}"
SEEDVR_OUTPUT_ROOT_TESTSET="${SEEDVR_OUTPUT_ROOT_TESTSET:?need SEEDVR_OUTPUT_ROOT_TESTSET}"
SEEDVR_OUTPUT_ROOT_INPUT5="${SEEDVR_OUTPUT_ROOT_INPUT5:?need SEEDVR_OUTPUT_ROOT_INPUT5}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-17}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
SEEDVR1_MODEL_DIR="${SEEDVR1_MODEL_DIR:-/mnt/models/SeedVR-3B}"
SEEDVR2_MODEL_DIR="${SEEDVR2_MODEL_DIR:-/mnt/models/SeedVR2-3B}"
SEEDVR_SHARED_EMB_DIR="${SEEDVR_SHARED_EMB_DIR:-/mnt/task_runtime/SeedVR}"

cd "${ROOT_DIR}"

run_batch_v2() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  mkdir -p "${output_dir}"
  : > "${output_dir}/run.log"
  for input_video in "${input_dir}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${gpu}" "${FLASHVSR_PYTHON_BIN}" \
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
      --checkpoint_path "${LORA_CKPT_PATH}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${input_video}" \
      --output_video "${output_dir}/${name}_sr.mp4" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale "${LQ_PROJ_SCALE}" \
      --device cuda \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${output_dir}/run.log"
  done
}

run_batch_v3() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  mkdir -p "${output_dir}"
  : > "${output_dir}/run.log"
  for input_video in "${input_dir}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${gpu}" "${FLASHVSR_PYTHON_BIN}" \
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v3_fullft.py \
      --checkpoint_path "${FULLFT_CKPT_PATH}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${input_video}" \
      --output_video "${output_dir}/${name}_sr.mp4" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale "${LQ_PROJ_SCALE}" \
      --device cuda \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${output_dir}/run.log"
  done
}

run_batch_official() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  mkdir -p "${output_dir}"
  : > "${output_dir}/run.log"
  INPUT_DIR="${input_dir}" \
  OUTPUT_DIR="${output_dir}" \
  CUDA_DEVICE="${gpu}" \
  SEED="${SEED}" \
  bash wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh \
    2>&1 | tee -a "${output_dir}/run.log"
}

run_official_queue() {
  local gpu="$1"
  run_batch_official "${INPUT_DIR_TESTSET}" "${OUTPUT_ROOT_TESTSET}/flashvsr_official" "${gpu}"
  run_batch_official "${INPUT_DIR_INPUT5}" "${OUTPUT_ROOT_INPUT5}/flashvsr_official" "${gpu}"
}

run_seedvr_queue() {
  local model_kind="$1"
  local model_dir="$2"
  local gpu="$3"
  local output_testset="$4"
  local output_input5="$5"

  mkdir -p "${output_testset}" "${output_input5}"
  if [[ ! -f "${model_dir}/pos_emb.pt" ]]; then
    ln -sfn "${SEEDVR_SHARED_EMB_DIR}/pos_emb.pt" "${model_dir}/pos_emb.pt"
  fi
  if [[ ! -f "${model_dir}/neg_emb.pt" ]]; then
    ln -sfn "${SEEDVR_SHARED_EMB_DIR}/neg_emb.pt" "${model_dir}/neg_emb.pt"
  fi

  MODEL_KIND="${model_kind}" \
  MODEL_DIR="${model_dir}" \
  INPUT_DIR="${INPUT_DIR_TESTSET}" \
  OUTPUT_DIR="${output_testset}" \
  CUDA_DEVICE="${gpu}" \
  bash wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh

  MODEL_KIND="${model_kind}" \
  MODEL_DIR="${model_dir}" \
  INPUT_DIR="${INPUT_DIR_INPUT5}" \
  OUTPUT_DIR="${output_input5}" \
  CUDA_DEVICE="${gpu}" \
  bash wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh
}

run_batch_v2 "${INPUT_DIR_TESTSET}" "${OUTPUT_ROOT_TESTSET}/lora17_step2500_alpha1_colorfix" 0 &
run_batch_v3 "${INPUT_DIR_TESTSET}" "${OUTPUT_ROOT_TESTSET}/fullft17_step2800_alpha1_colorfix" 1 &
run_batch_v2 "${INPUT_DIR_INPUT5}" "${OUTPUT_ROOT_INPUT5}/lora17_step2500_alpha1_colorfix" 3 &
run_batch_v3 "${INPUT_DIR_INPUT5}" "${OUTPUT_ROOT_INPUT5}/fullft17_step2800_alpha1_colorfix" 4 &
run_official_queue 2 &
run_seedvr_queue "seedvr1" "${SEEDVR1_MODEL_DIR}" 7 "${SEEDVR_OUTPUT_ROOT_TESTSET}/seedvr1_3b" "${SEEDVR_OUTPUT_ROOT_INPUT5}/seedvr1_3b" &
run_seedvr_queue "seedvr2" "${SEEDVR2_MODEL_DIR}" 6 "${SEEDVR_OUTPUT_ROOT_TESTSET}/seedvr2_3b" "${SEEDVR_OUTPUT_ROOT_INPUT5}/seedvr2_3b" &

wait
