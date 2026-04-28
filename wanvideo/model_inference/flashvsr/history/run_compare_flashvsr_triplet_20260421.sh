#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
FLASHVSR_REPO="${FLASHVSR_REPO:-/mnt/task_runtime/FlashVSR}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
INPUT_DIR="${INPUT_DIR:?need INPUT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:?need OUTPUT_ROOT}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-17}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_LORA="${DEVICE_LORA:-cuda}"
DEVICE_FULL="${DEVICE_FULL:-cuda}"
CUDA_LORA="${CUDA_LORA:-0}"
CUDA_FULL="${CUDA_FULL:-1}"
CUDA_FLASHVSR="${CUDA_FLASHVSR:-0}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
LORA_CKPT_PATH="${LORA_CKPT_PATH:?need LORA_CKPT_PATH}"
FULLFT_CKPT_PATH="${FULLFT_CKPT_PATH:?need FULLFT_CKPT_PATH}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_lora() {
  local run_dir="${OUTPUT_ROOT}/lora17_step2500_alpha1"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"
  for input_video in "${INPUT_DIR}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    CUDA_VISIBLE_DEVICES="${CUDA_LORA}" /mnt/conda_envs/flashvsr/bin/python \
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
      --checkpoint_path "${LORA_CKPT_PATH}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${input_video}" \
      --output_video "${run_dir}/${name}_sr.mp4" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale "${LQ_PROJ_SCALE}" \
      --device "${DEVICE_LORA}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${run_dir}/run.log"
  done
}

run_fullft() {
  local run_dir="${OUTPUT_ROOT}/fullft17_step2800_alpha1"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"
  for input_video in "${INPUT_DIR}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    CUDA_VISIBLE_DEVICES="${CUDA_FULL}" /mnt/conda_envs/flashvsr/bin/python \
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v3_fullft.py \
      --checkpoint_path "${FULLFT_CKPT_PATH}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${input_video}" \
      --output_video "${run_dir}/${name}_sr.mp4" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale "${LQ_PROJ_SCALE}" \
      --device "${DEVICE_FULL}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${run_dir}/run.log"
  done
}

run_flashvsr() {
  local run_dir="${OUTPUT_ROOT}/flashvsr_official"
  mkdir -p "${run_dir}"
  (
    cd "${FLASHVSR_REPO}"
    INPUT_DIR="${INPUT_DIR}" \
    OUTPUT_DIR="${run_dir}" \
    CUDA_DEVICE="${CUDA_FLASHVSR}" \
    SEED="${SEED}" \
    bash flashvsr_inference_cloud_full.sh
  ) 2>&1 | tee "${run_dir}/run.log"
}

run_lora &
PID_LORA=$!
run_fullft &
PID_FULL=$!
wait "${PID_LORA}"
wait "${PID_FULL}"
run_flashvsr
