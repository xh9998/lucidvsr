#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh
FLASHVSR_PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/task_runtime/lucidvsr}"

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
LORA_CKPT_PATH="${LORA_CKPT_PATH:?need LORA_CKPT_PATH}"
FULLFT_CKPT_PATH="${FULLFT_CKPT_PATH:?need FULLFT_CKPT_PATH}"
INPUT_DIR_TESTSET="${INPUT_DIR_TESTSET:-/mnt/task_wrapper/user_output/artifacts/input/testset5_takano_x4_20260421/lq_x4}"
INPUT_DIR_INPUT5="${INPUT_DIR_INPUT5:-/mnt/task_wrapper/user_output/artifacts/input/inference_input5_first17_20260421}"
OUTPUT_ROOT_TESTSET="${OUTPUT_ROOT_TESTSET:-/mnt/task_wrapper/user_output/artifacts/inference/compare_my_colorfix_testset5_x4_20260421}"
OUTPUT_ROOT_INPUT5="${OUTPUT_ROOT_INPUT5:-/mnt/task_wrapper/user_output/artifacts/inference/compare_my_colorfix_input5_native_20260421}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-17}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"
CUDA_TESTSET_LORA="${CUDA_TESTSET_LORA:-0}"
CUDA_TESTSET_FULL="${CUDA_TESTSET_FULL:-1}"
CUDA_INPUT5_LORA="${CUDA_INPUT5_LORA:-3}"
CUDA_INPUT5_FULL="${CUDA_INPUT5_FULL:-4}"

cd "${ROOT_DIR}"

run_batch_v2() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  local upscale="$4"
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
      --color_fix_method "${COLOR_FIX_METHOD}" \
      --input_bicubic_upscale "${upscale}" \
      --device cuda \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${output_dir}/run.log"
  done
}

run_batch_v3() {
  local input_dir="$1"
  local output_dir="$2"
  local gpu="$3"
  local upscale="$4"
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
      --color_fix_method "${COLOR_FIX_METHOD}" \
      --input_bicubic_upscale "${upscale}" \
      --device cuda \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled 2>&1 | tee -a "${output_dir}/run.log"
  done
}

run_batch_v2 "${INPUT_DIR_TESTSET}" "${OUTPUT_ROOT_TESTSET}/lora17_step2500_colorfix_${COLOR_FIX_METHOD}" "${CUDA_TESTSET_LORA}" 1 &
run_batch_v3 "${INPUT_DIR_TESTSET}" "${OUTPUT_ROOT_TESTSET}/fullft17_step2800_colorfix_${COLOR_FIX_METHOD}" "${CUDA_TESTSET_FULL}" 1 &
run_batch_v2 "${INPUT_DIR_INPUT5}" "${OUTPUT_ROOT_INPUT5}/lora17_step2500_colorfix_${COLOR_FIX_METHOD}" "${CUDA_INPUT5_LORA}" 1 &
run_batch_v3 "${INPUT_DIR_INPUT5}" "${OUTPUT_ROOT_INPUT5}/fullft17_step2800_colorfix_${COLOR_FIX_METHOD}" "${CUDA_INPUT5_FULL}" 1 &

wait
