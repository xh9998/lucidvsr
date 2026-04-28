#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
FLASHVSR_PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
INPUT_DIR_NATIVE="${INPUT_DIR_NATIVE:-/mnt/task_wrapper/user_output/artifacts/input/inference_input5_first17_20260421}"
INPUT_DIR_X4="${INPUT_DIR_X4:-/mnt/task_wrapper/user_output/artifacts/input/inference_input5_first17_x4_20260421}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/compare_triplet_input5_first17_x4_20260421}"
LORA_CKPT_PATH="${LORA_CKPT_PATH:?need LORA_CKPT_PATH}"
FULLFT_CKPT_PATH="${FULLFT_CKPT_PATH:?need FULLFT_CKPT_PATH}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-17}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
CUDA_LORA="${CUDA_LORA:-0}"
CUDA_FULL="${CUDA_FULL:-1}"

mkdir -p "${INPUT_DIR_X4}" "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

make_x4_inputs() {
  for input_video in "${INPUT_DIR_NATIVE}"/*.mp4; do
    local name width height out_w out_h
    name="$(basename "${input_video}")"
    read -r width height < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=' ' "${input_video}")
    out_w=$(( width / 4 ))
    out_h=$(( height / 4 ))
    if (( out_w < 1 )); then out_w=1; fi
    if (( out_h < 1 )); then out_h=1; fi
    ffmpeg -y -v error -i "${input_video}" -vf "scale=${out_w}:${out_h}:flags=bicubic" -an "${INPUT_DIR_X4}/${name}"
  done
}

run_batch_v2() {
  local output_dir="${OUTPUT_ROOT}/lora17_step2500_alpha1_x4"
  mkdir -p "${output_dir}"
  : > "${output_dir}/run.log"
  for input_video in "${INPUT_DIR_X4}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_LORA}" "${FLASHVSR_PYTHON_BIN}" \
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
  local output_dir="${OUTPUT_ROOT}/fullft17_step2800_alpha1_x4"
  mkdir -p "${output_dir}"
  : > "${output_dir}/run.log"
  for input_video in "${INPUT_DIR_X4}"/*.mp4; do
    local name
    name="$(basename "${input_video}" .mp4)"
    PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_FULL}" "${FLASHVSR_PYTHON_BIN}" \
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

make_x4_inputs
run_batch_v2 &
PID_LORA=$!
run_batch_v3 &
PID_FULL=$!
wait "${PID_LORA}"
wait "${PID_FULL}"
