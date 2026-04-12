#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
INPUT_DIR="${INPUT_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-step-*.safetensors}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-}"
INPUT_GLOB="${INPUT_GLOB:-sample_*/lq.mp4}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-89}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
DISABLE_LORA="${DISABLE_LORA:-0}"
DISABLE_PROJECTION="${DISABLE_PROJECTION:-0}"
PROJECTION_SCALE="${PROJECTION_SCALE:-1.0}"

if [ -z "${CHECKPOINT_DIR}" ] || [ -z "${INPUT_DIR}" ] || [ -z "${OUTPUT_ROOT}" ]; then
  echo "CHECKPOINT_DIR, INPUT_DIR, OUTPUT_ROOT are required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

if [ -n "${CHECKPOINT_NAMES}" ]; then
  IFS=',' read -r -a CKPT_NAMES_ARR <<< "${CHECKPOINT_NAMES}"
  CKPTS=()
  for ckpt_name in "${CKPT_NAMES_ARR[@]}"; do
    ckpt_path="${CHECKPOINT_DIR}/${ckpt_name}"
    if [ -f "${ckpt_path}" ]; then
      CKPTS+=("${ckpt_path}")
    fi
  done
else
  mapfile -t CKPTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "${CHECKPOINT_GLOB}" | sort -V)
fi
mapfile -t INPUTS < <(find "${INPUT_DIR}" -path "${INPUT_DIR}/${INPUT_GLOB}" -type f | sort)

if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "No checkpoints found under ${CHECKPOINT_DIR} matching ${CHECKPOINT_GLOB}" >&2
  exit 1
fi
if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "No input videos found under ${INPUT_DIR} matching ${INPUT_GLOB}" >&2
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  step_name="$(basename "${ckpt}" .safetensors)"
  for input_video in "${INPUTS[@]}"; do
    sample_name="$(basename "$(dirname "${input_video}")")"
    run_dir="${OUTPUT_ROOT}/${step_name}/${sample_name}"
    mkdir -p "${run_dir}"

    cmd=(
      /mnt/conda_envs/flashvsr/bin/python
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py
      --checkpoint_path "${ckpt}"
      --base_model_dir "${BASE_MODEL_DIR}"
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}"
      --input_video "${input_video}"
      --output_video "${run_dir}/sr.mp4"
      --height "${HEIGHT}"
      --width "${WIDTH}"
      --num_frames "${NUM_FRAMES}"
      --fps "${FPS}"
      --seed "${SEED}"
      --num_inference_steps "${NUM_INFERENCE_STEPS}"
      --device "${DEVICE}"
      --torch_dtype "${TORCH_DTYPE}"
      --projection_scale "${PROJECTION_SCALE}"
      --save_input_lq
    )

    if [ "${DISABLE_LORA}" = "1" ]; then
      cmd+=(--disable_lora)
    fi
    if [ "${DISABLE_PROJECTION}" = "1" ]; then
      cmd+=(--disable_projection)
    fi

    printf '%q ' "${cmd[@]}" > "${run_dir}/launch_command.sh"
    printf '\n' >> "${run_dir}/launch_command.sh"
    cp "${run_dir}/launch_command.sh" "${run_dir}/launch_command.txt"

    {
      echo "CHECKPOINT=${ckpt}"
      echo "INPUT_VIDEO=${input_video}"
      echo "RUN_DIR=${run_dir}"
      "${cmd[@]}"
    } 2>&1 | tee "${run_dir}/run.log"
  done
done
