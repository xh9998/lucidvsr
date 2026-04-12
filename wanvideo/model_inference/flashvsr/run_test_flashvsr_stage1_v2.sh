#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_8gpu_v2_20260409_121105/output/step-100.safetensors}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/eval_samples/v2/sample_000/lq.mp4}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_stage1_v2_${RUN_TS}}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${RUN_DIR}/sr.mp4}"
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

mkdir -p "${RUN_DIR}"

echo "ROOT_DIR=${ROOT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "INPUT_VIDEO=${INPUT_VIDEO}"
echo "PROJECTION_SCALE=${PROJECTION_SCALE}"

cd "${ROOT_DIR}"

EXTRA_ARGS=()
if [ "${DISABLE_LORA}" = "1" ]; then
  EXTRA_ARGS+=(--disable_lora)
fi
if [ "${DISABLE_PROJECTION}" = "1" ]; then
  EXTRA_ARGS+=(--disable_projection)
fi

{
  /mnt/conda_envs/flashvsr/bin/python \
    wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_video "${INPUT_VIDEO}" \
    --output_video "${OUTPUT_VIDEO}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --projection_scale "${PROJECTION_SCALE}" \
    --save_input_lq \
    "${EXTRA_ARGS[@]}"
} 2>&1 | tee "${RUN_DIR}/run.log"
