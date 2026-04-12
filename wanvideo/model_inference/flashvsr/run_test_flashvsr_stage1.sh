#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
if [ -f /miniforge/etc/profile.d/conda.sh ]; then
  source /miniforge/etc/profile.d/conda.sh
  conda activate flashvsr
fi
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_8gpu_debug_noval_20260407_144928/output/step-200.safetensors}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/exp/train_validation_smoke_2gpu/output/validation/step-1/sample_000/lq.mp4}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_infer_test_$(date +%Y%m%d_%H%M%S)/sr.mp4}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
NUM_FRAMES="${NUM_FRAMES:-89}"
FPS="${FPS:-8}"
SELECT_PHYSICAL_GPU="${SELECT_PHYSICAL_GPU:-}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ -n "${SELECT_PHYSICAL_GPU}" ]; then
    export CUDA_VISIBLE_DEVICES="${SELECT_PHYSICAL_GPU}"
  else
    SELECTED_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1 != 0 {gsub(/ /, \"\", $1); gsub(/ /, \"\", $2); print $1\",\"$2}' | sort -t, -k2n | head -n1 | cut -d, -f1)"
    if [ -z "${SELECTED_GPU}" ]; then
      SELECTED_GPU="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    fi
    export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"
  fi
fi

echo "Using physical GPU(s): ${CUDA_VISIBLE_DEVICES}"

python wanvideo/model_inference/flashvsr/inspect_flashvsr_stage1_ckpt.py \
  --checkpoint_path "${CHECKPOINT_PATH}"

python wanvideo/model_inference/flashvsr/infer_flashvsr_stage1.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
  --input_video "${INPUT_VIDEO}" \
  --output_video "${OUTPUT_VIDEO}" \
  --height 768 \
  --width 1280 \
  --num_frames "${NUM_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --fps "${FPS}" \
  --save_input_lq
