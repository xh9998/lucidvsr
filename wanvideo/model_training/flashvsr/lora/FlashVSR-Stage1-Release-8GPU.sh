#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/mnt/task_wrapper/user_output/artifacts/exp"
RUN_NAME="train_stage1_release_8gpu_${RUN_TS}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export FLASHVSR_DEBUG_DIR="${RUN_DIR}/debug"
mkdir -p "${FLASHVSR_DEBUG_DIR}"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "Using Python: $(command -v python)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"

/mnt/conda_envs/flashvsr/bin/accelerate launch \
  --config_file wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_8gpu.yaml \
  wanvideo/model_training/flashvsr/train_flashvsr_stage1.py \
  --config wanvideo/model_training/flashvsr/configs/stage1_release_8gpu.yaml \
  --output_path "${RUN_DIR}/output" \
  --wandb_name "${RUN_NAME}"
