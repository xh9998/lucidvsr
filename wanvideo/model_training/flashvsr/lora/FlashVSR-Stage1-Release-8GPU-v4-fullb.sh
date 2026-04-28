#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-4}"
export FLASHVSR_IO_NODE_LIMIT_DIR="${FLASHVSR_IO_NODE_LIMIT_DIR:-/tmp/flashvsr_io_limiter}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/mnt/task_wrapper/user_output/artifacts/exp"
RUN_NAME="train_stage1_release_8gpu_v4_fullb_${RUN_TS}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

TRAIN_PY="wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_fullb.py"
CONFIG_YAML="wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v4_fullb.yaml"
ACCEL_YAML="wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_8gpu.yaml"
SELF_SH="wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v4-fullb.sh"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "Using Python: $(command -v python)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"

CMD=(
  /mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  "${TRAIN_PY}"
  --config "${CONFIG_YAML}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
)

mkdir -p "${RUN_DIR}/snapshot"
cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_YAML}" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "${SELF_SH}" "${RUN_DIR}/snapshot/" || true
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"
printf '\n' >> "${RUN_DIR}/launch_command.sh"
cp "${RUN_DIR}/launch_command.sh" "${RUN_DIR}/launch_command.txt"

"${CMD[@]}"
