#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_TRAIN_DEBUG=0

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5.yaml"
OUTPUT_TAG="train_stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5"
HISTORY_DIR="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/mnt/task_wrapper/user_output/artifacts/exp"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
mkdir -p "${HISTORY_DIR}"

TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py"
ACCEL_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_8gpu.yaml"
SELF_SH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-8GPU-v2-17f-Full-bs4-lr1e5-alpha5.sh"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "Using Python: $(command -v python)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Config: ${CONFIG_PATH}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "${SELF_SH}" "${RUN_DIR}/snapshot/" || true

CMD=(
  /mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
)

printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"
printf '\n' >> "${RUN_DIR}/launch_command.sh"
cp "${RUN_DIR}/launch_command.sh" "${RUN_DIR}/launch_command.txt"

{
  echo "# ${RUN_NAME}"
  echo
  echo "- Run dir: \`${RUN_DIR}\`"
  echo "- Config: \`${CONFIG_PATH}\`"
  echo "- Train py: \`${TRAIN_PY}\`"
  echo "- Accelerate: \`${ACCEL_YAML}\`"
  echo
  echo '```bash'
  cat "${RUN_DIR}/launch_command.sh"
  echo '```'
} > "${HISTORY_DIR}/${RUN_NAME}.md"

"${CMD[@]}"
