#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_TRAIN_DEBUG=0

CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu_v2_debug_overfit.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v2_debug_overfit}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29620}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
MAX_TRAIN_STEPS_OVERRIDE="${MAX_TRAIN_STEPS_OVERRIDE:-}"
LQ_PROJ_SCALE_OVERRIDE="${LQ_PROJ_SCALE_OVERRIDE:-}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/mnt/task_wrapper/user_output/artifacts/exp"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"

TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2_debug.py"
ACCEL_SRC="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml"
DEEPSPEED_CONFIG_ABS="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/deepspeed_zero2_flashvsr.json"
ACCEL_DST="${RUN_DIR}/snapshot/accelerate_2gpu_resolved.yaml"
SELF_SH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU-v2-Debug.sh"
python - "$ACCEL_SRC" "$ACCEL_DST" "$DEEPSPEED_CONFIG_ABS" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
ds_abs = sys.argv[3]
text = src.read_text(encoding="utf-8")
text = text.replace("wanvideo/model_training/flashvsr/lora/deepspeed_zero2_flashvsr.json", ds_abs)
dst.write_text(text, encoding="utf-8")
PY

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "Using Python: $(command -v python)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Port: ${MAIN_PROCESS_PORT}"
echo "Batch override: ${BATCH_SIZE_OVERRIDE:-<none>}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_SRC}" "${RUN_DIR}/snapshot/" || true
cp "${SELF_SH}" "${RUN_DIR}/snapshot/" || true

CMD=(
  /mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_DST}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
)

if [[ -n "${BATCH_SIZE_OVERRIDE}" ]]; then
  CMD+=(--batch_size "${BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
  CMD+=(--max_train_steps "${MAX_TRAIN_STEPS_OVERRIDE}")
fi
if [[ -n "${LQ_PROJ_SCALE_OVERRIDE}" ]]; then
  CMD+=(--lq_proj_scale "${LQ_PROJ_SCALE_OVERRIDE}")
fi

printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"
printf '\n' >> "${RUN_DIR}/launch_command.sh"
cp "${RUN_DIR}/launch_command.sh" "${RUN_DIR}/launch_command.txt"

"${CMD[@]}"
