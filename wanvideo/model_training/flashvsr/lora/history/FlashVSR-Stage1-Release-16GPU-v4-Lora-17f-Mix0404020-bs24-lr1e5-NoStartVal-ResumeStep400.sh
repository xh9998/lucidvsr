#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-4}"
export FLASHVSR_IO_NODE_LIMIT_DIR="${FLASHVSR_IO_NODE_LIMIT_DIR:-/tmp/flashvsr_io_limiter}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"
export CONDUCTOR_CACHE_MAX_BYTES="${CONDUCTOR_CACHE_MAX_BYTES:-214748364800}"
export CONDUCTOR_CACHE_TARGET_PCT="${CONDUCTOR_CACHE_TARGET_PCT:-0.80}"
export CONDUCTOR_CACHE_TTL_SECONDS="${CONDUCTOR_CACHE_TTL_SECONDS:-21600}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v4_lora_17f_mix0404020_bs24_lr1e5_nostartval_resume_step400.yaml"
OUTPUT_TAG="train_stage1_release_16gpu_v4_lora_17f_mix0404020_bs24_lr1e5_nostartval_resume_step400"
HISTORY_DIR="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2node16gpu_nooffload.template.yaml"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0 or 1)}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="/mnt/task_wrapper/user_output/artifacts/exp"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="${EXP_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
mkdir -p "${HISTORY_DIR}"
mkdir -p /mnt/task_runtime/flashvsr_cache/v4/media /mnt/task_runtime/flashvsr_cache/v4/parquet

TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py"
SELF_SH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v4-Lora-17f-Mix0404020-bs24-lr1e5-NoStartVal-ResumeStep400.sh"
ACCEL_YAML="${RUN_DIR}/accelerate_2node16gpu.yaml"

sed \
  -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" \
  -e "s/__MASTER_PORT__/${MASTER_PORT}/g" \
  -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" \
  "${TEMPLATE_YAML}" > "${ACCEL_YAML}"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

echo "Using Python: $(command -v python)"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MACHINE_RANK: ${MACHINE_RANK}"
echo "Accelerate: ${ACCEL_YAML}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "${TEMPLATE_YAML}" "${RUN_DIR}/snapshot/" || true
cp "${SELF_SH}" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

CMD=(
  /mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  --num_machines 2
  --num_processes 16
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  --deepspeed_multinode_launcher standard
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
  echo "- Resume state: \`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v4_lora_17f_mix0404020_bs24_lr1e5_nostartval_20260420_143500/output/training_state/step-400\`"
  echo "- Accelerate template: \`${TEMPLATE_YAML}\`"
  echo "- Master addr: \`${MASTER_ADDR}\`"
  echo "- Master port: \`${MASTER_PORT}\`"
  echo "- Machine rank: \`${MACHINE_RANK}\`"
  echo
  echo '```bash'
  cat "${RUN_DIR}/launch_command.sh"
  echo '```'
} > "${HISTORY_DIR}/${RUN_NAME}.md"

"${CMD[@]}"
