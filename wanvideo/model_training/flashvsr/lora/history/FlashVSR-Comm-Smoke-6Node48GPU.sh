#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [ "${FORCE_NCCL_SOCKET:-0}" = "1" ]; then
  export NCCL_NET="Socket"
  export NCCL_IB_DISABLE="1"
fi

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0..5)}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="flashvsr_comm_smoke_6node48gpu_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/smoke/${RUN_NAME}"
mkdir -p "${RUN_DIR}/snapshot"

TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_6node48gpu_nooffload.template.yaml"
ACCEL_YAML="${RUN_DIR}/accelerate_6node48gpu.yaml"
sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"
cp /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/ddp_comm_smoke.py "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1
CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch --config_file "${ACCEL_YAML}" --num_machines 6 --num_processes 48 --machine_rank "${MACHINE_RANK}" --main_process_ip "${MASTER_ADDR}" --main_process_port "${MASTER_PORT}" --deepspeed_multinode_launcher standard /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/ddp_comm_smoke.py)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
