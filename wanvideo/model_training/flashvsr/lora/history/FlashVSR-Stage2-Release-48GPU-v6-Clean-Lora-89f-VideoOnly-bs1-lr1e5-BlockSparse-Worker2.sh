#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-4}"
export FLASHVSR_IO_NODE_LIMIT_DIR="${FLASHVSR_IO_NODE_LIMIT_DIR:-/tmp/flashvsr_io_limiter}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"
export CONDUCTOR_CACHE_MAX_BYTES="${CONDUCTOR_CACHE_MAX_BYTES:-214748364800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml"
OUTPUT_TAG="train_stage2_release_48gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_6node48gpu_nooffload.template.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_clean_lora.py"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0..5)}"
: "${RESUME_STAGE1_CHECKPOINT:?RESUME_STAGE1_CHECKPOINT must point to Stage1 v5.3.5 checkpoint}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_video_train_all.txt" "${TAKANO_MANIFEST}"
fi
echo "takano_manifest=${TAKANO_MANIFEST} lines=$(wc -l < "${TAKANO_MANIFEST}")"
echo "resume_stage1_checkpoint=${RESUME_STAGE1_CHECKPOINT}"
echo "worker_setting=dataset_num_workers:2 prefetch:1 persistent:true"

ACCEL_YAML="${RUN_DIR}/accelerate_6node48gpu.yaml"
sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/diffsynth/models/wan_video_dit_stage2_v6_clean.py" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  --num_machines 6
  --num_processes 48
  --machine_rank "${MACHINE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  --deepspeed_multinode_launcher standard
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --resume_stage1_checkpoint "${RESUME_STAGE1_CHECKPOINT}"
  --zero_init_lq_proj_in false
)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
