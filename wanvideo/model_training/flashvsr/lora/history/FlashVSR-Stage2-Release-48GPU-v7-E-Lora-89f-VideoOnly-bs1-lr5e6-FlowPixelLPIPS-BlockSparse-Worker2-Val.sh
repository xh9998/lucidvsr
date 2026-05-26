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
export FLASHVSR_STAGE7E_DEBUG_LOSS="${FLASHVSR_STAGE7E_DEBUG_LOSS:-1}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val.yaml"
OUTPUT_TAG="train_stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_6node48gpu_nooffload.template.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v7_e_pixelloss_lora.py"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0..5)}"
: "${RESUME_STAGE2_CHECKPOINT:?RESUME_STAGE2_CHECKPOINT must point to Stage2 v6.4.1 step-6000 checkpoint}"
START_OCCUPY_ON_EXIT="${START_OCCUPY_ON_EXIT:-1}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
mkdir -p "${WANDB_DIR}/wandb"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

finish() {
  rc=$?
  if [ "${START_OCCUPY_ON_EXIT}" = "1" ]; then
    echo "[guard] v7E exited rc=${rc}; starting occupy on this node"
    bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh || true
  fi
  exit "${rc}"
}
trap finish EXIT

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_video_train_all.txt" "${TAKANO_MANIFEST}"
fi
echo "takano_manifest=${TAKANO_MANIFEST} lines=$(wc -l < "${TAKANO_MANIFEST}")"
if [ ! -s "${RESUME_STAGE2_CHECKPOINT}" ]; then
  mkdir -p "$(dirname "${RESUME_STAGE2_CHECKPOINT}")"
  conductor s3 cp "s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors" "${RESUME_STAGE2_CHECKPOINT}"
fi
VGG16_PATH="${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
if [ ! -s "${VGG16_PATH}" ]; then
  mkdir -p "$(dirname "${VGG16_PATH}")"
  conductor s3 cp "s3://lxh/models/SR/vgg16-397923af.pth" "${VGG16_PATH}"
fi
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "worker_setting=dataset_num_workers:2 prefetch:1 persistent:true"
echo "stage2_v7e=Stage2 v6.4.1 data/attention/50-step validation + auxiliary pixel MSE/LPIPS loss, no DMD/fake"

ACCEL_YAML="${RUN_DIR}/accelerate_6node48gpu.yaml"
sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/diffsynth/models/wan_video_dit_stage2_v6_1.py" "${RUN_DIR}/snapshot/" || true

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
  --resume_stage2_checkpoint "${RESUME_STAGE2_CHECKPOINT}"
  --zero_init_lq_proj_in false
)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
