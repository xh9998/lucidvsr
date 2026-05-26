#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-4}"
export FLASHVSR_IO_NODE_LIMIT_DIR="${FLASHVSR_IO_NODE_LIMIT_DIR:-/tmp/flashvsr_io_limiter}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"
export CONDUCTOR_CACHE_MAX_BYTES="${CONDUCTOR_CACHE_MAX_BYTES:-214748364800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FLASHVSR_STAGE3_DEBUG_LOSS="${FLASHVSR_STAGE3_DEBUG_LOSS:-1}"
export FLASHVSR_STAGE3C_NO_GATHER_LOG="${FLASHVSR_STAGE3C_NO_GATHER_LOG:-1}"
export FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG="${FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG:-0}"
export FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG:-0}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_3_lora_89f_videoonly_usmgtpretrain_dualengine_dfake5_offlinewandb.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_release_48gpu_v7_d4_3_lora_89f_videoonly_usmgtpretrain_dualengine_dfake5_offlinewandb}"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_6node48gpu_nooffload.template.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_3_lora.py"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0..5)}"

STAGE2_EXP="train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100"
RESUME_STAGE2_CHECKPOINT="${RESUME_STAGE2_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"
RESUME_STAGE2_S3="${RESUME_STAGE2_S3:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"

STAGE1_EXP="train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn"
STAGE3_REAL_CHECKPOINT="${STAGE3_REAL_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_EXP}/output/step-3000.safetensors}"
STAGE3_REAL_S3="${STAGE3_REAL_S3:-s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors}"
STAGE3_FAKE_CHECKPOINT="${STAGE3_FAKE_CHECKPOINT:-${STAGE3_REAL_CHECKPOINT}}"
STAGE3_FAKE_S3="${STAGE3_FAKE_S3:-${STAGE3_REAL_S3}}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
mkdir -p "${WANDB_DIR}/wandb"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

if [ "${MACHINE_RANK}" = "0" ]; then
  WANDB_OFFLINE_S3_URI="${WANDB_OFFLINE_S3_URI:-s3://lxh/tmp/wandb_offline/${RUN_NAME}.tar.gz}"
  tmux kill-session -t wandb_package_v7d43_dfake5 2>/dev/null || true
  tmux new-session -d -s wandb_package_v7d43_dfake5 \
    "cd /mnt/task_runtime/lucidvsr && RUN_DIR='${RUN_DIR}' TRAIN_PROCESS_PATTERN='train_flashvsr_stage3_v7_d4_3_lora.py' WANDB_OFFLINE_S3_URI='${WANDB_OFFLINE_S3_URI}' WANDB_PACKAGE_INTERVAL_SECONDS=3600 bash /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/package_wandb_offline_to_s3_loop.sh"
  echo "wandb_offline_package_tmux=wandb_package_v7d43_dfake5 interval_seconds=3600 run_dir=${RUN_DIR} s3_uri=${WANDB_OFFLINE_S3_URI}"
fi

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_video_train_all.txt" "${TAKANO_MANIFEST}"
fi

VGG16_PATH="${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
if [ ! -s "${VGG16_PATH}" ]; then
  mkdir -p "$(dirname "${VGG16_PATH}")"
  conductor s3 cp "s3://lxh/models/SR/vgg16-397923af.pth" "${VGG16_PATH}"
fi

if [ ! -s "${RESUME_STAGE2_CHECKPOINT}" ]; then
  mkdir -p "$(dirname "${RESUME_STAGE2_CHECKPOINT}")"
  conductor s3 cp "${RESUME_STAGE2_S3}" "${RESUME_STAGE2_CHECKPOINT}"
fi
if [ ! -s "${STAGE3_REAL_CHECKPOINT}" ]; then
  mkdir -p "$(dirname "${STAGE3_REAL_CHECKPOINT}")"
  conductor s3 cp "${STAGE3_REAL_S3}" "${STAGE3_REAL_CHECKPOINT}"
fi
if [ ! -s "${STAGE3_FAKE_CHECKPOINT}" ]; then
  mkdir -p "$(dirname "${STAGE3_FAKE_CHECKPOINT}")"
  conductor s3 cp "${STAGE3_FAKE_S3}" "${STAGE3_FAKE_CHECKPOINT}"
fi

echo "takano_manifest=${TAKANO_MANIFEST} lines=$(wc -l < "${TAKANO_MANIFEST}")"
echo "yubari_video_tar_url=conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "stage3_real_checkpoint=${STAGE3_REAL_CHECKPOINT}"
echo "stage3_fake_checkpoint=${STAGE3_FAKE_CHECKPOINT}"
echo "vgg16_cache=${VGG16_PATH} size=$(du -h "${VGG16_PATH}" | awk '{print $1}')"
echo "worker_setting=$(grep -E 'dataset_num_workers|dataloader_prefetch_factor|dataloader_persistent_workers' "${CONFIG_PATH}" | tr '\n' ' ')"
echo "stage3_v7_d4_3=dual_deepspeed_engine fake_fm_current_zpred_detach=1 dmd=1 shared_dmd_noisy_point usmgt_stage1_teacher_step3000 teacher_lq_trim_front_to_match spike_skip5 dfake_gen_update_ratio=5 validation=c6_stable"
echo "distributed_shape=6node48gpu"

ACCEL_YAML="${RUN_DIR}/accelerate_6node48gpu.yaml"
sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py" "${RUN_DIR}/snapshot/" || true
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
  --stage3_real_checkpoint "${STAGE3_REAL_CHECKPOINT}"
  --stage3_fake_checkpoint "${STAGE3_FAKE_CHECKPOINT}"
  --zero_init_lq_proj_in false
)
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
