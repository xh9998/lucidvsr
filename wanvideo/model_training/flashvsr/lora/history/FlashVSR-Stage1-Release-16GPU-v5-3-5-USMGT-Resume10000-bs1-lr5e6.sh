#!/bin/bash
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

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml"
OUTPUT_TAG="train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000"
HISTORY_DIR="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2node16gpu_nooffload.template.yaml"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0 or 1)}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot" "${HISTORY_DIR}"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_20250205_test_4k_tar_manifest.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  if [ -s "/mnt/task_runtime/lucidvsr/wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt" ]; then
    cp "/mnt/task_runtime/lucidvsr/wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt" "${TAKANO_MANIFEST}"
  else
    conductor s3 cp "s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt" "${TAKANO_MANIFEST}"
  fi
fi
echo "takano_manifest=${TAKANO_MANIFEST} lines=$(wc -l < "${TAKANO_MANIFEST}")"

IMAGE_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt"
if [ ! -s "${IMAGE_MANIFEST}" ]; then
  mkdir -p "$(dirname "${IMAGE_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt" "${IMAGE_MANIFEST}"
fi
echo "image_manifest=${IMAGE_MANIFEST} lines=$(wc -l < "${IMAGE_MANIFEST}")"

STAGE1_EXP="train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300"
STAGE1_CKPT="/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors"
if [ ! -s "${STAGE1_CKPT}" ]; then
  mkdir -p "$(dirname "${STAGE1_CKPT}")"
  conductor s3 cp "s3://lxh/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors" "${STAGE1_CKPT}" || \
    conductor s3 cp "s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors" "${STAGE1_CKPT}"
fi
echo "resume_stage1_checkpoint=${STAGE1_CKPT}"

TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py"
SELF_SH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh"
ACCEL_YAML="${RUN_DIR}/accelerate_2node16gpu.yaml"

sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"
cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "${SELF_SH}" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
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
  --zero_init_lq_proj_in false)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
