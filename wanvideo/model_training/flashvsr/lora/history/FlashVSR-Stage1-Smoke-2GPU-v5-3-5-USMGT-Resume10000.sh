#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"
export CONDUCTOR_CACHE_MAX_BYTES="${CONDUCTOR_CACHE_MAX_BYTES:-214748364800}"

REPO_ROOT="/mnt/task_runtime/lucidvsr"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000}"
RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/output" "${RUN_DIR}/snapshot"

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
IMAGE_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt"
if [ ! -s "${IMAGE_MANIFEST}" ]; then
  mkdir -p "$(dirname "${IMAGE_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt" "${IMAGE_MANIFEST}"
fi

STAGE1_EXP="train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300"
STAGE1_CKPT="/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors"
if [ ! -s "${STAGE1_CKPT}" ]; then
  mkdir -p "$(dirname "${STAGE1_CKPT}")"
  conductor s3 cp "s3://lxh/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors" "${STAGE1_CKPT}" || \
    conductor s3 cp "s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE1_EXP}/output/step-10000.safetensors" "${STAGE1_CKPT}"
fi

TRAIN_PY="${REPO_ROOT}/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py"
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true

/mnt/conda_envs/flashvsr/bin/accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  "${TRAIN_PY}" \
  --config "${CONFIG_PATH}" \
  --output_path "${RUN_DIR}/output" \
  --wandb_name "${RUN_NAME}" \
  --zero_init_lq_proj_in false

echo "RUN_DIR=${RUN_DIR}"
