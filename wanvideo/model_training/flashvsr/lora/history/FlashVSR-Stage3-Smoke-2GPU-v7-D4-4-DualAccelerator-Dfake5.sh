#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_STAGE3_DEBUG_LOSS=1
export FLASHVSR_STAGE3C_NO_GATHER_LOG=1
export FLASHVSR_STAGE3C_DMD_DEBUG="${FLASHVSR_STAGE3C_DMD_DEBUG:-0}"
export FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG="${FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG:-0}"
export FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG:-0}"
export FLASHVSR_STAGE3_FORCE_RECON_START="${FLASHVSR_STAGE3_FORCE_RECON_START:-0}"
export FLASHVSR_STAGE3_DS_CONFIG="${FLASHVSR_STAGE3_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json}"
export FLASHVSR_STAGE3_FAKE_DS_CONFIG="${FLASHVSR_STAGE3_FAKE_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload_fake_stable.json}"
export FLASHVSR_TRAIN_DEBUG="${FLASHVSR_TRAIN_DEBUG:-1}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5}"
ACCEL_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2gpu_noactckpt.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py"

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
export FLASHVSR_DEBUG_DIR="${FLASHVSR_DEBUG_DIR:-${RUN_DIR}/debug}"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

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

echo "stage3_v7_d4_4_dualaccelerator_dfake5 force_recon_start=${FLASHVSR_STAGE3_FORCE_RECON_START}"
echo "v7-D4.4 uses dual Accelerate DeepSpeed engines: fake updates every runner step from current z_pred.detach(); student updates every 5 runner steps"
echo "stage3_ds_config=${FLASHVSR_STAGE3_DS_CONFIG}"
echo "stage3_fake_ds_config=${FLASHVSR_STAGE3_FAKE_DS_CONFIG}"
echo "flash_debug_dir=${FLASHVSR_DEBUG_DIR}"
echo "takano_manifest=${TAKANO_MANIFEST} lines=$(wc -l < "${TAKANO_MANIFEST}")"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "stage3_real_checkpoint=${STAGE3_REAL_CHECKPOINT}"
echo "stage3_fake_checkpoint=${STAGE3_FAKE_CHECKPOINT}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/diffsynth/models/wan_video_dit_stage2_v6_1.py" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --resume_stage2_checkpoint "${RESUME_STAGE2_CHECKPOINT}"
  --stage3_real_checkpoint "${STAGE3_REAL_CHECKPOINT}"
  --stage3_fake_checkpoint "${STAGE3_FAKE_CHECKPOINT}"
  --zero_init_lq_proj_in false
)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
