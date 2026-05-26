#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH"
export PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"
if [ "${PARAMGRAD_SINGLEGPU:-0}" = "1" ]; then
  unset PYTHONNOUSERSITE
else
  export PYTHONNOUSERSITE=1
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_STAGE3_DEBUG_LOSS="${FLASHVSR_STAGE3_DEBUG_LOSS:-1}"
export FLASHVSR_STAGE3C_NO_GATHER_LOG="${FLASHVSR_STAGE3C_NO_GATHER_LOG:-1}"
export FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG:-0}"
export FLASHVSR_STAGE3_TIMING_DEBUG="${FLASHVSR_STAGE3_TIMING_DEBUG:-0}"
export FLASHVSR_STAGE3_FORCE_RECON_START="${FLASHVSR_STAGE3_FORCE_RECON_START:-0}"
export FLASHVSR_STAGE3_PARAM_GRAD_NO_ACTCKPT="${FLASHVSR_STAGE3_PARAM_GRAD_NO_ACTCKPT:-0}"
export FLASHVSR_STAGE3_PARAM_GRAD_DISABLE_DS="${FLASHVSR_STAGE3_PARAM_GRAD_DISABLE_DS:-0}"
export FLASHVSR_STAGE3_PARAM_GRAD_USE_LOCAL_SAFE="${FLASHVSR_STAGE3_PARAM_GRAD_USE_LOCAL_SAFE:-1}"
export FLASHVSR_STAGE3_DS_CONFIG="${FLASHVSR_STAGE3_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json}"
export FLASHVSR_STAGE3_PARAM_STATS_MAX_LORA_PARAMS="${FLASHVSR_STAGE3_PARAM_STATS_MAX_LORA_PARAMS:-12}"
export FLASHVSR_STAGE3_PARAM_STATS_MAX_LQ_PARAMS="${FLASHVSR_STAGE3_PARAM_STATS_MAX_LQ_PARAMS:-2}"
export FLASHVSR_STAGE3_PARAM_STATS_MAX_OTHER_PARAMS="${FLASHVSR_STAGE3_PARAM_STATS_MAX_OTHER_PARAMS:-0}"
export FLASHVSR_STAGE3_FAKE_DS_CONFIG="${FLASHVSR_STAGE3_FAKE_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload_fake_stable.json}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

: "${GRAD_CASE:?GRAD_CASE must be one of ParamFlow, ParamMSE, ParamLPIPS, ParamDMD}"

CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_paramgrad_2gpu_v7_d4_4_${GRAD_CASE}}"
ACCEL_YAML="${ACCEL_YAML:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2gpu_noactckpt.yaml}"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_paramgrad_lora.py"

STAGE2_EXP="train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100"
RESUME_STAGE2_CHECKPOINT="${RESUME_STAGE2_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"
RESUME_STAGE2_S3="${RESUME_STAGE2_S3:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"

STAGE1_USMGT_EXP="train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn"
STAGE3_REAL_CHECKPOINT="${STAGE3_REAL_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_USMGT_EXP}/output/step-3000.safetensors}"
STAGE3_REAL_S3="${STAGE3_REAL_S3:-s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors}"
STAGE3_FAKE_CHECKPOINT="${STAGE3_FAKE_CHECKPOINT:-${STAGE3_REAL_CHECKPOINT}}"
STAGE3_FAKE_S3="${STAGE3_FAKE_S3:-${STAGE3_REAL_S3}}"

case "${GRAD_CASE}" in
  ParamFlow)
    export FLASHVSR_STAGE3_PARAM_GRAD_SUBSET=0
    export FLASHVSR_STAGE3_PARAM_GRAD_AFTER_BACKWARD=1
    export FLASHVSR_STAGE3_PARAM_GRAD_SKIP_FAKE_WHEN_ZERO=1
    export FLASHVSR_STAGE3_PARAM_GRAD_LABEL=flow
    CASE_ARGS=(--stage3_flow_weight 1 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0 --max_train_steps 1)
    ;;
  ParamMSE)
    export FLASHVSR_STAGE3_PARAM_GRAD_SUBSET=0
    export FLASHVSR_STAGE3_PARAM_GRAD_AFTER_BACKWARD=1
    export FLASHVSR_STAGE3_PARAM_GRAD_SKIP_FAKE_WHEN_ZERO=1
    export FLASHVSR_STAGE3_PARAM_GRAD_LABEL=mse
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 1 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0 --max_train_steps 1)
    ;;
  ParamLPIPS)
    export FLASHVSR_STAGE3_PARAM_GRAD_SUBSET=0
    export FLASHVSR_STAGE3_PARAM_GRAD_AFTER_BACKWARD=1
    export FLASHVSR_STAGE3_PARAM_GRAD_SKIP_FAKE_WHEN_ZERO=1
    export FLASHVSR_STAGE3_PARAM_GRAD_LABEL=lpips
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 2 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0 --max_train_steps 1)
    ;;
  ParamDMD)
    export FLASHVSR_STAGE3_PARAM_GRAD_SUBSET=0
    export FLASHVSR_STAGE3_PARAM_GRAD_AFTER_BACKWARD=1
    export FLASHVSR_STAGE3_PARAM_GRAD_LABEL=dmd
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 1 --stage3_fake_fm_weight 1 --max_train_steps 2)
    ;;
  *)
    echo "Unsupported GRAD_CASE=${GRAD_CASE}" >&2
    exit 2
    ;;
esac

if [ "${PARAMGRAD_SMALL:-0}" = "1" ]; then
  CASE_ARGS+=(--height 256 --width 512 --num_frames 17 --max_source_frames 64)
fi

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
mkdir -p "${WANDB_DIR}/wandb"
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

echo "stage3_d44_paramgrad_2gpu case=${GRAD_CASE}"
echo "case_args=${CASE_ARGS[*]}"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "stage3_real_checkpoint=${STAGE3_REAL_CHECKPOINT}"
echo "stage3_fake_checkpoint=${STAGE3_FAKE_CHECKPOINT}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/original_config.yaml" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

if [ "${PARAMGRAD_SINGLEGPU:-0}" = "1" ]; then
  CMD=("${PYTHON_BIN}" "${TRAIN_PY}" --config "${CONFIG_PATH}"
  )
else
  CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
    --config_file "${ACCEL_YAML}"
    "${TRAIN_PY}"
    --config "${CONFIG_PATH}"
  )
fi
CMD+=(
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --validation_num_samples 0
  --save_steps 1000000
  --extra_save_steps ""
  --resume_stage2_checkpoint "${RESUME_STAGE2_CHECKPOINT}"
  --stage3_real_checkpoint "${STAGE3_REAL_CHECKPOINT}"
  --stage3_fake_checkpoint "${STAGE3_FAKE_CHECKPOINT}"
  --zero_init_lq_proj_in false
  "${CASE_ARGS[@]}"
)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
