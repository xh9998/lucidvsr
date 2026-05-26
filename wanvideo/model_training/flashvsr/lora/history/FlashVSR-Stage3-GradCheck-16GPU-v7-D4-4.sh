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
export FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG:-1}"
export FLASHVSR_STAGE3_GRAD_SCALE_DEBUG="${FLASHVSR_STAGE3_GRAD_SCALE_DEBUG:-0}"
export FLASHVSR_STAGE3_TIMING_DEBUG="${FLASHVSR_STAGE3_TIMING_DEBUG:-1}"
export FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG="${FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG:-0}"
export FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG:-0}"
export FLASHVSR_STAGE3_FORCE_RECON_START="${FLASHVSR_STAGE3_FORCE_RECON_START:-0}"
export FLASHVSR_STAGE3_DS_CONFIG="${FLASHVSR_STAGE3_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${MASTER_PORT:=29500}"
: "${MACHINE_RANK:?MACHINE_RANK must be set (0..1)}"
: "${GRAD_CASE:?GRAD_CASE must be one of Ownership, Pixel, DMD, Fake, All, FakeLoraOnly, FakeLqProjOnly, GradScale}"

CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_gradcheck_16gpu_v7_d4_4_${GRAD_CASE}}"
TEMPLATE_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2node16gpu_nooffload.template.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py"

STAGE2_EXP="train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100"
RESUME_STAGE2_CHECKPOINT="${RESUME_STAGE2_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"
RESUME_STAGE2_S3="${RESUME_STAGE2_S3:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"

STAGE1_USMGT_EXP="train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn"
STAGE3_REAL_CHECKPOINT="${STAGE3_REAL_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_USMGT_EXP}/output/step-3000.safetensors}"
STAGE3_REAL_S3="${STAGE3_REAL_S3:-s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors}"
STAGE3_FAKE_CHECKPOINT="${STAGE3_FAKE_CHECKPOINT:-${STAGE3_REAL_CHECKPOINT}}"
STAGE3_FAKE_S3="${STAGE3_FAKE_S3:-${STAGE3_REAL_S3}}"

case "${GRAD_CASE}" in
  Ownership|All)
    CASE_ARGS=(--stage3_flow_weight 1 --stage3_mse_weight 1 --stage3_lpips_weight 2 --stage3_dmd_weight 1 --stage3_fake_fm_weight 1 --max_train_steps 2)
    ;;
  Pixel)
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 1 --stage3_lpips_weight 2 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0 --max_train_steps 1)
    ;;
  DMD)
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 1 --stage3_fake_fm_weight 0 --max_train_steps 1)
    ;;
  Fake)
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 1 --max_train_steps 1)
    ;;
  FakeLoraOnly)
    export FLASHVSR_STAGE3_FAKE_TRAINABLE_FILTER=lora_only
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 1 --max_train_steps 1)
    ;;
  FakeLqProjOnly)
    export FLASHVSR_STAGE3_FAKE_TRAINABLE_FILTER=lq_proj_only
    CASE_ARGS=(--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 1 --max_train_steps 1)
    ;;
  GradScale)
    export FLASHVSR_STAGE3_GRAD_SCALE_DEBUG=1
    CASE_ARGS=(--stage3_flow_weight 1 --stage3_mse_weight 1 --stage3_lpips_weight 2 --stage3_dmd_weight 1 --stage3_fake_fm_weight 1 --max_train_steps 2)
    ;;
  *)
    echo "Unsupported GRAD_CASE=${GRAD_CASE}" >&2
    exit 2
    ;;
esac

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
mkdir -p "${WANDB_DIR}/wandb"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

GRAD_CONFIG="${RUN_DIR}/gradcheck_config.yaml"
"${PYTHON_BIN}" - "${CONFIG_PATH}" "${GRAD_CONFIG}" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file) or {}
cfg.setdefault("wandb", {})["use_wandb"] = False
cfg.setdefault("validation", {})["validation_num_samples"] = 0
cfg.setdefault("train", {})["save_steps"] = 1000000
cfg.setdefault("train", {})["extra_save_steps"] = ""
cfg.setdefault("train", {})["max_train_steps"] = 2
with open(dst, "w", encoding="utf-8") as file:
    yaml.safe_dump(cfg, file, sort_keys=False)
PY

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

echo "stage3_d44_gradcheck case=${GRAD_CASE}"
echo "case_args=${CASE_ARGS[*]}"
echo "fake_trainable_filter=${FLASHVSR_STAGE3_FAKE_TRAINABLE_FILTER:-all}"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "stage3_real_checkpoint=${STAGE3_REAL_CHECKPOINT}"
echo "stage3_fake_checkpoint=${STAGE3_FAKE_CHECKPOINT}"

ACCEL_YAML="${RUN_DIR}/accelerate_2node16gpu.yaml"
sed -e "s/__MASTER_ADDR__/${MASTER_ADDR}/g" -e "s/__MASTER_PORT__/${MASTER_PORT}/g" -e "s/__MACHINE_RANK__/${MACHINE_RANK}/g" "${TEMPLATE_YAML}" > "${ACCEL_YAML}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/original_config.yaml" || true
cp "${GRAD_CONFIG}" "${RUN_DIR}/snapshot/gradcheck_config.yaml" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
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
  --config "${GRAD_CONFIG}"
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
