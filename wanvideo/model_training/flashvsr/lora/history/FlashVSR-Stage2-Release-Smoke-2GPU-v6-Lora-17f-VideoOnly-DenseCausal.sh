#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/task_runtime/lucidvsr
cd "${REPO_ROOT}"

export PATH="/mnt/conda_envs/flashvsr/bin:${PATH}"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-4}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_densecausal.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2gpu_noactckpt.yaml}"
OUTPUT_TAG="${OUTPUT_TAG:-train_stage2_release_smoke_2gpu_v6_lora_17f_videoonly_densecausal}"
TRAIN_PY="${REPO_ROOT}/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py"
LQ_PROJ_CKPT="${LQ_PROJ_CKPT:-/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/output" "${RUN_DIR}/snapshot"

cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${REPO_ROOT}/diffsynth/models/wan_video_dit_stage2_v6.py" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --lq_proj_checkpoint "${LQ_PROJ_CKPT}"
)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
echo "RUN_DIR=${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1
"${CMD[@]}"

