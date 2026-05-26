#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/bin:/miniforge/bin:$PATH"
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
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

CONFIG_PATH="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm.yaml"
OUTPUT_TAG="train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm"
ACCEL_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2gpu_noactckpt.yaml"
TRAIN_PY="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py"

: "${RESUME_STAGE2_CHECKPOINT:?RESUME_STAGE2_CHECKPOINT must point to Stage2 v6 checkpoint, e.g. step-3300.safetensors}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"

exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

VGG16_PATH="${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
if [ ! -s "${VGG16_PATH}" ]; then
  mkdir -p "$(dirname "${VGG16_PATH}")"
  echo "vgg16_cache=missing path=${VGG16_PATH}; trying conductor s3 cp from s3://lxh/models/SR/vgg16-397923af.pth"
  conductor s3 cp "s3://lxh/models/SR/vgg16-397923af.pth" "${VGG16_PATH}" || true
fi
if [ -s "${VGG16_PATH}" ]; then
  echo "vgg16_cache=ready path=${VGG16_PATH} size=$(du -h "${VGG16_PATH}" | awk '{print $1}')"
else
  echo "vgg16_cache=missing_after_s3_pull path=${VGG16_PATH}; LPIPS may try online download if enabled"
fi

SMOKE_VIDEO_DIR="/mnt/task_wrapper/user_output/artifacts/data/stage3_smoke_videos"
if ! find "${SMOKE_VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | grep -q .; then
  mkdir -p "${SMOKE_VIDEO_DIR}"
  ffmpeg -y -f lavfi -i "testsrc2=size=1280x768:rate=8:duration=12" \
    -frames:v 96 -an -c:v libx264 -pix_fmt yuv420p "${SMOKE_VIDEO_DIR}/stage3_smoke_00.mp4"
  ffmpeg -y -f lavfi -i "smptebars=size=1280x768:rate=8:duration=12" \
    -frames:v 96 -an -c:v libx264 -pix_fmt yuv420p "${SMOKE_VIDEO_DIR}/stage3_smoke_01.mp4"
fi
echo "smoke_video_dir=${SMOKE_VIDEO_DIR} files=$(find "${SMOKE_VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "extra_args=${EXTRA_ARGS:-}"
echo "stage3_v7_b=one_step_student_gfake_scaffold_wan_decode_mse_lpips"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --resume_stage2_checkpoint "${RESUME_STAGE2_CHECKPOINT}"
  --zero_init_lq_proj_in false
)
if [ -n "${EXTRA_ARGS:-}" ]; then
  # Keep this simple: EXTRA_ARGS is intended for flag/value pairs without spaces.
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
