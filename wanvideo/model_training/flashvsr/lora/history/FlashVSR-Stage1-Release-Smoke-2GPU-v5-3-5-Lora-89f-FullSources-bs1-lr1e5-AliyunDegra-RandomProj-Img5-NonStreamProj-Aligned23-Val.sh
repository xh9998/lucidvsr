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

REPO_ROOT=/mnt/task_runtime/lucidvsr
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_val.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml}
OUTPUT_TAG=${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_val}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${OUTPUT_TAG}_${RUN_TS}
RUN_DIR=/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}
mkdir -p "$RUN_DIR" "$RUN_DIR/output" "$RUN_DIR/snapshot"

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_video_train_all.txt" "${TAKANO_MANIFEST}"
fi
IMAGE_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt"
if [ ! -s "${IMAGE_MANIFEST}" ]; then
  mkdir -p "$(dirname "${IMAGE_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt" "${IMAGE_MANIFEST}"
fi

cp "$CONFIG_PATH" "$RUN_DIR/snapshot/" || true
cp "$0" "$RUN_DIR/snapshot/" || true
cp "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py" "$RUN_DIR/snapshot/" || true

nohup /mnt/conda_envs/flashvsr/bin/accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py" \
  --config "$CONFIG_PATH" \
  --output_path "$RUN_DIR/output" \
  --wandb_name "$RUN_NAME" \
  --zero_init_lq_proj_in true \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &

echo "RUN_DIR=$RUN_DIR"
