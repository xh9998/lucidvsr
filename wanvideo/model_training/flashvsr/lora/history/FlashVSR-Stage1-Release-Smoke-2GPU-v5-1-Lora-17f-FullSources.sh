#!/usr/bin/env bash
set -euo pipefail
FLASHVSR_ENV_PREFIX=${FLASHVSR_ENV_PREFIX:-/mnt/conda_envs/flashvsr}
export PATH="$FLASHVSR_ENV_PREFIX/bin:$PATH"
REPO_ROOT=/mnt/task_runtime/lucidvsr
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_1_lora_17f_fullsources.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml}
OUTPUT_TAG=${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v5_1_lora_17f_fullsources}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${OUTPUT_TAG}_${RUN_TS}
RUN_DIR=/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}
mkdir -p "$RUN_DIR" "$RUN_DIR/output" "$RUN_DIR/snapshot"
cp "$CONFIG_PATH" "$RUN_DIR/snapshot/" || true
cp "$0" "$RUN_DIR/snapshot/" || true
cp "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_1_lora.py" "$RUN_DIR/snapshot/" || true
(
  cd "$REPO_ROOT"
  nohup "$FLASHVSR_ENV_PREFIX/bin/accelerate" launch --config_file "$ACCELERATE_CONFIG" "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_1_lora.py" --config "$CONFIG_PATH" --output_path "$RUN_DIR/output" --wandb_name "$RUN_NAME" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
)
echo "RUN_DIR=$RUN_DIR"
