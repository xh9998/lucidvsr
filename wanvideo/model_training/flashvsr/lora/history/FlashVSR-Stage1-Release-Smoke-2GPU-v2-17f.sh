#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

REPO_DIR="/mnt/task_runtime/lucidvsr"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu_v2_17f_bs4.yaml}"
ACCEL_CONFIG_SRC="$REPO_DIR/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml"
DEEPSPEED_CONFIG_ABS="$REPO_DIR/wanvideo/model_training/flashvsr/lora/deepspeed_zero2_flashvsr.json"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${OUTPUT_TAG:-$(basename "${CONFIG_PATH%.yaml}")}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "$RUN_DIR" "$RUN_DIR/output" "$RUN_DIR/snapshot"
ACCEL_CONFIG_TMP="$RUN_DIR/accelerate_2gpu_resolved.yaml"

echo "Using Python: ${PYTHON_BIN}"
echo "Run name: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Main process port: ${MAIN_PROCESS_PORT}"

python - <<PY
from pathlib import Path
src = Path("$ACCEL_CONFIG_SRC")
dst = Path("$ACCEL_CONFIG_TMP")
text = src.read_text(encoding="utf-8")
text = text.replace("wanvideo/model_training/flashvsr/lora/deepspeed_zero2_flashvsr.json", "$DEEPSPEED_CONFIG_ABS")
dst.write_text(text, encoding="utf-8")
print(dst)
PY

cp "$REPO_DIR/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py" "$RUN_DIR/snapshot/" || true
cp "$CONFIG_PATH" "$RUN_DIR/snapshot/" || true
cp "$ACCEL_CONFIG_TMP" "$RUN_DIR/snapshot/" || true
cp "$0" "$RUN_DIR/snapshot/" || true

printf '%q ' /mnt/conda_envs/flashvsr/bin/accelerate launch \
  --config_file "$ACCEL_CONFIG_TMP" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$REPO_DIR/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py" \
  --config "$CONFIG_PATH" \
  --output_path "$RUN_DIR/output" \
  > "$RUN_DIR/launch_command.txt"
cp "$RUN_DIR/launch_command.txt" "$RUN_DIR/launch_command.sh"
chmod +x "$RUN_DIR/launch_command.sh"

/mnt/conda_envs/flashvsr/bin/accelerate launch \
  --config_file "$ACCEL_CONFIG_TMP" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  "$REPO_DIR/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py" \
  --config "$CONFIG_PATH" \
  --output_path "$RUN_DIR/output" \
  2>&1 | tee "$RUN_DIR/run.log"
