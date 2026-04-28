#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

REPO_ROOT=/mnt/task_runtime/lucidvsr
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v3_17f_takano_fullft_bs1_lr1e5_alpha5_nostartval.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_2gpu_noactckpt.yaml}
DS_CONFIG=${DS_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_noactckpt.json}
OUTPUT_TAG=${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v3_17f_takano_fullft_bs1_lr1e5_alpha5_nostartval_ds_noactckpt}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${OUTPUT_TAG}_${RUN_TS}
RUN_DIR=/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}

mkdir -p "$RUN_DIR" "$RUN_DIR/output" "$RUN_DIR/snapshot"

cp "$CONFIG_PATH" "$RUN_DIR/snapshot/$(basename "$CONFIG_PATH")"
cp "$ACCELERATE_CONFIG" "$RUN_DIR/snapshot/$(basename "$ACCELERATE_CONFIG")"
cp "$DS_CONFIG" "$RUN_DIR/snapshot/$(basename "$DS_CONFIG")"
cp "$0" "$RUN_DIR/snapshot/$(basename "$0")"
cp "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3.py" "$RUN_DIR/snapshot/train_flashvsr_stage1_v3.py"

cat > "$RUN_DIR/launch_command.sh" <<EOF
#!/usr/bin/env bash
cd $REPO_ROOT
conda activate flashvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
accelerate launch --config_file $ACCELERATE_CONFIG \
  wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3.py \
  --config $CONFIG_PATH \
  --output_path $RUN_DIR/output
EOF
chmod +x "$RUN_DIR/launch_command.sh"
cp "$RUN_DIR/launch_command.sh" "$RUN_DIR/launch_command.txt"

echo "Using Python: $(command -v python)"
echo "Run name: $RUN_NAME"
echo "Run dir: $RUN_DIR"

cd "$REPO_ROOT"
nohup accelerate launch --config_file "$ACCELERATE_CONFIG" \
  wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3.py \
  --config "$CONFIG_PATH" \
  --output_path "$RUN_DIR/output" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &

PID=$!
echo "PID=$PID"
echo "RUN_DIR=$RUN_DIR"
