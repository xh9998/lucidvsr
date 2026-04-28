#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

REPO_ROOT=/mnt/task_runtime/lucidvsr
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v4_lora_tarv3_17f_yubari_picked17k_1to1.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/accelerate_zero2_flashvsr_2gpu.yaml}
OUTPUT_TAG=${OUTPUT_TAG:-train_stage1_release_smoke_2gpu_v4_lora_tarv3_17f_yubari_picked17k_1to1}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${OUTPUT_TAG}_${RUN_TS}
RUN_DIR=/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}

mkdir -p "$RUN_DIR" "$RUN_DIR/output" "$RUN_DIR/snapshot"

cp "$CONFIG_PATH" "$RUN_DIR/snapshot/$(basename "$CONFIG_PATH")"
cp "$ACCELERATE_CONFIG" "$RUN_DIR/snapshot/$(basename "$ACCELERATE_CONFIG")"
cp "$0" "$RUN_DIR/snapshot/$(basename "$0")"
cp "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py" "$RUN_DIR/snapshot/train_flashvsr_stage1_v4_lora.py"
cp "$REPO_ROOT/wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v3.py" "$RUN_DIR/snapshot/tar_streaming_dataset_v3.py"
cp "$REPO_ROOT/wanvideo/data/flashvsr/datasets/joint_batching_v1.py" "$RUN_DIR/snapshot/joint_batching_v1.py"
cp "$REPO_ROOT/diffsynth/models/wan_video_dit_joint_v1.py" "$RUN_DIR/snapshot/wan_video_dit_joint_v1.py"

cat > "$RUN_DIR/launch_command.sh" <<EOF
#!/usr/bin/env bash
cd $REPO_ROOT
conda activate flashvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
accelerate launch --config_file $ACCELERATE_CONFIG \
  wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py \
  --config $CONFIG_PATH \
  --output_path $RUN_DIR/output \
  --wandb_name $RUN_NAME
EOF
chmod +x "$RUN_DIR/launch_command.sh"
cp "$RUN_DIR/launch_command.sh" "$RUN_DIR/launch_command.txt"

echo "Using Python: $(command -v python)"
echo "Run name: $RUN_NAME"
echo "Run dir: $RUN_DIR"

cd "$REPO_ROOT"
nohup accelerate launch --config_file "$ACCELERATE_CONFIG" \
  wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py \
  --config "$CONFIG_PATH" \
  --output_path "$RUN_DIR/output" \
  --wandb_name "$RUN_NAME" \
  > "$RUN_DIR/run.log" 2>&1 < /dev/null &

PID=$!
echo "PID=$PID"
echo "RUN_DIR=$RUN_DIR"
