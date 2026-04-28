#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/task_runtime/lucidvsr}
PYTHON_BIN=${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_3_lora_17f_fullsources.yaml}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$REPO_ROOT/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_4gpu_noactckpt.yaml}
OUTPUT_TAG=${OUTPUT_TAG:-flashvsr_worker_sweep_4gpu_v5_3_smoke}
WORKER_LIST=${WORKER_LIST:-"0 1 2 4"}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-12}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

RUN_TS=$(date +%Y%m%d_%H%M%S)
ROOT_DIR=/mnt/task_wrapper/user_output/artifacts/exp/${OUTPUT_TAG}_${RUN_TS}
mkdir -p "$ROOT_DIR"

echo "ROOT_DIR=$ROOT_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "ACCELERATE_CONFIG=$ACCELERATE_CONFIG"
echo "WORKER_LIST=$WORKER_LIST"
echo "MAX_TRAIN_STEPS=$MAX_TRAIN_STEPS"

for WORKERS in $WORKER_LIST; do
  RUN_NAME=${OUTPUT_TAG}_workers${WORKERS}_${RUN_TS}
  RUN_DIR=$ROOT_DIR/workers${WORKERS}
  mkdir -p "$RUN_DIR/output" "$RUN_DIR/snapshot"
  cp "$CONFIG_PATH" "$RUN_DIR/snapshot/" || true
  cp "$0" "$RUN_DIR/snapshot/" || true

  EXTRA_WORKER_ARGS=()
  if [[ "$WORKERS" -gt 0 ]]; then
    EXTRA_WORKER_ARGS+=(--dataloader_prefetch_factor "$PREFETCH_FACTOR")
    EXTRA_WORKER_ARGS+=(--dataloader_persistent_workers)
    EXTRA_WORKER_ARGS+=(--dataloader_multiprocessing_context spawn)
    EXTRA_WORKER_ARGS+=(--no-dataloader_in_order)
  fi

  echo "[worker-sweep] start workers=$WORKERS run_dir=$RUN_DIR"
  (
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
    "$PYTHON_BIN" -m accelerate.commands.launch \
      --config_file "$ACCELERATE_CONFIG" \
      "$REPO_ROOT/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py" \
      --config "$CONFIG_PATH" \
      --output_path "$RUN_DIR/output" \
      --wandb_name "$RUN_NAME" \
      --dataset_num_workers "$WORKERS" \
      "${EXTRA_WORKER_ARGS[@]}" \
      --max_train_steps "$MAX_TRAIN_STEPS" \
      --save_steps 1000000 \
      --extra_save_steps "" \
      --validation_num_samples 0
  ) > "$RUN_DIR/run.log" 2>&1
  echo "[worker-sweep] done workers=$WORKERS"
done

echo "[worker-sweep] all done root=$ROOT_DIR"
