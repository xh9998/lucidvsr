#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
RUN_TS=$(date +%Y%m%d_%H%M%S)
BASE_DIR=/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/takano_seed_repro_${RUN_TS}
mkdir -p "$BASE_DIR"
export CUDA_VISIBLE_DEVICES=""

INTERNAL_URL="conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier1-qwen3/1080p/,conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier2-qwen3/1080p/"

run_once() {
  local seed="$1"
  local tag="$2"
  local out_dir="${BASE_DIR}/${tag}"
  mkdir -p "$out_dir"
  echo "[run] seed=${seed} tag=${tag} out_dir=${out_dir}"
  "$PYTHON_BIN" wanvideo/data/flashvsr/tests/test_takano_seed_repro.py \
    --internal_url "$INTERNAL_URL" \
    --height 768 \
    --width 1280 \
    --num_frames 89 \
    --max_source_frames 160 \
    --global_seed "$seed" \
    --num_samples 3 \
    --save_dir "$out_dir" \
    2>&1 | tee "$out_dir/run.log"
}

run_once 1 seed1_run1
run_once 1 seed1_run2
run_once 2 seed2_run1

echo "BASE_DIR=${BASE_DIR}"
