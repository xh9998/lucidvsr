#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
RUN_TS=$(date +%Y%m%d_%H%M%S)
BASE_DIR=/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/takano_parquet_v2_${RUN_TS}
mkdir -p "$BASE_DIR"

METADATA_URL="conductor://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/,conductor://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/"

run_once() {
  local seed="$1"
  local tag="$2"
  local out_dir="${BASE_DIR}/${tag}"
  mkdir -p "$out_dir"
  echo "[run] seed=${seed} tag=${tag} out_dir=${out_dir}"
  "$PYTHON_BIN" wanvideo/data/flashvsr/tests/test_takano_parquet_dataset_v2.py \
    --metadata_url "$METADATA_URL" \
    --height 768 \
    --width 1280 \
    --num_frames 89 \
    --max_source_frames 160 \
    --global_seed "$seed" \
    --num_samples 3 \
    --max_parquet_records 256 \
    --save_dir "$out_dir" \
    2>&1 | tee "$out_dir/run.log"
}

run_once 1 seed1_run1
run_once 1 seed1_run2
run_once 2 seed2_run1

echo "BASE_DIR=${BASE_DIR}"
