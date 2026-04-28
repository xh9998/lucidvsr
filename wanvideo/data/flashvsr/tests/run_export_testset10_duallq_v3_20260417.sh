#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_duallq_20260417}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
SEED="${SEED:-20260417}"
NUM_PER_SOURCE="${NUM_PER_SOURCE:-5}"
POOL_SIZE="${POOL_SIZE:-20}"
LQ_DOWNSAMPLE_FACTOR="${LQ_DOWNSAMPLE_FACTOR:-4}"
FLASHVSR_PARQUET_CACHE_DIR="${FLASHVSR_PARQUET_CACHE_DIR:-/mnt/task_runtime/flashvsr_cache/testset10_duallq_parquet}"

cd "${REPO_ROOT}"
mkdir -p "${FLASHVSR_PARQUET_CACHE_DIR}"
export FLASHVSR_PARQUET_CACHE_DIR
/mnt/conda_envs/flashvsr/bin/python wanvideo/data/flashvsr/tests/export_testset10_paired_v3.py \
  --output_root "${OUTPUT_ROOT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_per_source "${NUM_PER_SOURCE}" \
  --pool_size "${POOL_SIZE}" \
  --lq_downsample_factor "${LQ_DOWNSAMPLE_FACTOR}"
