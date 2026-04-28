#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset5_takano_x4_20260421}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-20260421}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"

cd "${ROOT_DIR}"

/mnt/conda_envs/flashvsr/bin/python wanvideo/data/flashvsr/tests/export_testset5_takano_x4_lq_v1.py \
  --output_root "${OUTPUT_ROOT}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_samples "${NUM_SAMPLES}"
