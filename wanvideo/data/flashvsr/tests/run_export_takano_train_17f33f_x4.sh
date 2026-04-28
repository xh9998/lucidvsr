#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/takano_train_17f33f_x4_20260422}"
METADATA_URL="${METADATA_URL:-s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/,s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/}"
DEGRADATION_CONFIG_PATH="${DEGRADATION_CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/data/flashvsr/degradation/configs/params_realesrgan_with_second.yaml}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES_FULL="${NUM_FRAMES_FULL:-33}"
NUM_FRAMES_SHORT="${NUM_FRAMES_SHORT:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-20260422}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_PARQUET_RECORDS="${MAX_PARQUET_RECORDS:-512}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

echo "Using Python: ${PYTHON_BIN}"

"${PYTHON_BIN}" \
  wanvideo/data/flashvsr/tests/export_takano_train_17f33f_x4.py \
  --output_root "${OUTPUT_ROOT}" \
  --metadata_url "${METADATA_URL}" \
  --degradation_config_path "${DEGRADATION_CONFIG_PATH}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames_full "${NUM_FRAMES_FULL}" \
  --num_frames_short "${NUM_FRAMES_SHORT}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_parquet_records "${MAX_PARQUET_RECORDS}"
