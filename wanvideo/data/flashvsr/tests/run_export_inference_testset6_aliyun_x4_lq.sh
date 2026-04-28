#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq_20260427}"
TAKANO_MANIFEST="${TAKANO_MANIFEST:-/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt}"
YUBARI_URL="${YUBARI_URL:-conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/}"
DEGRADATION_CONFIG_PATH="${DEGRADATION_CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1.yaml}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-20260427}"
NUM_PER_SOURCE="${NUM_PER_SOURCE:-3}"

cd "${REPO_ROOT}"
export PYTHONNOUSERSITE=1
mkdir -p "$(dirname "${TAKANO_MANIFEST}")" "${OUTPUT_ROOT}"
if [[ ! -s "${TAKANO_MANIFEST}" ]]; then
  conductor s3 cp s3://lxh/data/mainfest/takano_video_train_all.txt "${TAKANO_MANIFEST}"
fi

"${PYTHON_BIN}" wanvideo/data/flashvsr/tests/export_inference_testset6_aliyun_x4_lq.py \
  --output_root "${OUTPUT_ROOT}" \
  --takano_url "${TAKANO_MANIFEST}" \
  --yubari_url "${YUBARI_URL}" \
  --degradation_config_path "${DEGRADATION_CONFIG_PATH}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_per_source "${NUM_PER_SOURCE}"
