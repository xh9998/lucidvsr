#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_20260430}"
REAL_RAW_DIR="${REAL_RAW_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_raw_20260430}"
REAL_PROCESSED_DIR="${REAL_PROCESSED_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x192_resizecrop_20260430}"

TAKANO_MANIFEST="${TAKANO_MANIFEST:-/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt}"
YUBARI_URL="${YUBARI_URL:-conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/}"
DEGRADATION_CONFIG_PATH="${DEGRADATION_CONFIG_PATH:-${REPO_ROOT}/wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml}"

GT_HEIGHT="${GT_HEIGHT:-768}"
GT_WIDTH="${GT_WIDTH:-1280}"
REAL_HEIGHT="${REAL_HEIGHT:-192}"
REAL_WIDTH="${REAL_WIDTH:-320}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-20260430}"
NUM_PER_SOURCE="${NUM_PER_SOURCE:-5}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${TAKANO_MANIFEST}")" "${SYNTHETIC_ROOT}" "${REAL_PROCESSED_DIR}"

if [[ ! -s "${TAKANO_MANIFEST}" ]]; then
  conductor s3 cp s3://lxh/data/mainfest/takano_video_train_all.txt "${TAKANO_MANIFEST}"
fi

"${PYTHON_BIN}" wanvideo/data/flashvsr/tests/export_inference_testset6_aliyun_x4_lq.py \
  --output_root "${SYNTHETIC_ROOT}" \
  --takano_url "${TAKANO_MANIFEST}" \
  --yubari_url "${YUBARI_URL}" \
  --degradation_config_path "${DEGRADATION_CONFIG_PATH}" \
  --height "${GT_HEIGHT}" \
  --width "${GT_WIDTH}" \
  --num_frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --num_per_source "${NUM_PER_SOURCE}"

if [[ -d "${REAL_RAW_DIR}" ]]; then
  "${PYTHON_BIN}" wanvideo/data/flashvsr/tests/process_challenging_real_native_crop.py \
    --input_dir "${REAL_RAW_DIR}" \
    --output_dir "${REAL_PROCESSED_DIR}" \
    --width "${REAL_WIDTH}" \
    --height "${REAL_HEIGHT}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}"
else
  echo "[skip-real] REAL_RAW_DIR not found: ${REAL_RAW_DIR}" >&2
fi

conductor s3 sync "${SYNTHETIC_ROOT}" "s3://lxh/data/test/$(basename "${SYNTHETIC_ROOT}")"
if [[ -d "${REAL_PROCESSED_DIR}" ]]; then
  conductor s3 sync "${REAL_PROCESSED_DIR}" "s3://lxh/data/test/$(basename "${REAL_PROCESSED_DIR}")"
fi

echo "[done] synthetic=${SYNTHETIC_ROOT}"
echo "[done] real=${REAL_PROCESSED_DIR}"
