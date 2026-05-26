#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

TAG="${TAG:-20260503}"
DATA_ROOT="${DATA_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference}"
REAL_RAW_DIR="${REAL_RAW_DIR:-${DATA_ROOT}/challenging_test_lxh_raw_${TAG}}"

TAKANO_MANIFEST="${TAKANO_MANIFEST:-/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt}"
YUBARI_URL="${YUBARI_URL:-conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/}"
DEGRADATION_CONFIG_PATH="${DEGRADATION_CONFIG_PATH:-${REPO_ROOT}/wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml}"

GT_HEIGHT="${GT_HEIGHT:-768}"
GT_WIDTH="${GT_WIDTH:-1280}"
LQ_HEIGHT="${LQ_HEIGHT:-192}"
LQ_WIDTH="${LQ_WIDTH:-320}"
FPS="${FPS:-8}"
SEED="${SEED:-20260503}"
NUM_PER_SOURCE="${NUM_PER_SOURCE:-5}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "$(dirname "${TAKANO_MANIFEST}")" "${DATA_ROOT}"

if [[ ! -s "${TAKANO_MANIFEST}" ]]; then
  conductor s3 cp s3://lxh/data/mainfest/takano_video_train_all.txt "${TAKANO_MANIFEST}"
fi

make_synthetic() {
  local frames="$1"
  local out_root="${DATA_ROOT}/testset10_${frames}f_aliyun_light_x4_lq_${TAG}"
  "${PYTHON_BIN}" wanvideo/data/flashvsr/tests/export_inference_testset20_fast_aliyun_x4_lq.py \
    --output_root "${out_root}" \
    --yubari_root "${YUBARI_URL}" \
    --degradation_config_path "${DEGRADATION_CONFIG_PATH}" \
    --height "${GT_HEIGHT}" \
    --width "${GT_WIDTH}" \
    --num_frames "${frames}" \
    --fps "${FPS}" \
    --seed "$((SEED + frames))" \
    --num_per_source "${NUM_PER_SOURCE}" \
    --max_records "${MAX_RECORDS:-2000}"
  mkdir -p "${out_root}/gt" "${out_root}/lq"
  find "${out_root}" -mindepth 3 -maxdepth 3 -type f -path '*/gt/*.mp4' -exec cp -f {} "${out_root}/gt/" \;
  find "${out_root}" -mindepth 3 -maxdepth 3 -type f -path '*/lq/*.mp4' -exec cp -f {} "${out_root}/lq/" \;
  conductor s3 sync "${out_root}" "s3://lxh/data/test/$(basename "${out_root}")"
}

make_real() {
  local frames="$1"
  local out_root="${DATA_ROOT}/challenging_test_lxh_${frames}f_${LQ_WIDTH}x${LQ_HEIGHT}_resizecrop_${TAG}"
  if [[ ! -d "${REAL_RAW_DIR}" ]]; then
    echo "[skip-real] REAL_RAW_DIR not found: ${REAL_RAW_DIR}" >&2
    return 0
  fi
  "${PYTHON_BIN}" wanvideo/data/flashvsr/tests/process_challenging_real_native_crop.py \
    --input_dir "${REAL_RAW_DIR}" \
    --output_dir "${out_root}" \
    --width "${LQ_WIDTH}" \
    --height "${LQ_HEIGHT}" \
    --num_frames "${frames}" \
    --fps "${FPS}"
  conductor s3 sync "${out_root}" "s3://lxh/data/test/$(basename "${out_root}")"
}

make_synthetic 17
make_synthetic 89
make_real 17
make_real 89

echo "[done] synthetic17=${DATA_ROOT}/testset10_17f_aliyun_light_x4_lq_${TAG}"
echo "[done] synthetic89=${DATA_ROOT}/testset10_89f_aliyun_light_x4_lq_${TAG}"
echo "[done] real17=${DATA_ROOT}/challenging_test_lxh_17f_${LQ_WIDTH}x${LQ_HEIGHT}_resizecrop_${TAG}"
echo "[done] real89=${DATA_ROOT}/challenging_test_lxh_89f_${LQ_WIDTH}x${LQ_HEIGHT}_resizecrop_${TAG}"
