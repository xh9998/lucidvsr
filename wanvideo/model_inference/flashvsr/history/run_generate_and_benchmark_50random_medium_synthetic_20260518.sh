#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
DATA_S3="${DATA_S3:-s3://lxh/data/test/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_50random_medium_synthetic_20260518}"
OUTPUT_S3="${OUTPUT_S3:-s3://lxh/data/test/ppt_benchmark_50random_medium_synthetic_20260518}"
LOG_DIR="${LOG_DIR:-/mnt/task_wrapper/user_output/artifacts/logs}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
FINAL_OCCUPY_GPUS="${FINAL_OCCUPY_GPUS:-0,1,2,3,4,5,6,7}"
SPARE_OCCUPY_GPUS="${SPARE_OCCUPY_GPUS:-7}"
SEED="${SEED:-20260519}"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

echo "[stop_occupy] freeing GPUs for benchmark"
pkill -f '[g]pu_stress_tc.py' || true
pkill -f '[g]pu_stress_tc.sh' || true
tmux kill-session -t occupy 2>/dev/null || true
tmux kill-session -t occupy_pptbench50 2>/dev/null || true
tmux kill-session -t occupy_pptbench50_spare 2>/dev/null || true
sleep 3

echo "[check] scripts"
"${PYTHON_BIN}" -m py_compile wanvideo/data/flashvsr/tests/export_inference_testset50_random_takano_yubari_light_x4_lq.py
bash -n wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_50random_medium_synthetic_20260518.sh

if [[ ! -s "${DATA_ROOT}/summary.json" ]]; then
  echo "[generate] ${DATA_ROOT}"
  rm -rf "${DATA_ROOT}"
  "${PYTHON_BIN}" -u wanvideo/data/flashvsr/tests/export_inference_testset50_random_takano_yubari_light_x4_lq.py \
    --output_root "${DATA_ROOT}" \
    --num_frames 89 \
    --num_per_source 25 \
    --seed "${SEED}" \
    --degradation_config_path "/mnt/task_runtime/lucidvsr/wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_medium_x4test.yaml" \
    2>&1 | tee "${LOG_DIR}/export_testset50_random_medium_20260518.log"
else
  echo "[generate] existing summary found, skip generation: ${DATA_ROOT}/summary.json"
fi

echo "[sync_data] ${DATA_S3}"
conductor s3 sync "${DATA_ROOT}" "${DATA_S3}"

echo "[benchmark] ${OUTPUT_ROOT}"
GPU_LIST="${GPU_LIST}" \
FINAL_OCCUPY_GPUS="${FINAL_OCCUPY_GPUS}" \
SPARE_OCCUPY_GPUS="${SPARE_OCCUPY_GPUS}" \
SYNTHETIC_ROOT="${DATA_ROOT}" \
SYNTHETIC_S3="${DATA_S3}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OUTPUT_S3="${OUTPUT_S3}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_50random_medium_synthetic_20260518.sh \
  2>&1 | tee "${LOG_DIR}/ppt_benchmark_50random_medium_synthetic_20260518.log"

echo "[done] output=${OUTPUT_ROOT}"
