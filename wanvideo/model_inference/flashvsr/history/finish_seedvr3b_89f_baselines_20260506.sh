#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/baselines_flash_seedvr3b_17f89f_20260506_by_dataset}"
SYNTHETIC_89_INPUT_DIR="${SYNTHETIC_89_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
REAL_89_INPUT_DIR="${REAL_89_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503}"
SEEDVR3B_MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
SYNC_TO_S3="${SYNC_TO_S3:-1}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

count_inputs() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

count_outputs() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

record_time() {
  local method="$1"
  local dataset="$2"
  local seconds="$3"
  local status="$4"
  local input_dir="$5"
  local output_dir="$6"
  local count
  count="$(count_inputs "${input_dir}")"
  {
    echo "method=${method}"
    echo "dataset=${dataset}"
    echo "status=${status}"
    echo "seconds=${seconds}"
    echo "num_inputs=${count}"
    awk -v s="${seconds}" -v n="${count}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "input_dir=${input_dir}"
    echo "output_dir=${output_dir}"
    echo "fps=${FPS}"
  } | tee "${OUTPUT_ROOT}/logs/${method}_${dataset}.time"
}

run_seedvr3b_dataset() {
  local dataset="$1" input_dir="$2" gpu="$3"
  local out_dir="${OUTPUT_ROOT}/${dataset}/seedvr3b"
  local log_file="${OUTPUT_ROOT}/logs/seedvr3b_${dataset}.log"
  local start end status num_in num_out
  mkdir -p "${out_dir}"
  start="$(date +%s)"
  set +e
  CUDA_DEVICE="${gpu}" MODEL_KIND="seedvr1" MODEL_DIR="${SEEDVR3B_MODEL_DIR}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${out_dir}" \
    SEEDVR_PYTHON="${SEEDVR_PYTHON}" RES_H=768 RES_W=1280 SEED="${SEED}" LOG_FILE="${log_file}" OUT_FPS="${FPS}" \
    MASTER_PORT="$((29581 + gpu))" \
    bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh" 2>&1 | tee "${OUTPUT_ROOT}/logs/seedvr3b_${dataset}_launcher.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  num_in="$(count_inputs "${input_dir}")"
  num_out="$(count_outputs "${out_dir}")"
  if [[ "${status}" != "0" && "${num_out}" == "${num_in}" && "${num_in}" != "0" ]]; then
    echo "[warn] ${dataset}: SeedVR exited ${status}, but outputs are complete (${num_out}/${num_in}); accepting result." | tee -a "${OUTPUT_ROOT}/logs/seedvr3b_${dataset}_launcher.log"
    status=0
  fi
  record_time "seedvr3b" "${dataset}" "$((end - start))" "${status}" "${input_dir}" "${out_dir}"
  return "${status}"
}

run_seedvr3b_dataset synthetic_89f "${SYNTHETIC_89_INPUT_DIR}" 0 &
pid0=$!
run_seedvr3b_dataset real_89f "${REAL_89_INPUT_DIR}" 1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
cat "${OUTPUT_ROOT}"/logs/*.time | tee "${OUTPUT_ROOT}/timing_summary.txt"

if [[ "${SYNC_TO_S3}" == "1" ]]; then
  conductor s3 sync "${OUTPUT_ROOT}" "s3://lxh/data/test/$(basename "${OUTPUT_ROOT}")"
fi

if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
  echo "[occupy] starting gpu_stress_tc.sh"
  pkill -f gpu_stress_tc.py || true
  pkill -f gpu_stress_tc.sh || true
  sleep 2
  bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh || true
fi

echo "[done] ${OUTPUT_ROOT}"
