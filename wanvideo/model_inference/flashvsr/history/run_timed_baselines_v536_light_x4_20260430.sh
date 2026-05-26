#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"

SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_20260430/lq}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x192_resizecrop_20260430}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/compare_timed_light_x4_flash_seedvr_v536_20260430_by_dataset}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
SEEDVR3B_MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}"
SEEDVR2_3B_MODEL_DIR="${SEEDVR2_3B_MODEL_DIR:-/mnt/models/SeedVR2-3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
V536_CKPT="${V536_CKPT:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200/output/step-2100.safetensors}"
V536_CKPT_S3="${V536_CKPT_S3:-s3://lxh/tmp/v536_step2100_nonstreamproj_aligned5.safetensors}"

FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_FRAMES="${NUM_FRAMES:-17}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"
RUN_FLASH="${RUN_FLASH:-1}"
RUN_SEEDVR3="${RUN_SEEDVR3:-1}"
RUN_SEEDVR2="${RUN_SEEDVR2:-1}"
RUN_V536="${RUN_V536:-1}"

mkdir -p "${OUTPUT_ROOT}/logs" "$(dirname "${V536_CKPT}")"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${V536_CKPT}" ]]; then
  conductor s3 cp "${V536_CKPT_S3}" "${V536_CKPT}"
fi

count_inputs() {
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

run_flashvsr_dataset() {
  local gpu="$1" dataset="$2" input_dir="$3"
  local out_dir="${OUTPUT_ROOT}/${dataset}/flashvsr_official"
  local log_file="${OUTPUT_ROOT}/logs/flashvsr_official_${dataset}.log"
  mkdir -p "${out_dir}"
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_DEVICE="${gpu}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${out_dir}" MODEL_DIR="${FLASHVSR_MODEL_DIR}" SCALE=4 SEED="${SEED}" \
    bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh" 2>&1 | tee "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  record_time "flashvsr_official" "${dataset}" "$((end - start))" "${status}" "${input_dir}" "${out_dir}"
  return "${status}"
}

run_seedvr_dataset() {
  local gpu="$1" model_kind="$2" label="$3" model_dir="$4" dataset="$5" input_dir="$6" res_h="$7" res_w="$8"
  local out_dir="${OUTPUT_ROOT}/${dataset}/${label}"
  local log_file="${OUTPUT_ROOT}/logs/${label}_${dataset}.log"
  mkdir -p "${out_dir}"
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_DEVICE="${gpu}" MODEL_KIND="${model_kind}" MODEL_DIR="${model_dir}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${out_dir}" \
    SEEDVR_PYTHON="${SEEDVR_PYTHON}" RES_H="${res_h}" RES_W="${res_w}" SEED="${SEED}" LOG_FILE="${log_file}" \
    MASTER_PORT="$((29531 + gpu))" \
    bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh" 2>&1 | tee "${OUTPUT_ROOT}/logs/${label}_${dataset}_launcher.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  record_time "${label}" "${dataset}" "$((end - start))" "${status}" "${input_dir}" "${out_dir}"
  return "${status}"
}

run_v536_dataset() {
  local gpu="$1" dataset="$2" input_dir="$3"
  local out_dir="${OUTPUT_ROOT}/${dataset}/v5_3_6_step2100_nonstream_aligned"
  local log_file="${OUTPUT_ROOT}/logs/v5_3_6_step2100_${dataset}.log"
  mkdir -p "${out_dir}"
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned_batch.py \
    --checkpoint_path "${V536_CKPT}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_dir "${input_dir}" \
    --output_dir "${out_dir}" \
    --height 768 \
    --width 1280 \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --device cuda \
    --torch_dtype "${TORCH_DTYPE}" \
    --lq_proj_layer_num 1 \
    --lq_proj_temporal_mode nonstreaming_aligned \
    --lq_proj_scale 1.0 \
    --projection_scale 1.0 \
    --input_bicubic_upscale 4.0 \
    --color_fix_method "${COLOR_FIX_METHOD}" \
    2>&1 | tee "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  record_time "v5_3_6_step2100_nonstream_aligned" "${dataset}" "$((end - start))" "${status}" "${input_dir}" "${out_dir}"
  return "${status}"
}

{
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "SYNTHETIC_INPUT_DIR=${SYNTHETIC_INPUT_DIR}"
  echo "REAL_INPUT_DIR=${REAL_INPUT_DIR}"
  echo "V536_CKPT=${V536_CKPT}"
  echo "V536 projector mode: nonstreaming_aligned, input_bicubic_upscale=4.0, num_frames=17"
  echo "FlashVSR/SeedVR input fps is inherited from test videos; generated test videos are fps=${FPS}."
} | tee "${OUTPUT_ROOT}/settings.txt"

pids=()
if [[ "${RUN_FLASH}" == "1" ]]; then
  (
    run_flashvsr_dataset 0 synthetic "${SYNTHETIC_INPUT_DIR}"
    run_flashvsr_dataset 0 real "${REAL_INPUT_DIR}"
  ) >"${OUTPUT_ROOT}/logs/flashvsr_official_all.log" 2>&1 &
  pids+=($!)
fi

if [[ "${RUN_SEEDVR3}" == "1" ]]; then
  (
    run_seedvr_dataset 1 seedvr1 seedvr3b "${SEEDVR3B_MODEL_DIR}" synthetic "${SYNTHETIC_INPUT_DIR}" 768 1280
    run_seedvr_dataset 1 seedvr1 seedvr3b "${SEEDVR3B_MODEL_DIR}" real "${REAL_INPUT_DIR}" 768 1280
  ) >"${OUTPUT_ROOT}/logs/seedvr3b_all.log" 2>&1 &
  pids+=($!)
fi

if [[ "${RUN_SEEDVR2}" == "1" ]]; then
  (
    run_seedvr_dataset 2 seedvr2 seedvr2_3b "${SEEDVR2_3B_MODEL_DIR}" synthetic "${SYNTHETIC_INPUT_DIR}" 768 1280
    run_seedvr_dataset 2 seedvr2 seedvr2_3b "${SEEDVR2_3B_MODEL_DIR}" real "${REAL_INPUT_DIR}" 768 1280
  ) >"${OUTPUT_ROOT}/logs/seedvr2_3b_all.log" 2>&1 &
  pids+=($!)
fi

if [[ "${RUN_V536}" == "1" ]]; then
  (
    run_v536_dataset 3 synthetic "${SYNTHETIC_INPUT_DIR}"
    run_v536_dataset 3 real "${REAL_INPUT_DIR}"
  ) >"${OUTPUT_ROOT}/logs/v5_3_6_step2100_all.log" 2>&1 &
  pids+=($!)
fi

for pid in "${pids[@]}"; do
  wait "${pid}"
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
cat "${OUTPUT_ROOT}"/logs/*.time | tee "${OUTPUT_ROOT}/timing_summary.txt"
conductor s3 sync "${OUTPUT_ROOT}" "s3://lxh/data/test/$(basename "${OUTPUT_ROOT}")"
echo "[done] ${OUTPUT_ROOT}"
