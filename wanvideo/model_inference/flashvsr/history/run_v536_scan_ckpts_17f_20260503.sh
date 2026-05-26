#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

TAG="${TAG:-20260503}"
CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_resume_step4600_seed20260501_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260502_012300/output}"
SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_${TAG}/lq}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x192_resizecrop_${TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/v536_scan17_${TAG}_by_ckpt}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-17}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mapfile -t CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'step-*.safetensors' | sort -V)
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[error] no step-*.safetensors found in ${CKPT_DIR}" >&2
  exit 1
fi

run_one_dataset() {
  local gpu="$1" ckpt="$2" dataset="$3" input_dir="$4"
  local step base out_dir log_file start end status count
  base="$(basename "${ckpt}" .safetensors)"
  step="${base#step-}"
  out_dir="${OUTPUT_ROOT}/${base}/${dataset}/v5_3_6_nonstream_aligned_colorfix"
  log_file="${OUTPUT_ROOT}/logs/${base}_${dataset}.log"
  mkdir -p "${out_dir}"
  count="$(find "${input_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned_batch.py \
    --checkpoint_path "${ckpt}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_dir "${input_dir}" \
    --output_dir "${out_dir}" \
    --height 768 \
    --width 1280 \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --seed 0 \
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
  {
    echo "step=${step}"
    echo "dataset=${dataset}"
    echo "status=${status}"
    echo "seconds=$((end - start))"
    echo "num_inputs=${count}"
    awk -v s="$((end - start))" -v n="${count}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "checkpoint=${ckpt}"
    echo "input_dir=${input_dir}"
    echo "output_dir=${out_dir}"
    echo "color_fix=${COLOR_FIX_METHOD}"
    echo "input_bicubic_upscale=4.0"
  } | tee "${OUTPUT_ROOT}/logs/${base}_${dataset}.time"
  return "${status}"
}

pids=()
job_index=0
for ckpt in "${CKPTS[@]}"; do
  for dataset in synthetic real; do
    if [[ "${dataset}" == "synthetic" ]]; then
      input_dir="${SYNTHETIC_INPUT_DIR}"
    else
      input_dir="${REAL_INPUT_DIR}"
    fi
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    (
      run_one_dataset "${gpu}" "${ckpt}" "${dataset}" "${input_dir}"
    ) &
    pids+=($!)
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
cat "${OUTPUT_ROOT}"/logs/*.time > "${OUTPUT_ROOT}/timing_summary.txt"
conductor s3 sync "${OUTPUT_ROOT}" "s3://lxh/data/test/$(basename "${OUTPUT_ROOT}")"

if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
  echo "[occupy] starting gpu_stress_tc.sh"
  bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh || true
fi

echo "[done] ${OUTPUT_ROOT}"
