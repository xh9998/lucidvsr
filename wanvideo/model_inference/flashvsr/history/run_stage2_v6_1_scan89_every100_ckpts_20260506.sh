#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT_DIR="${CKPT_DIR:?CKPT_DIR is required}"
SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:?SYNTHETIC_INPUT_DIR is required}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:?REAL_INPUT_DIR is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-89}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
MIN_STEP="${MIN_STEP:-100}"
STEP_MOD="${STEP_MOD:-100}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"
SYNC_TO_S3="${SYNC_TO_S3:-0}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}/logs"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

mapfile -t CKPTS < <(
  find "${CKPT_DIR}" -maxdepth 1 -type f -name 'step-*.safetensors' \
    | sort -V \
    | while read -r ckpt; do
        base="$(basename "${ckpt}")"
        step="${base#step-}"
        step="${step%.safetensors}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= MIN_STEP )) && (( step % STEP_MOD == 0 )); then
          printf '%s\n' "${ckpt}"
        fi
      done
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[error] no checkpoints found in ${CKPT_DIR} with step >= ${MIN_STEP} and step % ${STEP_MOD} == 0" >&2
  exit 1
fi

run_one_dataset() {
  local gpu="$1" ckpt="$2" dataset="$3" input_dir="$4"
  local base out_dir log_file meta_file start end status count
  base="$(basename "${ckpt}" .safetensors)"
  out_dir="${OUTPUT_ROOT}/${dataset}/${base}"
  log_file="${OUTPUT_ROOT}/logs/${dataset}_${base}.log"
  meta_file="${OUTPUT_ROOT}/logs/${dataset}_${base}.time"
  mkdir -p "${out_dir}"
  count="$(find "${input_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1_batch \
    --checkpoint_path "${ckpt}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_dir "${input_dir}" \
    --output_dir "${out_dir}" \
    --height 768 \
    --width 1280 \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --device cuda \
    --torch_dtype "${TORCH_DTYPE}" \
    --lq_proj_scale 1.0 \
    --stage2_attention_mode block_sparse_chunk_causal \
    --stage2_topk_ratio 2.0 \
    --stage2_local_num -1 \
    --stage2_kv_ratio 3.0 \
    --input_bicubic_upscale 4.0 \
    --color_fix_method "${COLOR_FIX_METHOD}" \
    > "${log_file}" 2>&1
  status=$?
  set -e
  end="$(date +%s)"
  {
    echo "dataset=${dataset}"
    echo "checkpoint=${ckpt}"
    echo "status=${status}"
    echo "input_dir=${input_dir}"
    echo "output_dir=${out_dir}"
    echo "num_inputs=${count}"
    echo "seconds=$((end - start))"
    awk -v s="$((end - start))" -v n="${count}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "num_frames=${NUM_FRAMES}"
    echo "fps=${FPS}"
    echo "input_bicubic_upscale=4.0"
    echo "color_fix_method=${COLOR_FIX_METHOD}"
    echo "stage2_attention_mode=block_sparse_chunk_causal"
    echo "stage2_topk_ratio=2.0"
    echo "stage2_local_num=-1"
    echo "stage2_kv_ratio=3.0"
  } > "${meta_file}"
  return "${status}"
}

pkill -f infer_flashvsr_stage2_v6_1_batch || true
pkill -f gpu_stress_tc.py || true
sleep 2

echo "[info] ckpt_dir=${CKPT_DIR}"
echo "[info] num_ckpts=${#CKPTS[@]}"
echo "[info] synthetic_input_dir=${SYNTHETIC_INPUT_DIR}"
echo "[info] real_input_dir=${REAL_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"

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
    pids+=("$!")
    job_index=$((job_index + 1))
    if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
find "${OUTPUT_ROOT}/logs" -type f -name '*.time' | sort | while read -r f; do cat "${f}"; echo; done > "${OUTPUT_ROOT}/timing_summary.txt"

if [[ "${SYNC_TO_S3}" == "1" && -n "${S3_OUTPUT_DIR}" ]]; then
  conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"
fi

if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
  bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh >/tmp/occupy_after_stage2_v61_scan.log 2>&1 &
fi

echo "[done] status=${status} output_root=${OUTPUT_ROOT}"
exit "${status}"
