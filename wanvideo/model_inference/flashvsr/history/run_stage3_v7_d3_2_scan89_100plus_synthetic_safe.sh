#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage3_v7_d3_2_100plus_20260516}"
CKPT_S3_DIR="${CKPT_S3_DIR:-s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
SOURCE_INPUT_S3_DIR="${SOURCE_INPUT_S3_DIR:-s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7_d3_2_scan89_100plus_synthetic_20260516}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://lxh/data/test/stage3_v7_d3_2_scan89_100plus_synthetic_20260516}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
FPS="${FPS:-8}"
NUM_FRAMES="${NUM_FRAMES:-89}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
COLOR_FIX_METHOD="${COLOR_FIX_METHOD:-adain}"
TILED="${TILED:-0}"
GPU_LIST="${GPU_LIST:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
MIN_STEP="${MIN_STEP:-100}"
STEP_MOD="${STEP_MOD:-100}"
SYNC_TO_S3="${SYNC_TO_S3:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${CKPT_DIR}" "${SOURCE_INPUT_DIR}" "${OUTPUT_ROOT}/logs"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python_bin=${PYTHON_BIN}"
echo "[info] ckpt_dir=${CKPT_DIR}"
echo "[info] ckpt_s3_dir=${CKPT_S3_DIR}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] source_input_s3_dir=${SOURCE_INPUT_S3_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] s3_output_dir=${S3_OUTPUT_DIR}"
echo "[info] gpu_list=${GPU_LIST} max_parallel=${MAX_PARALLEL}"
echo "[info] tiled=${TILED}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] flashvsr python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m py_compile \
  wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py \
  wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d.py \
  wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py

input_count="$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
if [[ "${input_count}" -le 0 ]]; then
  echo "[info] local synthetic inputs missing; downloading from ${SOURCE_INPUT_S3_DIR}"
  conductor s3 sync "${SOURCE_INPUT_S3_DIR}" "${SOURCE_INPUT_DIR}"
  input_count="$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
fi
if [[ "${input_count}" -le 0 ]]; then
  echo "[error] no input mp4 found in ${SOURCE_INPUT_DIR}" >&2
  exit 1
fi
echo "[info] input_count=${input_count}"

ckpt_list="${OUTPUT_ROOT}/logs/stage3_v7_d3_2_s3_ckpt_list.txt"
conductor s3 ls "${CKPT_S3_DIR}/" | awk '{print $NF}' | grep -E '^step-[0-9]+\.safetensors$' | sort -V > "${ckpt_list}"
if [[ ! -s "${ckpt_list}" ]]; then
  echo "[error] no remote step-*.safetensors under ${CKPT_S3_DIR}" >&2
  exit 1
fi

while IFS= read -r file; do
  step="${file#step-}"
  step="${step%.safetensors}"
  if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= MIN_STEP )) && (( step % STEP_MOD == 0 )); then
    if [[ ! -s "${CKPT_DIR}/${file}" ]]; then
      echo "[download] ${file}"
      conductor s3 cp "${CKPT_S3_DIR}/${file}" "${CKPT_DIR}/${file}"
    else
      echo "[skip-download] ${file}"
    fi
  fi
done < "${ckpt_list}"

mapfile -t CKPTS < <(
  find "${CKPT_DIR}" -maxdepth 1 -type f -name 'step-*.safetensors' \
    | sort -V \
    | while read -r ckpt; do
        base="$(basename "${ckpt}")"
        step="${base#step-}"
        step="${step%.safetensors}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= MIN_STEP )) && (( step % STEP_MOD == 0 )); then
          out_dir="${OUTPUT_ROOT}/step-${step}"
          done_count=0
          if [[ -d "${out_dir}" ]]; then
            done_count="$(find "${out_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
          fi
          if (( done_count >= input_count )); then
            echo "[skip-tested] step-${step} count=${done_count}" >&2
          else
            printf '%s\n' "${ckpt}"
          fi
        fi
      done
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[info] no new checkpoints need testing"
  find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
  if [[ "${SYNC_TO_S3}" == "1" ]]; then
    conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"
  fi
  exit 0
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "[error] GPU_LIST is empty" >&2
  exit 1
fi

printf '%s\n' "${CKPTS[@]}" > "${OUTPUT_ROOT}/logs/ckpts_to_test.txt"
echo "[info] new_ckpts_to_test=${#CKPTS[@]}"
cat "${OUTPUT_ROOT}/logs/ckpts_to_test.txt"

run_one_ckpt() {
  local gpu="$1"
  local ckpt="$2"
  local step out_dir log_file meta_file start end status count
  step="$(basename "${ckpt}" .safetensors)"
  out_dir="${OUTPUT_ROOT}/${step}"
  log_file="${OUTPUT_ROOT}/logs/${step}.log"
  meta_file="${OUTPUT_ROOT}/logs/${step}.time"
  mkdir -p "${out_dir}"
  start="$(date +%s)"
  echo "[launch] ${step} cuda=${gpu}"
  extra_args=()
  if [[ "${TILED}" == "1" ]]; then
    extra_args+=(--tiled)
  fi
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage3_v7_d_batch \
      --checkpoint_path "${ckpt}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${SOURCE_INPUT_DIR}" \
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
      "${extra_args[@]}" \
      > "${log_file}" 2>&1
  status=$?
  set -e
  end="$(date +%s)"
  count="$(find "${out_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
  {
    echo "checkpoint=${ckpt}"
    echo "step=${step}"
    echo "status=${status}"
    echo "input_dir=${SOURCE_INPUT_DIR}"
    echo "output_dir=${out_dir}"
    echo "num_inputs=${input_count}"
    echo "num_outputs=${count}"
    echo "seconds=$((end - start))"
    awk -v s="$((end - start))" -v n="${input_count}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "num_frames=${NUM_FRAMES}"
    echo "fps=${FPS}"
    echo "num_inference_steps=${NUM_INFERENCE_STEPS}"
    echo "input_bicubic_upscale=4.0"
    echo "color_fix_method=${COLOR_FIX_METHOD}"
    echo "tiled=${TILED}"
    echo "stage2_attention_mode=block_sparse_chunk_causal"
    echo "stage2_topk_ratio=2.0"
    echo "stage2_local_num=-1"
    echo "stage2_kv_ratio=3.0"
  } > "${meta_file}"
  return "${status}"
}

status=0
pids=()
job_index=0
for ckpt in "${CKPTS[@]}"; do
  gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  (
    run_one_ckpt "${gpu}" "${ckpt}"
  ) &
  pids+=("$!")
  job_index=$((job_index + 1))
  if [[ "${#pids[@]}" -ge "${MAX_PARALLEL}" ]]; then
    if ! wait "${pids[0]}"; then
      status=1
    fi
    pids=("${pids[@]:1}")
  fi
done

for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print0 \
  | while IFS= read -r -d '' step_dir; do
      count="$(find "${step_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
      echo "$(basename "${step_dir}") ${count}"
    done | sort -V > "${OUTPUT_ROOT}/summary_counts.txt"
find "${OUTPUT_ROOT}/logs" -type f -name '*.time' | sort | while read -r f; do cat "${f}"; echo; done > "${OUTPUT_ROOT}/timing_summary.txt"

if [[ "${SYNC_TO_S3}" == "1" ]]; then
  echo "[info] syncing output to ${S3_OUTPUT_DIR}"
  conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"
fi

echo "[done] status=${status} output_root=${OUTPUT_ROOT}"
exit "${status}"
