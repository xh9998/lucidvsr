#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_early_20260510}"
CKPT_S3_DIR="${CKPT_S3_DIR:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_early_ckpts_v61_20260510}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://lxh/tmp/stage2_v6_early_ckpts_v61_20260510}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
EARLY_STEPS="${EARLY_STEPS:-10 50 100 500 1000 1500 2000 2500 3000 3500 4000 4500 5000}"
MAX_CKPTS_PER_WAVE="${MAX_CKPTS_PER_WAVE:-4}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${CKPT_DIR}" "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_ROOT}/_input_splits/part0" "${OUTPUT_ROOT}/_input_splits/part1"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python_bin=${PYTHON_BIN}"
echo "[info] ckpt_s3_dir=${CKPT_S3_DIR}"
echo "[info] ckpt_dir=${CKPT_DIR}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] early_steps=${EARLY_STEPS}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] flashvsr python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

tmux kill-session -t occupy 2>/dev/null || true
pkill -f infer_flashvsr_stage2_v6_1_batch || true
pkill -f gpu_stress_tc.py || true
sleep 2

rm -f "${OUTPUT_ROOT}/_input_splits/part0"/*.mp4 "${OUTPUT_ROOT}/_input_splits/part1"/*.mp4
idx=0
while IFS= read -r input; do
  part=$((idx % 2))
  ln -sf "${input}" "${OUTPUT_ROOT}/_input_splits/part${part}/$(basename "${input}")"
  idx=$((idx + 1))
done < <(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort)

EXPECTED_INPUTS="$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
if [[ "${EXPECTED_INPUTS}" -le 0 ]]; then
  echo "[error] no input mp4 found in ${SOURCE_INPUT_DIR}" >&2
  exit 1
fi

for step in ${EARLY_STEPS}; do
  file="step-${step}.safetensors"
  if [[ ! -s "${CKPT_DIR}/${file}" ]]; then
    echo "[download] ${file}"
    conductor s3 cp "${CKPT_S3_DIR}/${file}" "${CKPT_DIR}/${file}"
  else
    echo "[skip-download] ${file}"
  fi
done

mapfile -t CKPTS < <(
  for step in ${EARLY_STEPS}; do
    ckpt="${CKPT_DIR}/step-${step}.safetensors"
    [[ -s "${ckpt}" ]] || continue
    out_dir="${OUTPUT_ROOT}/step-${step}"
    count=0
    if [[ -d "${out_dir}" ]]; then
      count="$(find "${out_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
    fi
    if (( count >= EXPECTED_INPUTS )); then
      echo "[skip-tested] step-${step} count=${count}" >&2
    else
      printf '%s\n' "${ckpt}"
    fi
  done
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[info] no new checkpoints need testing"
  find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
  conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"
  if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
    tmux new-session -d -s occupy "${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"
  fi
  exit 0
fi

printf '%s\n' "${CKPTS[@]}" > "${OUTPUT_ROOT}/logs/early_ckpts_to_test.txt"
echo "[info] new_ckpts_to_test=${#CKPTS[@]}"
cat "${OUTPUT_ROOT}/logs/early_ckpts_to_test.txt"

run_one_part() {
  local gpu="$1"
  local ckpt="$2"
  local part="$3"
  local step out_dir input_dir log_file
  step="$(basename "${ckpt}" .safetensors)"
  out_dir="${OUTPUT_ROOT}/${step}"
  input_dir="${OUTPUT_ROOT}/_input_splits/part${part}"
  log_file="${OUTPUT_ROOT}/logs/${step}_part${part}.log"
  mkdir -p "${out_dir}"
  echo "[launch] ${step} part=${part} cuda=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1_batch \
      --checkpoint_path "${ckpt}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${input_dir}" \
      --output_dir "${out_dir}" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale 1.0 \
      --stage2_attention_mode block_sparse_chunk_causal \
      --stage2_topk_ratio 2.0 \
      --stage2_local_num -1 \
      --stage2_kv_ratio 3.0 \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      > "${log_file}" 2>&1
}

status=0
for ((offset=0; offset<${#CKPTS[@]}; offset+=MAX_CKPTS_PER_WAVE)); do
  pids=()
  gpu=0
  for ((i=0; i<MAX_CKPTS_PER_WAVE && offset+i<${#CKPTS[@]}; i++)); do
    ckpt="${CKPTS[$((offset+i))]}"
    for part in 0 1; do
      run_one_part "${gpu}" "${ckpt}" "${part}" &
      pids+=("$!")
      gpu=$((gpu + 1))
    done
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print0 \
  | while IFS= read -r -d '' step_dir; do
      count="$(find "${step_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
      echo "$(basename "${step_dir}") ${count}"
    done | sort -V > "${OUTPUT_ROOT}/summary_counts.txt"

echo "[info] syncing output to ${S3_OUTPUT_DIR}"
conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"

if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
  echo "[info] restoring occupy jobs on all GPUs"
  tmux kill-session -t occupy 2>/dev/null || true
  pkill -f gpu_stress_tc.py || true
  tmux new-session -d -s occupy "${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"
fi

echo "[done] status=${status} output_root=${OUTPUT_ROOT}"
exit "${status}"
