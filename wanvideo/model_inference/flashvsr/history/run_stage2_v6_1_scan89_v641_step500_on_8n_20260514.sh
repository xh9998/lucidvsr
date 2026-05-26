#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PATH="/mnt/conda_envs/DiffVSR_b200/bin:/mnt/conda_envs/flashvsr/bin:${PATH}"

CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_4_1_89f_step500_20260514}"
CKPT_S3_DIR="${CKPT_S3_DIR:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
SOURCE_INPUT_S3_DIR="${SOURCE_INPUT_S3_DIR:-s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_official85f_step500_8n_20260514}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://lxh/tmp/stage2_v6_4_1_official85f_step500_8n_20260514}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
STEP_MOD="${STEP_MOD:-500}"
MIN_STEP="${MIN_STEP:-500}"
MAX_STEP="${MAX_STEP:-8000}"
MAX_CKPTS_PER_WAVE="${MAX_CKPTS_PER_WAVE:-8}"
SHARDS_PER_CKPT="${SHARDS_PER_CKPT:-1}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"
if (( MAX_CKPTS_PER_WAVE * SHARDS_PER_CKPT > 8 )); then
  echo "[error] MAX_CKPTS_PER_WAVE * SHARDS_PER_CKPT must be <= 8" >&2
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

mkdir -p "${CKPT_DIR}" "${SOURCE_INPUT_DIR}" "${OUTPUT_ROOT}/logs"
for part in {0..7}; do
  mkdir -p "${OUTPUT_ROOT}/_input_splits/part${part}"
done
for part in {0..1}; do
  mkdir -p "${OUTPUT_ROOT}/_input_splits_2way/part${part}"
done
mkdir -p "${OUTPUT_ROOT}/_input_all"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python_bin=${PYTHON_BIN}"
echo "[info] ckpt_s3_dir=${CKPT_S3_DIR}"
echo "[info] ckpt_dir=${CKPT_DIR}"
echo "[info] source_input_s3_dir=${SOURCE_INPUT_S3_DIR}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] s3_output_dir=${S3_OUTPUT_DIR}"
echo "[info] step_mod=${STEP_MOD} min_step=${MIN_STEP} max_step=${MAX_STEP}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] flashvsr python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[info] stopping stale inference / occupy processes"
tmux kill-session -t occupy 2>/dev/null || true
pkill -f infer_flashvsr_stage2_v6_1_batch || true
pkill -f gpu_stress_tc.py || true
sleep 2

if [[ "$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')" -lt 10 ]]; then
  echo "[info] syncing testset from ${SOURCE_INPUT_S3_DIR}"
  conductor s3 sync "${SOURCE_INPUT_S3_DIR}" "${SOURCE_INPUT_DIR}"
fi

for part in {0..7}; do
  rm -f "${OUTPUT_ROOT}/_input_splits/part${part}"/*.mp4
done
for part in {0..1}; do
  rm -f "${OUTPUT_ROOT}/_input_splits_2way/part${part}"/*.mp4
done
rm -f "${OUTPUT_ROOT}/_input_all"/*.mp4
idx=0
while IFS= read -r input; do
  part=$((idx % 8))
  ln -sf "${input}" "${OUTPUT_ROOT}/_input_splits/part${part}/$(basename "${input}")"
  part2=$((idx % 2))
  ln -sf "${input}" "${OUTPUT_ROOT}/_input_splits_2way/part${part2}/$(basename "${input}")"
  ln -sf "${input}" "${OUTPUT_ROOT}/_input_all/$(basename "${input}")"
  idx=$((idx + 1))
done < <(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort | head -10)

EXPECTED_INPUTS="$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort | head -10 | wc -l | tr -d ' ')"
if [[ "${EXPECTED_INPUTS}" -le 0 ]]; then
  echo "[error] no input mp4 found in ${SOURCE_INPUT_DIR}" >&2
  exit 1
fi
echo "[info] expected_inputs=${EXPECTED_INPUTS}"

ckpt_list="${OUTPUT_ROOT}/logs/stage2_v6_4_1_step500_s3_ckpt_list.txt"
conductor s3 ls "${CKPT_S3_DIR}/" | awk '{print $NF}' | grep -E '^step-[0-9]+\.safetensors$' | sort -V > "${ckpt_list}"

echo "[info] downloading missing step-${STEP_MOD} checkpoints"
while IFS= read -r file; do
  step="${file#step-}"
  step="${step%.safetensors}"
  if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= MIN_STEP && step <= MAX_STEP && step % STEP_MOD == 0 )); then
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
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= MIN_STEP && step <= MAX_STEP && step % STEP_MOD == 0 )); then
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
        fi
      done
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[info] no new checkpoints need testing"
  find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
  conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"
  if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
    tmux kill-session -t occupy 2>/dev/null || true
    pkill -f gpu_stress_tc.py || true
    tmux new-session -d -s occupy "${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"
  fi
  exit 0
fi

printf '%s\n' "${CKPTS[@]}" > "${OUTPUT_ROOT}/logs/step500_to_test.txt"
echo "[info] new_ckpts_to_test=${#CKPTS[@]}"
cat "${OUTPUT_ROOT}/logs/step500_to_test.txt"

run_one_part() {
  local gpu="$1"
  local ckpt="$2"
  local part="$3"
  local step out_dir input_dir log_file
  step="$(basename "${ckpt}" .safetensors)"
  out_dir="${OUTPUT_ROOT}/${step}"
  if [[ "${SHARDS_PER_CKPT}" -eq 1 ]]; then
    input_dir="${OUTPUT_ROOT}/_input_all"
  elif [[ "${SHARDS_PER_CKPT}" -eq 2 ]]; then
    input_dir="${OUTPUT_ROOT}/_input_splits_2way/part${part}"
  else
    input_dir="${OUTPUT_ROOT}/_input_splits/part${part}"
  fi
  log_file="${OUTPUT_ROOT}/logs/${step}_part${part}.log"
  mkdir -p "${out_dir}"
  if [[ "$(find -L "${input_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')" -eq 0 ]]; then
    echo "[skip-empty-part] ${step} part=${part}"
    return 0
  fi
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
  for ((i=0; i<MAX_CKPTS_PER_WAVE && offset+i<${#CKPTS[@]}; i++)); do
    ckpt="${CKPTS[$((offset+i))]}"
    for ((part=0; part<SHARDS_PER_CKPT; part++)); do
      gpu=$((i * SHARDS_PER_CKPT + part))
      run_one_part "${gpu}" "${ckpt}" "${part}" &
      pids+=("$!")
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
