#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_worker2_20260506}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_worker2_scan89_v61_20260506_8way}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
STEP_MOD="${STEP_MOD:-500}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/_input_splits/part0" "${OUTPUT_ROOT}/_input_splits/part1"

echo "[info] cleaning stale inference / occupy processes"
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

mapfile -t CKPTS < <(
  find "${CKPT_DIR}" -maxdepth 1 -type f -name 'step-*.safetensors' \
    | sort -V \
    | while read -r ckpt; do
        base="$(basename "${ckpt}")"
        step="${base#step-}"
        step="${step%.safetensors}"
        if [[ "${step}" =~ ^[0-9]+$ ]] && (( step % STEP_MOD == 0 )); then
          printf '%s\n' "${ckpt}"
        fi
      done
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "[error] no checkpoint found in ${CKPT_DIR} with step % ${STEP_MOD} == 0" >&2
  exit 1
fi

echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] num_inputs=$(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
echo "[info] num_ckpts=${#CKPTS[@]}"

pids=()
gpu=0
for ckpt in "${CKPTS[@]}"; do
  step="$(basename "${ckpt}" .safetensors)"
  for part in 0 1; do
    input_dir="${OUTPUT_ROOT}/_input_splits/part${part}"
    out_dir="${OUTPUT_ROOT}/${step}"
    log_file="${OUTPUT_ROOT}/logs/${step}_part${part}.log"
    echo "[launch] ${step} part=${part} cuda=${gpu}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
        --stage2_kv_ratio 3.0 \
        --input_bicubic_upscale 4.0 \
        --color_fix_method adain
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print0 \
  | while IFS= read -r -d '' step_dir; do
      count="$(find "${step_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
      echo "$(basename "${step_dir}") ${count}"
    done | sort -V > "${OUTPUT_ROOT}/summary_counts.txt"

echo "[info] restoring occupy jobs on all GPUs"
pkill -f gpu_stress_tc.py || true
bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh >/tmp/occupy_all_after_v61_8way.log 2>&1 &

echo "[done] status=${status} output_root=${OUTPUT_ROOT}"
exit "${status}"
