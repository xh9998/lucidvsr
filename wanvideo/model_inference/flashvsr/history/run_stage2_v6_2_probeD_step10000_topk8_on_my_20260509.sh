#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeD_step10000_v62_fixedlocal11_topk8_20260509}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

prepare_splits() {
  mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/_input_splits/part0" "${OUTPUT_ROOT}/_input_splits/part1"
  rm -f "${OUTPUT_ROOT}/_input_splits/part0"/*.mp4 "${OUTPUT_ROOT}/_input_splits/part1"/*.mp4
  local idx=0
  while IFS= read -r input; do
    local part=$((idx % 2))
    ln -sf "${input}" "${OUTPUT_ROOT}/_input_splits/part${part}/$(basename "${input}")"
    idx=$((idx + 1))
  done < <(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort)
}

run_part() {
  local gpu="$1"
  local part="$2"
  local log_file="${OUTPUT_ROOT}/logs/step-10000_part${part}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_2_batch \
      --checkpoint_path "${CKPT}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${OUTPUT_ROOT}/_input_splits/part${part}" \
      --output_dir "${OUTPUT_ROOT}/step-10000" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale 1.0 \
      --stage2_attention_mode block_sparse_chunk_causal \
      --stage2_topk_ratio 8.0 \
      --stage2_local_num 11 \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      --print_debug \
      > "${log_file}" 2>&1
  echo "[done-part] probe=D part=${part} gpu=${gpu}"
}

echo "[info] probe=D setting=v6.2 full-DiT mask, fixed local_num=11, topk_ratio=8.0"
echo "[info] ckpt=${CKPT}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
prepare_splits

pids=()
run_part 6 0 &
pids+=("$!")
run_part 7 1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

find "${OUTPUT_ROOT}/step-10000" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | sort > "${OUTPUT_ROOT}/outputs.txt"
count="$(find "${OUTPUT_ROOT}/step-10000" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
echo "step-10000 ${count}" > "${OUTPUT_ROOT}/summary_counts.txt"
echo "[summary] ${OUTPUT_ROOT} count=${count}"
echo "[done] status=${status}"
exit "${status}"
