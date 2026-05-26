#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_20closeup_synthetic_20260518}"
CRITICAL_LOCAL="${CRITICAL_LOCAL:-/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
GPU_LIST="${GPU_LIST:-0,2}"
METHODS="${METHODS:-stage2_v641_step6000,stage3_v7d32_step2000}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"

INPUT_DIR="${OUTPUT_ROOT}/_inputs/synthetic_89f"
STAGE1_USMGT_CKPT="${CRITICAL_LOCAL}/stage1_usmgt_takano20250205_step3000.safetensors"
STAGE2_641_CKPT="${CRITICAL_LOCAL}/stage2_v641_step6000.safetensors"
STAGE3_D32_CKPT="${CRITICAL_LOCAL}/stage3_v7d32_step2000.safetensors"

count_inputs() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.mp4' | wc -l | tr -d ' '
}

count_outputs() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

write_time() {
  local method="$1" seconds="$2" status="$3" output_dir="$4"
  local n out
  n="$(count_inputs "${INPUT_DIR}")"
  out="$(count_outputs "${output_dir}")"
  {
    echo "method=${method}"
    echo "dataset=synthetic_89f_closeup"
    echo "status=${status}"
    echo "seconds=${seconds}"
    echo "num_inputs=${n}"
    echo "num_outputs=${out}"
    awk -v s="${seconds}" -v n="${n}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "input_dir=${INPUT_DIR}"
    echo "output_dir=${output_dir}"
  } | tee "${OUTPUT_ROOT}/logs/${method}.time"
}

maybe_skip_done() {
  local method="$1" output_dir="$2"
  local n out status
  n="$(count_inputs "${INPUT_DIR}")"
  out="$(count_outputs "${output_dir}")"
  status="$(grep -E '^status=' "${OUTPUT_ROOT}/logs/${method}.time" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
  [[ "${status}" == "0" && "${n}" != "0" && "${out}" == "${n}" ]]
}

run_stage1_usmgt() {
  local gpu="$1" method="stage1_usmgt_step3000"
  local output_dir="${OUTPUT_ROOT}/synthetic_89f/${method}"
  mkdir -p "${output_dir}"
  if maybe_skip_done "${method}" "${output_dir}"; then
    echo "[skip_done] ${method}"
    return 0
  fi
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" CUDA_DEVICE="${gpu}" \
  PYTHON_BIN="${PYTHON_BIN}" CHECKPOINT_PATH="${STAGE1_USMGT_CKPT}" INPUT_DIR="${INPUT_DIR}" OUTPUT_DIR="${output_dir}" \
  BASE_MODEL_DIR="${BASE_MODEL_DIR}" PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH}" NUM_FRAMES=89 FPS="${FPS}" SEED="${SEED}" \
  NUM_INFERENCE_STEPS=50 INPUT_BICUBIC_UPSCALE=4.0 LQ_PROJ_TEMPORAL_MODE=nonstreaming_aligned COLOR_FIX_METHOD=adain \
  bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_stage1_v5_3_aligned_dir.sh" 2>&1 | tee "${OUTPUT_ROOT}/logs/${method}.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  write_time "${method}" "$((end - start))" "${status}" "${output_dir}"
  return "${status}"
}

run_stage2() {
  local gpu="$1" method="stage2_v641_step6000"
  local output_dir="${OUTPUT_ROOT}/synthetic_89f/${method}"
  mkdir -p "${output_dir}"
  if maybe_skip_done "${method}" "${output_dir}"; then
    echo "[skip_done] ${method}"
    return 0
  fi
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" CUDA_DEVICE="${gpu}" "${PYTHON_BIN}" -u "${ROOT_DIR}/wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1_batch.py" \
    --checkpoint_path "${STAGE2_641_CKPT}" --base_model_dir "${BASE_MODEL_DIR}" --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_dir "${INPUT_DIR}" --output_dir "${output_dir}" --height 768 --width 1280 --num_frames 89 --fps "${FPS}" --seed "${SEED}" \
    --num_inference_steps 50 --stage2_attention_mode block_sparse_chunk_causal --stage2_topk_ratio 2.0 --stage2_local_num -1 \
    --stage2_kv_ratio 3.0 --input_bicubic_upscale 4.0 --color_fix_method adain 2>&1 | tee "${OUTPUT_ROOT}/logs/${method}.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  write_time "${method}" "$((end - start))" "${status}" "${output_dir}"
  return "${status}"
}

run_stage3() {
  local gpu="$1" method="stage3_v7d32_step2000"
  local output_dir="${OUTPUT_ROOT}/synthetic_89f/${method}"
  mkdir -p "${output_dir}"
  if maybe_skip_done "${method}" "${output_dir}"; then
    echo "[skip_done] ${method}"
    return 0
  fi
  local start end status
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" CUDA_DEVICE="${gpu}" "${PYTHON_BIN}" -u "${ROOT_DIR}/wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py" \
    --checkpoint_path "${STAGE3_D32_CKPT}" --base_model_dir "${BASE_MODEL_DIR}" --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_dir "${INPUT_DIR}" --output_dir "${output_dir}" --height 768 --width 1280 --num_frames 89 --fps "${FPS}" --seed "${SEED}" \
    --num_inference_steps 1 --stage2_attention_mode block_sparse_chunk_causal --stage2_topk_ratio 2.0 --stage2_local_num -1 \
    --stage2_kv_ratio 3.0 --input_bicubic_upscale 4.0 --color_fix_method adain 2>&1 | tee "${OUTPUT_ROOT}/logs/${method}.log"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  write_time "${method}" "$((end - start))" "${status}" "${output_dir}"
  return "${status}"
}

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
IFS=',' read -r -a JOBS <<< "${METHODS}"
next=0
while [[ "${next}" -lt "${#JOBS[@]}" ]]; do
  pids=()
  for gpu in "${GPUS[@]}"; do
    [[ "${next}" -lt "${#JOBS[@]}" ]] || break
    method="${JOBS[${next}]}"
    case "${method}" in
      stage1_usmgt_step3000) run_stage1_usmgt "${gpu}" & ;;
      stage2_v641_step6000) run_stage2 "${gpu}" & ;;
      stage3_v7d32_step2000) run_stage3 "${gpu}" & ;;
      *) echo "unknown method=${method}" >&2; exit 2 ;;
    esac
    pids+=("$!")
    next=$((next + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done
