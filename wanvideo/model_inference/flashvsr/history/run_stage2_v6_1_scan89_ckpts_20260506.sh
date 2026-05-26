#!/usr/bin/env bash
set -euo pipefail

# Do not inherit a generic PYTHON_BIN from the interactive shell; on Bolt it
# is often bound to b200. Stage2 FlashVSR inference must run in flashvsr.
PYTHON_BIN="${FLASHVSR_PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_worker2_20260506}"
INPUT_DIR="${INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_worker2_scan89_v61_20260506}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
STEP_MOD="${STEP_MOD:-500}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}/logs"

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

IFS=',' read -r -a DEVICES <<< "${CUDA_DEVICE_LIST}"
echo "[info] ckpt_dir=${CKPT_DIR}"
echo "[info] input_dir=${INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"
echo "[info] devices=${CUDA_DEVICE_LIST}"
echo "[info] num_ckpts=${#CKPTS[@]}"

idx=0
pids=()
for ckpt in "${CKPTS[@]}"; do
  base="$(basename "${ckpt}")"
  step="${base%.safetensors}"
  device="${DEVICES[$((idx % ${#DEVICES[@]}))]}"
  out_dir="${OUTPUT_ROOT}/${step}"
  log_file="${OUTPUT_ROOT}/logs/${step}.log"
  echo "[launch] ${step} cuda=${device}"
  (
    export CUDA_VISIBLE_DEVICES="${device}"
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1_batch \
      --checkpoint_path "${ckpt}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${INPUT_DIR}" \
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
  idx=$((idx + 1))
  if (( idx % ${#DEVICES[@]} == 0 )); then
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        status=1
      fi
    done
    pids=()
    if (( status != 0 )); then
      echo "[error] one or more checkpoint inference jobs failed" >&2
      exit "${status}"
    fi
  fi
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "[error] one or more checkpoint inference jobs failed" >&2
  exit "${status}"
fi

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
echo "[done] output_root=${OUTPUT_ROOT}"
