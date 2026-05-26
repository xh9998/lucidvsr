#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

prepare_splits() {
  local output_root="$1"
  mkdir -p "${output_root}/logs" "${output_root}/_input_splits/part0" "${output_root}/_input_splits/part1"
  rm -f "${output_root}/_input_splits/part0"/*.mp4 "${output_root}/_input_splits/part1"/*.mp4
  local idx=0
  while IFS= read -r input; do
    local part=$((idx % 2))
    ln -sf "${input}" "${output_root}/_input_splits/part${part}/$(basename "${input}")"
    idx=$((idx + 1))
  done < <(find "${SOURCE_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort)
}

run_probe_part() {
  local gpu="$1"
  local probe="$2"
  local output_root="$3"
  local part="$4"
  local attention_mode="$5"
  local topk_ratio="$6"
  local local_num="$7"
  local log_file="${output_root}/logs/step-10000_part${part}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_2_batch \
      --checkpoint_path "${CKPT}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${output_root}/_input_splits/part${part}" \
      --output_dir "${output_root}/step-10000" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale 1.0 \
      --stage2_attention_mode "${attention_mode}" \
      --stage2_topk_ratio "${topk_ratio}" \
      --stage2_local_num "${local_num}" \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      --print_debug \
      > "${log_file}" 2>&1
  echo "[done-part] probe=${probe} part=${part} gpu=${gpu}"
}

B_ROOT="${BASE_OUTPUT_ROOT}/stage2_v6_probeB_step10000_v62_fixedlocal11_topk4_20260509"
C_ROOT="${BASE_OUTPUT_ROOT}/stage2_v6_probeC_step10000_v62_densefull_20260509"

echo "[info] probe=B setting=v6.2 full-DiT mask, fixed local_num=11, topk_ratio=4.0"
echo "[info] probe=C setting=v6.2 dense_full self-attention, fixed local_num ignored"
echo "[info] ckpt=${CKPT}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"

prepare_splits "${B_ROOT}"
prepare_splits "${C_ROOT}"

echo "[info] stop existing occupy jobs before using GPUs 2-5 for inference"
tmux kill-session -t occupy 2>/dev/null || true
tmux kill-session -t occupy_idle_0_1_6_7 2>/dev/null || true
tmux kill-session -t occupy_idle_6_7 2>/dev/null || true
pkill -f "/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py" 2>/dev/null || true
sleep 3

pids=()
run_probe_part 2 B "${B_ROOT}" 0 block_sparse_chunk_causal 4.0 11 &
pids+=("$!")
run_probe_part 3 B "${B_ROOT}" 1 block_sparse_chunk_causal 4.0 11 &
pids+=("$!")
run_probe_part 4 C "${C_ROOT}" 0 dense_full 2.0 11 &
pids+=("$!")
run_probe_part 5 C "${C_ROOT}" 1 dense_full 2.0 11 &
pids+=("$!")

tmux new-session -d -s occupy_idle_0_1_6_7 "CUDA_VISIBLE_DEVICES=0,1,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

for output_root in "${B_ROOT}" "${C_ROOT}"; do
  mkdir -p "${output_root}"
  find "${output_root}/step-10000" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | sort > "${output_root}/outputs.txt"
  count="$(find "${output_root}/step-10000" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')"
  echo "step-10000 ${count}" > "${output_root}/summary_counts.txt"
  echo "[summary] ${output_root} count=${count}"
done

tmux kill-session -t occupy_idle_0_1_6_7 2>/dev/null || true
pkill -f "/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py" 2>/dev/null || true
tmux new-session -d -s occupy "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

echo "[done] status=${status}"
exit "${status}"
