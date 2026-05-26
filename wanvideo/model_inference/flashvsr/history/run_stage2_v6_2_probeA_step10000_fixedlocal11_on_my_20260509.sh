#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeA_step10000_v62_fixedlocal11_20260509}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/_input_splits/part0" "${OUTPUT_ROOT}/_input_splits/part1"

echo "[info] probe=A"
echo "[info] setting=v6.2 full-DiT mask, fixed stage2_local_num=11, topk_ratio=2.0"
echo "[info] ckpt=${CKPT}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"

tmux kill-session -t occupy 2>/dev/null || true
tmux kill-session -t occupy_idle_2_7 2>/dev/null || true
pkill -f infer_flashvsr_stage2_v6_2_batch || true
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
      --stage2_topk_ratio 2.0 \
      --stage2_local_num 11 \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      --print_debug \
      > "${log_file}" 2>&1
}

run_part 0 0 &
pid0="$!"
run_part 1 1 &
pid1="$!"

tmux new-session -d -s occupy_idle_2_7 "CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1

find "${OUTPUT_ROOT}/step-10000" -maxdepth 1 -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
count="$(find "${OUTPUT_ROOT}/step-10000" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
echo "step-10000 ${count}" > "${OUTPUT_ROOT}/summary_counts.txt"

tmux kill-session -t occupy_idle_2_7 2>/dev/null || true
pkill -f gpu_stress_tc.py || true
tmux new-session -d -s occupy "${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

echo "[done] status=${status} count=${count} output_root=${OUTPUT_ROOT}"
exit "${status}"
