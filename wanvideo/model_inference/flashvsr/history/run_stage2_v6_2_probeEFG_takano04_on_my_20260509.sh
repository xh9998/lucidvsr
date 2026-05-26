#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq/takano_04_lq.mp4}"
OUT_ROOT="${OUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeEFG_takano04_20260509}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/videos" "${OUT_ROOT}/diagnostics"

run_v62() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="${OUT_ROOT}/logs/${name}.log"
  echo "[run] ${name} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_2 \
      --checkpoint_path "${CKPT}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${INPUT_VIDEO}" \
      --output_video "${OUT_ROOT}/videos/${name}.mp4" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --lq_proj_scale 1.0 \
      --stage2_attention_mode block_sparse_chunk_causal \
      --stage2_topk_ratio 2.0 \
      --stage2_local_num 11 \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      --print_debug \
      "$@" \
      > "${log_file}" 2>&1
}

tmux kill-session -t occupy 2>/dev/null || true
tmux kill-session -t occupy_idle_6_7 2>/dev/null || true
sleep 2
tmux new-session -d -s occupy_idle_6_7 "CUDA_VISIBLE_DEVICES=6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

pids=()
run_v62 0 "E_steps1_adain_lq0" --num_inference_steps 1 --color_fix_lq_start_frame 0 &
pids+=("$!")
run_v62 1 "F_steps50_no_colorfix" --num_inference_steps 50 --disable_color_fix &
pids+=("$!")
run_v62 2 "F_steps50_adain_lq1" --num_inference_steps 50 --color_fix_lq_start_frame 1 &
pids+=("$!")
run_v62 3 "F_steps50_adain_lq4" --num_inference_steps 50 --color_fix_lq_start_frame 4 &
pids+=("$!")
run_v62 4 "F_steps50_wavelet_lq0" --num_inference_steps 50 --color_fix_method wavelet --color_fix_lq_start_frame 0 &
pids+=("$!")

echo "[run] G_projector_chunk_stats"
CUDA_VISIBLE_DEVICES="5" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.diagnose_stage2_v6_projector_chunks \
    --checkpoint_path "${CKPT}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_video "${INPUT_VIDEO}" \
    --output_csv "${OUT_ROOT}/diagnostics/takano04_projector_chunk_stats.csv" \
    --height 768 \
    --width 1280 \
    --num_frames 89 \
    --fps 8 \
    --lq_proj_scale 1.0 \
    --stage2_attention_mode block_sparse_chunk_causal \
    --stage2_topk_ratio 2.0 \
    --stage2_local_num 11 \
    --input_bicubic_upscale 4.0 \
    > "${OUT_ROOT}/logs/G_projector_chunk_stats.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

find "${OUT_ROOT}" -maxdepth 3 -type f | sort > "${OUT_ROOT}/manifest.txt"
tmux kill-session -t occupy_idle_6_7 2>/dev/null || true
tmux new-session -d -s occupy "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"
echo "[done] ${OUT_ROOT}"
exit "${status}"
