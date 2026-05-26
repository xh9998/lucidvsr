#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq/takano_04_lq.mp4}"
OUT_ROOT="${OUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeH_takano04_20260510}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/videos"

tmux kill-session -t occupy 2>/dev/null || true
tmux kill-session -t occupy_idle_1_7 2>/dev/null || true
sleep 2
tmux new-session -d -s occupy_idle_1_7 "CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

CUDA_VISIBLE_DEVICES="0" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_2 \
    --checkpoint_path "${CKPT}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_video "${INPUT_VIDEO}" \
    --output_video "${OUT_ROOT}/videos/H_official_mask_topk2_local11_takano04.mp4" \
    --height 768 \
    --width 1280 \
    --num_frames 89 \
    --fps 8 \
    --num_inference_steps 50 \
    --lq_proj_scale 1.0 \
    --stage2_attention_mode block_sparse_official_mask \
    --stage2_topk_ratio 2.0 \
    --stage2_local_num 11 \
    --input_bicubic_upscale 4.0 \
    --color_fix_method adain \
    --print_debug \
    > "${OUT_ROOT}/logs/H_official_mask_topk2_local11_takano04.log" 2>&1

find "${OUT_ROOT}" -maxdepth 3 -type f | sort > "${OUT_ROOT}/manifest.txt"
tmux kill-session -t occupy_idle_1_7 2>/dev/null || true
tmux new-session -d -s occupy "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"
echo "[done] ${OUT_ROOT}"
