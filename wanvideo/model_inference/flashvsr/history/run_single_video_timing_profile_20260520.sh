#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_single_video_timing_20260520}"
OUTPUT_S3="${OUTPUT_S3:-s3://lxh/data/test/ppt_benchmark_single_video_timing_20260520}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
SYNTHETIC_S3="${SYNTHETIC_S3:-s3://lxh/data/test/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
CRITICAL_S3="${CRITICAL_S3:-s3://lxh/models/flashvsr/critical_ppt_20260518}"
CRITICAL_LOCAL="${CRITICAL_LOCAL:-/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"

STAGE1_535_CKPT="${CRITICAL_LOCAL}/stage1_v535_step10000.safetensors"
STAGE1_USMGT_CKPT="${CRITICAL_LOCAL}/stage1_usmgt_takano20250205_step3000.safetensors"
STAGE2_641_CKPT="${CRITICAL_LOCAL}/stage2_v641_step6000.safetensors"
STAGE3_D32_CKPT="${CRITICAL_LOCAL}/stage3_v7d32_step2000.safetensors"

GPU="${GPU:-2}"
OCCUPY_GPUS_DURING="${OCCUPY_GPUS_DURING-0,1,3}"
FINAL_OCCUPY_GPUS="${FINAL_OCCUPY_GPUS-0,1,2,3}"
METHODS="${METHODS:-flashvsr_official,seedvr3b,seedvr2_3b,stage1_535_step10000,stage1_usmgt_step3000,stage2_v641_step6000,stage3_v7d32_step2000}"
INPUT_VIDEO="${INPUT_VIDEO-}"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${ROOT_DIR}"

if [[ ! -d "${SYNTHETIC_ROOT}/lq" ]]; then
  conductor s3 sync "${SYNTHETIC_S3}" "${SYNTHETIC_ROOT}"
fi
if [[ ! -s "${STAGE1_535_CKPT}" || ! -s "${STAGE1_USMGT_CKPT}" || ! -s "${STAGE2_641_CKPT}" || ! -s "${STAGE3_D32_CKPT}" ]]; then
  conductor s3 sync "${CRITICAL_S3}" "${CRITICAL_LOCAL}"
fi
if [[ -z "${INPUT_VIDEO}" ]]; then
  INPUT_VIDEO="$(find "${SYNTHETIC_ROOT}/lq" -maxdepth 1 -type f -name '*.mp4' | sort | head -1)"
fi
if [[ -z "${INPUT_VIDEO}" || ! -s "${INPUT_VIDEO}" ]]; then
  echo "missing INPUT_VIDEO=${INPUT_VIDEO}" >&2
  exit 2
fi

echo "input_video=${INPUT_VIDEO}" | tee "${OUTPUT_ROOT}/settings.txt"
echo "methods=${METHODS}" | tee -a "${OUTPUT_ROOT}/settings.txt"
echo "gpu=${GPU}" | tee -a "${OUTPUT_ROOT}/settings.txt"
echo "timing_note=single cold process; SeedVR timings are external subprocess totals; internal FlashVSR models split DiT/VAE/save." | tee -a "${OUTPUT_ROOT}/settings.txt"

tmux kill-session -t occupy_single_profile_final 2>/dev/null || true
tmux kill-session -t occupy_single_profile_spare 2>/dev/null || true
pkill -f '[g]pu_stress_tc.py' || true
pkill -f '[g]pu_stress_tc.sh' || true
sleep 2
if [[ -n "${OCCUPY_GPUS_DURING}" ]]; then
  tmux new-session -d -s occupy_single_profile_spare "PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python GPUS='${OCCUPY_GPUS_DURING}' bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh"
fi

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u wanvideo/model_inference/flashvsr/profile_single_video_timing_20260520.py \
  --input_video "${INPUT_VIDEO}" \
  --output_dir "${OUTPUT_ROOT}" \
  --methods "${METHODS}" \
  --base_model_dir "${BASE_MODEL_DIR}" \
  --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
  --height 768 \
  --width 1280 \
  --num_frames 89 \
  --fps 8 \
  --seed 0 \
  --num_inference_steps 50 \
  --input_bicubic_upscale 4.0 \
  --color_fix_method adain \
  --stage2_attention_mode block_sparse_chunk_causal \
  --stage2_topk_ratio 2.0 \
  --stage2_local_num -1 \
  --stage2_kv_ratio 3.0 \
  --stage1_535_ckpt "${STAGE1_535_CKPT}" \
  --stage1_usmgt_ckpt "${STAGE1_USMGT_CKPT}" \
  --stage2_641_ckpt "${STAGE2_641_CKPT}" \
  --stage3_d32_ckpt "${STAGE3_D32_CKPT}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/profile.log"
status="${PIPESTATUS[0]}"
set -e

conductor s3 sync "${OUTPUT_ROOT}" "${OUTPUT_S3}" || true

tmux kill-session -t occupy_single_profile_spare 2>/dev/null || true
if [[ -n "${FINAL_OCCUPY_GPUS}" ]]; then
  tmux kill-session -t occupy_single_profile_final 2>/dev/null || true
  tmux new-session -d -s occupy_single_profile_final "PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python GPUS='${FINAL_OCCUPY_GPUS}' bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh"
fi

exit "${status}"
