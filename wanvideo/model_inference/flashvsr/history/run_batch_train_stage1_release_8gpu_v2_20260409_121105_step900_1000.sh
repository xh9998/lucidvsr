#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export PYTHONNOUSERSITE=1

EXP_NAME="train_stage1_release_8gpu_v2_20260409_121105"
CHECKPOINT_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${EXP_NAME}/output"
INPUT_DIR="/mnt/task_wrapper/user_output/artifacts/eval_samples/${EXP_NAME}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="/mnt/task_wrapper/user_output/artifacts/inference/${EXP_NAME}_v2_batch_step900_1000_${RUN_TS}"
SELF_SH="/mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_batch_train_stage1_release_8gpu_v2_20260409_121105_step900_1000.sh"

mkdir -p "${OUTPUT_ROOT}"
cp "${SELF_SH}" "${OUTPUT_ROOT}/driver.sh" || true

echo "EXP_NAME=${EXP_NAME}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
INPUT_DIR="${INPUT_DIR}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CHECKPOINT_NAMES="step-900.safetensors,step-1000.safetensors" \
INPUT_GLOB="sample_*/lq.mp4" \
NUM_INFERENCE_STEPS="50" \
PROJECTION_SCALE="1.0" \
bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/run_batch_flashvsr_stage1_v2.sh
