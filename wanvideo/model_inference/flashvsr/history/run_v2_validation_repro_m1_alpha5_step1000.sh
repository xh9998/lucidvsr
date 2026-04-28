#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXPERIMENT_DIR}/output/step-1000.safetensors}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/v2_validation_repro_m1_alpha5_step1000_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SAMPLE_INDICES="${SAMPLE_INDICES:-0,1,2}"
SAVE_TENSOR_PT="${SAVE_TENSOR_PT:-1}"
ALSO_RUN_MP4_ROUNDTRIP="${ALSO_RUN_MP4_ROUNDTRIP:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

/mnt/conda_envs/flashvsr/bin/python \
  wanvideo/model_inference/flashvsr/scripts/infer_flashvsr_v2_validation_repro.py \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --sample_indices "${SAMPLE_INDICES}" \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  $( [ "${SAVE_TENSOR_PT}" = "1" ] && echo "--save_tensor_pt" ) \
  $( [ "${ALSO_RUN_MP4_ROUNDTRIP}" = "1" ] && echo "--also_run_mp4_roundtrip" ) \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
