#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800/output/step-2500.safetensors}"
TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/takano_train_17f33f_x4_20260422}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/takano_train_17f33f_x4_lora17_step2500_20260422}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
INPUT_BICUBIC_UPSCALE="${INPUT_BICUBIC_UPSCALE:-4.0}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

echo "Using Python: ${PYTHON_BIN}"

run_variant() {
  local tag="$1"
  local num_frames="$2"
  local input_dir="${TESTSET_ROOT}/takano_train_${tag}/lq_x4"
  local run_dir="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"

  for input_video in "${input_dir}"/*.mp4; do
    local sample_name
    sample_name="$(basename "${input_video}" .mp4)"
    local output_video="${run_dir}/${sample_name}_sr.mp4"

    {
      echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
      echo "INPUT_VIDEO=${input_video}"
      echo "NUM_FRAMES=${num_frames}"
      echo "LQ_PROJ_SCALE=${LQ_PROJ_SCALE}"
      echo "INPUT_BICUBIC_UPSCALE=${INPUT_BICUBIC_UPSCALE}"
      "${PYTHON_BIN}" \
        wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
        --checkpoint_path "${CHECKPOINT_PATH}" \
        --base_model_dir "${BASE_MODEL_DIR}" \
        --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
        --input_video "${input_video}" \
        --output_video "${output_video}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${num_frames}" \
        --fps "${FPS}" \
        --seed "${SEED}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --lq_proj_scale "${LQ_PROJ_SCALE}" \
        --input_bicubic_upscale "${INPUT_BICUBIC_UPSCALE}" \
        --device "${DEVICE}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --tiled
    } 2>&1 | tee -a "${run_dir}/run.log"
  done
}

run_variant "17f" 17
run_variant "33f" 33
