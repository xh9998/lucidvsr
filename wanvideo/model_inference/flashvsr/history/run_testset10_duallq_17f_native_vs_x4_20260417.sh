#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CKPT_PATH="${CKPT_PATH:-/mnt/task_wrapper/user_output/artifacts/inference/tmp_ckpts_20260417/m1_train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800_step-1000.safetensors}"
TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_duallq_20260417/testset10_17f}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/testset10_duallq_17f_native_vs_x4_20260417}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_X4="${RUN_X4:-1}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_variant() {
  local input_subdir="$1"
  local tag="$2"
  local input_dir="${TESTSET_ROOT}/${input_subdir}"
  local run_dir="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"

  for input_video in "${input_dir}"/*.mp4; do
    local sample_name
    sample_name="$(basename "${input_video}" .mp4)"
    local output_video="${run_dir}/${sample_name}_sr.mp4"

    {
      echo "CKPT_PATH=${CKPT_PATH}"
      echo "INPUT_SUBDIR=${input_subdir}"
      echo "INPUT_VIDEO=${input_video}"
      echo "OUTPUT_VIDEO=${output_video}"
      echo "LQ_PROJ_SCALE=${LQ_PROJ_SCALE}"
      /mnt/conda_envs/flashvsr/bin/python \
        wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
        --checkpoint_path "${CKPT_PATH}" \
        --base_model_dir "${BASE_MODEL_DIR}" \
        --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
        --input_video "${input_video}" \
        --output_video "${output_video}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --num_frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --seed "${SEED}" \
        --num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --lq_proj_scale "${LQ_PROJ_SCALE}" \
        --device "${DEVICE}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --tiled
    } 2>&1 | tee -a "${run_dir}/run.log"
  done
}

if [[ "${RUN_NATIVE}" == "1" ]]; then
  run_variant "lq_native" "native"
fi

if [[ "${RUN_X4}" == "1" ]]; then
  run_variant "lq_x4" "x4"
fi
