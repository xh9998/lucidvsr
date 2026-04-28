#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/task_wrapper/user_output/artifacts/inference/tmp_ckpts_20260417/m1_train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800_step-1000.safetensors}"
INPUT_VIDEO="${INPUT_VIDEO:-/mnt/task_wrapper/user_output/artifacts/input/testset10_duallq_20260417/testset10_17f/lq_native/takano_00_lq_native.mp4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/single_native_scale_compare_20260417}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-17}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_variant() {
  local tag="$1"
  shift
  local output_video="${OUTPUT_ROOT}/${tag}.mp4"
  local log_path="${OUTPUT_ROOT}/${tag}.log"

  /mnt/conda_envs/flashvsr/bin/python \
    wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
    --input_video "${INPUT_VIDEO}" \
    --output_video "${output_video}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --tiled \
    "$@" \
    2>&1 | tee "${log_path}"
}

run_variant "scale1" --lq_proj_scale 1
run_variant "scale5" --lq_proj_scale 5
run_variant "projection_off" --disable_projection --lq_proj_scale 1
