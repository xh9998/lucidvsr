#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_duallq_20260417}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/testset10_duallq_x4_ckpts_20260417}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/tmp_ckpts_20260417}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
LQ_PROJ_SCALE="${LQ_PROJ_SCALE:-1.0}"

mkdir -p "${OUTPUT_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${ROOT_DIR}"

run_dir="${OUTPUT_ROOT}/train_stage1_release_16gpu_v3_17f_takano_fullft_bs24_lr1e5_alpha5_nostartval_20260414_125800_step-1100_testset10_17f"
mkdir -p "${run_dir}"
: > "${run_dir}/run.log"

for input_video in "${TESTSET_ROOT}/testset10_17f/lq_x4"/*.mp4; do
  sample_name="$(basename "${input_video}" .mp4)"
  output_video="${run_dir}/${sample_name}_sr.mp4"
  {
    echo "TAG=train_stage1_release_16gpu_v3_17f_takano_fullft_bs24_lr1e5_alpha5_nostartval_20260414_125800"
    echo "STEP=step-1100"
    echo "KIND=fullft"
    echo "TESTSET_VARIANT=testset10_17f"
    echo "INPUT_KIND=lq_x4"
    echo "INPUT_VIDEO=${input_video}"
    echo "LQ_PROJ_SCALE=${LQ_PROJ_SCALE}"
    /mnt/conda_envs/flashvsr/bin/python \
      wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v3_fullft.py \
      --checkpoint_path "${LOCAL_CKPT_DIR}/m3_train_stage1_release_16gpu_v3_17f_takano_fullft_bs24_lr1e5_alpha5_nostartval_20260414_125800_step-1100.safetensors" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_video "${input_video}" \
      --output_video "${output_video}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames 17 \
      --fps "${FPS}" \
      --seed "${SEED}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --lq_proj_scale "${LQ_PROJ_SCALE}" \
      --device "${DEVICE}" \
      --torch_dtype "${TORCH_DTYPE}" \
      --tiled
  } 2>&1 | tee -a "${run_dir}/run.log"
done
