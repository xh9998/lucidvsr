#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
TESTSET_DIR="${TESTSET_DIR:-/mnt/task_wrapper/user_output/artifacts/input/testset10}"
LQ_DIR="${LQ_DIR:-${TESTSET_DIR}/lq}"
GT_DIR="${GT_DIR:-${TESTSET_DIR}/gt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/testset10_ckpts_20260417}"
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

run_one() {
  local tag="$1"
  local step="$2"
  local kind="$3"
  local num_frames="$4"
  local ckpt_path="$5"

  local run_dir="${OUTPUT_ROOT}/${tag}_${step}"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"

  for input_video in "${LQ_DIR}"/*.mp4; do
    local sample_name
    sample_name="$(basename "${input_video}" .mp4)"
    local output_video="${run_dir}/${sample_name}_sr.mp4"

    local infer_py
    if [ "${kind}" = "fullft" ]; then
      infer_py="wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v3_fullft.py"
    else
      infer_py="wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py"
    fi

    local -a cmd=(
      /mnt/conda_envs/flashvsr/bin/python
      "${infer_py}"
      --checkpoint_path "${ckpt_path}"
      --base_model_dir "${BASE_MODEL_DIR}"
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}"
      --input_video "${input_video}"
      --output_video "${output_video}"
      --height "${HEIGHT}"
      --width "${WIDTH}"
      --num_frames "${num_frames}"
      --fps "${FPS}"
      --seed "${SEED}"
      --num_inference_steps "${NUM_INFERENCE_STEPS}"
      --lq_proj_scale "${LQ_PROJ_SCALE}"
      --device "${DEVICE}"
      --torch_dtype "${TORCH_DTYPE}"
      --tiled
    )

    printf '%q ' "${cmd[@]}" > "${run_dir}/${sample_name}_launch_command.sh"
    printf '\n' >> "${run_dir}/${sample_name}_launch_command.sh"

    {
      echo "TAG=${tag}"
      echo "STEP=${step}"
      echo "KIND=${kind}"
      echo "CHECKPOINT=${ckpt_path}"
      echo "INPUT_VIDEO=${input_video}"
      echo "NUM_FRAMES=${num_frames}"
      echo "LQ_PROJ_SCALE=${LQ_PROJ_SCALE}"
      echo "OUTPUT_VIDEO=${output_video}"
      "${cmd[@]}"
    } 2>&1 | tee -a "${run_dir}/run.log"
  done
}

run_one \
  "train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800" \
  "step-1000" \
  "lora" \
  "17" \
  "${LOCAL_CKPT_DIR}/m1_train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800_step-1000.safetensors"

run_one \
  "train_stage1_release_16gpu_v2_89f_takano_bs4_lr1e5_alpha5_resume_step1000_seed20260415_20260415_012200" \
  "step-1200" \
  "lora" \
  "17" \
  "${LOCAL_CKPT_DIR}/m2_train_stage1_release_16gpu_v2_89f_takano_bs4_lr1e5_alpha5_resume_step1000_seed20260415_20260415_012200_step-1200.safetensors"

run_one \
  "train_stage1_release_16gpu_v3_17f_takano_fullft_bs24_lr1e5_alpha5_nostartval_20260414_125800" \
  "step-1100" \
  "fullft" \
  "17" \
  "${LOCAL_CKPT_DIR}/m3_train_stage1_release_16gpu_v3_17f_takano_fullft_bs24_lr1e5_alpha5_nostartval_20260414_125800_step-1100.safetensors"
