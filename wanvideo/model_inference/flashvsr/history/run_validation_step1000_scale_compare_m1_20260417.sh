#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/validation_step1000_m1_20260417/step-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/validation_step1000_m1_scale_compare_20260417}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/tmp_ckpts_20260417}"
HEIGHT="${HEIGHT:-768}"
WIDTH="${WIDTH:-1280}"
FPS="${FPS:-8}"
SEED="${SEED:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
SCALES="${SCALES:-1 5}"
SAVE_INPUT_LQ="${SAVE_INPUT_LQ:-1}"

mkdir -p "${OUTPUT_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${ROOT_DIR}"

CHECKPOINT_PATH="${LOCAL_CKPT_DIR}/m1_train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800_step-1000.safetensors"

for scale in ${SCALES}; do
  run_dir="${OUTPUT_ROOT}/scale${scale}"
  mkdir -p "${run_dir}"
  : > "${run_dir}/run.log"

  for sample_dir in "${VALIDATION_ROOT}"/sample_*; do
    [ -d "${sample_dir}" ] || continue
    sample_name="$(basename "${sample_dir}")"
    input_video="${sample_dir}/lq.mp4"
    output_video="${run_dir}/${sample_name}_sr.mp4"
    meta_copy="${run_dir}/${sample_name}_meta.json"
    val_sr_copy="${run_dir}/${sample_name}_validation_sr.mp4"
    val_hr_copy="${run_dir}/${sample_name}_validation_hr.mp4"
    cp "${sample_dir}/meta.json" "${meta_copy}"
    cp "${sample_dir}/sr.mp4" "${val_sr_copy}"
    cp "${sample_dir}/hr.mp4" "${val_hr_copy}"
    {
      echo "SCALE=${scale}"
      echo "SAMPLE=${sample_name}"
      echo "INPUT_VIDEO=${input_video}"
      echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
      /mnt/conda_envs/flashvsr/bin/python \
        wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py \
        --checkpoint_path "${CHECKPOINT_PATH}" \
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
        --lq_proj_scale "${scale}" \
        --device "${DEVICE}" \
        --torch_dtype "${TORCH_DTYPE}" \
        --tiled \
        $( [ "${SAVE_INPUT_LQ}" = "1" ] && echo "--save_input_lq" )
    } 2>&1 | tee -a "${run_dir}/run.log"
  done
done
