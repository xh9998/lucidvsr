#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"
if [[ "${PYTHON_BIN}" == "python" || "${PYTHON_BIN}" == "python3" || ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="/mnt/conda_envs/flashvsr/bin/python"
fi
if [[ "${SEEDVR_PYTHON}" == "python" || "${SEEDVR_PYTHON}" == "python3" || ! -x "${SEEDVR_PYTHON}" ]]; then
  SEEDVR_PYTHON="/mnt/conda_envs/seedvr/bin/python"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_50random_medium_synthetic_20260518}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
SYNTHETIC_S3="${SYNTHETIC_S3:-s3://lxh/data/test/testset50_89f_random25takano25yubari_medium_x4_lq_20260518}"
OUTPUT_S3="${OUTPUT_S3:-s3://lxh/data/test/ppt_benchmark_50random_medium_synthetic_20260518}"
CRITICAL_S3="${CRITICAL_S3:-s3://lxh/models/flashvsr/critical_ppt_20260518}"
CRITICAL_LOCAL="${CRITICAL_LOCAL:-/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
FLASHVSR_MODEL_DIR="${FLASHVSR_MODEL_DIR:-/mnt/models/FlashVSR-v1.1}"
SEEDVR3B_MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}"
SEEDVR2_3B_MODEL_DIR="${SEEDVR2_3B_MODEL_DIR:-/mnt/models/SeedVR2-3B}"

STAGE1_535_SRC="${STAGE1_535_SRC:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors}"
STAGE1_USMGT_SRC="${STAGE1_USMGT_SRC:-s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors}"
STAGE2_641_SRC="${STAGE2_641_SRC:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors}"
STAGE3_D32_SRC="${STAGE3_D32_SRC:-s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/step-2000.safetensors}"

STAGE1_535_CKPT="${CRITICAL_LOCAL}/stage1_v535_step10000.safetensors"
STAGE1_USMGT_CKPT="${CRITICAL_LOCAL}/stage1_usmgt_takano20250205_step3000.safetensors"
STAGE2_641_CKPT="${CRITICAL_LOCAL}/stage2_v641_step6000.safetensors"
STAGE3_D32_CKPT="${CRITICAL_LOCAL}/stage3_v7d32_step2000.safetensors"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
SYNC_TO_S3="${SYNC_TO_S3:-1}"
START_OCCUPY_AFTER="${START_OCCUPY_AFTER:-1}"
FINAL_OCCUPY_GPUS="${FINAL_OCCUPY_GPUS:-0,1,2,3,4,5,6,7}"
SPARE_OCCUPY_GPUS="${SPARE_OCCUPY_GPUS:-7}"
SEED="${SEED:-0}"
FPS="${FPS:-8}"

mkdir -p "${OUTPUT_ROOT}/logs" "${CRITICAL_LOCAL}" "${OUTPUT_ROOT}/_inputs/synthetic_89f"
cd "${ROOT_DIR}"

fetch_file() {
  local src="$1"
  local dst="$2"
  if [[ -s "${dst}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${dst}")"
  echo "[fetch] ${src} -> ${dst}"
  if [[ "${src}" == /* ]]; then
    cp "${src}" "${dst}"
  else
    conductor s3 cp "${src}" "${dst}"
  fi
}

prepare_inputs() {
  local src_dir="$1"
  local dst_dir="$2"
  rm -rf "${dst_dir}"
  mkdir -p "${dst_dir}"
  find "${src_dir}" -maxdepth 1 -type f -name '*.mp4' -print0 | sort -z | while IFS= read -r -d '' item; do
    ln -s "${item}" "${dst_dir}/$(basename "${item}")"
  done
}

count_inputs() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.mp4' | wc -l | tr -d ' '
}

count_outputs() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' '
}

record_time() {
  local method="$1"
  local seconds="$2"
  local status="$3"
  local input_dir="$4"
  local output_dir="$5"
  local n
  n="$(count_inputs "${input_dir}")"
  {
    echo "method=${method}"
    echo "dataset=synthetic_89f_random50_medium"
    echo "status=${status}"
    echo "seconds=${seconds}"
    echo "num_inputs=${n}"
    echo "num_outputs=$(count_outputs "${output_dir}")"
    awk -v s="${seconds}" -v n="${n}" 'BEGIN { if (n > 0) printf("seconds_per_video=%.3f\n", s/n); }'
    echo "input_dir=${input_dir}"
    echo "output_dir=${output_dir}"
  } | tee "${OUTPUT_ROOT}/logs/${method}.time"
}

run_timed_job() {
  local gpu="$1"
  local method="$2"
  local input_dir="$3"
  local output_dir="$4"
  shift 4
  mkdir -p "${output_dir}"
  local start end status log_file time_file num_in num_out old_status
  log_file="${OUTPUT_ROOT}/logs/${method}.log"
  time_file="${OUTPUT_ROOT}/logs/${method}.time"
  num_in="$(count_inputs "${input_dir}")"
  num_out="$(count_outputs "${output_dir}")"
  old_status="$(grep -E '^status=' "${time_file}" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
  if [[ "${old_status}" == "0" && "${num_in}" != "0" && "${num_out}" == "${num_in}" ]]; then
    echo "[skip_done] gpu=${gpu} method=${method} outputs=${num_out}/${num_in}"
    return 0
  fi
  echo "[job] gpu=${gpu} method=${method} input=${num_in}"
  start="$(date +%s)"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" CUDA_DEVICE="${gpu}" "$@" 2>&1 | tee "${log_file}"
  status="${PIPESTATUS[0]}"
  set -e
  end="$(date +%s)"
  record_time "${method}" "$((end - start))" "${status}" "${input_dir}" "${output_dir}"
  if [[ "${status}" != "0" ]]; then
    echo "[job_failed] method=${method} status=${status}" >&2
    return "${status}"
  fi
}

run_flashvsr_official() {
  local gpu="$1" input_dir="$2" output_dir="$3"
  run_timed_job "${gpu}" flashvsr_official "${input_dir}" "${output_dir}" \
    env FLASHVSR_PYTHON_BIN="${PYTHON_BIN}" MODEL_DIR="${FLASHVSR_MODEL_DIR}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${output_dir}" \
      SEED="${SEED}" SCALE=4 TILED_FLAG=1 bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh"
}

run_seedvr() {
  local gpu="$1" method="$2" kind="$3" model_dir="$4" input_dir="$5" output_dir="$6"
  run_timed_job "${gpu}" "${method}" "${input_dir}" "${output_dir}" \
    env SEEDVR_PYTHON="${SEEDVR_PYTHON}" MODEL_KIND="${kind}" MODEL_DIR="${model_dir}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${output_dir}" \
      RES_H=768 RES_W=1280 OUT_FPS="${FPS}" SEED="${SEED}" MASTER_PORT="$((29600 + gpu))" \
      LOG_FILE="${OUTPUT_ROOT}/logs/${method}_inner.log" bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh"
}

run_stage1() {
  local gpu="$1" method="$2" ckpt="$3" input_dir="$4" output_dir="$5"
  run_timed_job "${gpu}" "${method}" "${input_dir}" "${output_dir}" \
    env PYTHON_BIN="${PYTHON_BIN}" CHECKPOINT_PATH="${ckpt}" INPUT_DIR="${input_dir}" OUTPUT_DIR="${output_dir}" \
      BASE_MODEL_DIR="${BASE_MODEL_DIR}" PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH}" NUM_FRAMES=89 FPS="${FPS}" SEED="${SEED}" \
      NUM_INFERENCE_STEPS=50 INPUT_BICUBIC_UPSCALE=4.0 LQ_PROJ_TEMPORAL_MODE=nonstreaming_aligned COLOR_FIX_METHOD=adain \
      bash "${ROOT_DIR}/wanvideo/model_inference/flashvsr/history/run_stage1_v5_3_aligned_dir.sh"
}

run_stage2_like() {
  local gpu="$1" method="$2" ckpt="$3" steps="$4" input_dir="$5" output_dir="$6"
  run_timed_job "${gpu}" "${method}" "${input_dir}" "${output_dir}" \
    env PYTHON_BIN="${PYTHON_BIN}" "${PYTHON_BIN}" -u "${ROOT_DIR}/wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1_batch.py" \
      --checkpoint_path "${ckpt}" --base_model_dir "${BASE_MODEL_DIR}" --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${input_dir}" --output_dir "${output_dir}" --height 768 --width 1280 --num_frames 89 --fps "${FPS}" --seed "${SEED}" \
      --num_inference_steps "${steps}" --stage2_attention_mode block_sparse_chunk_causal --stage2_topk_ratio 2.0 --stage2_local_num -1 \
      --stage2_kv_ratio 3.0 --input_bicubic_upscale 4.0 --color_fix_method adain
}

run_stage3() {
  local gpu="$1" method="$2" ckpt="$3" input_dir="$4" output_dir="$5"
  run_timed_job "${gpu}" "${method}" "${input_dir}" "${output_dir}" \
    env PYTHON_BIN="${PYTHON_BIN}" "${PYTHON_BIN}" -u "${ROOT_DIR}/wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py" \
      --checkpoint_path "${ckpt}" --base_model_dir "${BASE_MODEL_DIR}" --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${input_dir}" --output_dir "${output_dir}" --height 768 --width 1280 --num_frames 89 --fps "${FPS}" --seed "${SEED}" \
      --num_inference_steps 1 --stage2_attention_mode block_sparse_chunk_causal --stage2_topk_ratio 2.0 --stage2_local_num -1 \
      --stage2_kv_ratio 3.0 --input_bicubic_upscale 4.0 --color_fix_method adain
}

echo "[prepare] restoring closeup dataset and critical checkpoints"
if [[ ! -d "${SYNTHETIC_ROOT}/lq" ]]; then
  conductor s3 sync "${SYNTHETIC_S3}" "${SYNTHETIC_ROOT}"
fi
fetch_file "${STAGE1_535_SRC}" "${STAGE1_535_CKPT}"
fetch_file "${STAGE1_USMGT_SRC}" "${STAGE1_USMGT_CKPT}"
fetch_file "${STAGE2_641_SRC}" "${STAGE2_641_CKPT}"
fetch_file "${STAGE3_D32_SRC}" "${STAGE3_D32_CKPT}"
if [[ "${SYNC_TO_S3}" == "1" ]]; then
  conductor s3 sync "${CRITICAL_LOCAL}" "${CRITICAL_S3}"
fi

prepare_inputs "${SYNTHETIC_ROOT}/lq" "${OUTPUT_ROOT}/_inputs/synthetic_89f"
mkdir -p "${OUTPUT_ROOT}/synthetic_89f"
rm -rf "${OUTPUT_ROOT}/synthetic_89f/gt" "${OUTPUT_ROOT}/synthetic_89f/lq"
cp -a "${SYNTHETIC_ROOT}/gt" "${OUTPUT_ROOT}/synthetic_89f/gt"
cp -a "${SYNTHETIC_ROOT}/lq" "${OUTPUT_ROOT}/synthetic_89f/lq"

{
  echo "benchmark=ppt_benchmark_50random_medium_synthetic_20260518"
  echo "synthetic_root=${SYNTHETIC_ROOT}"
  echo "synthetic_s3=${SYNTHETIC_S3}"
  echo "critical_s3=${CRITICAL_S3}"
  echo "output_s3=${OUTPUT_S3}"
  echo "gpu_rule=one GPU runs one method job; no stacked jobs on one GPU"
  echo "gpu_list=${GPU_LIST}"
  echo "spare_occupy_gpus=${SPARE_OCCUPY_GPUS}"
  echo "final_occupy_gpus=${FINAL_OCCUPY_GPUS}"
  echo "stage1_535_ckpt=${STAGE1_535_CKPT}"
  echo "stage1_usmgt_ckpt=${STAGE1_USMGT_CKPT}"
  echo "stage2_641_ckpt=${STAGE2_641_CKPT}"
  echo "stage3_d32_ckpt=${STAGE3_D32_CKPT}"
} | tee "${OUTPUT_ROOT}/settings.txt"

if [[ -n "${SPARE_OCCUPY_GPUS}" ]]; then
  echo "[occupy] starting spare occupy on GPUS=${SPARE_OCCUPY_GPUS}"
  tmux kill-session -t occupy_pptbench50_spare 2>/dev/null || true
  tmux new-session -d -s occupy_pptbench50_spare "PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python GPUS='${SPARE_OCCUPY_GPUS}' bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
declare -a JOBS=(
  "flashvsr_official"
  "seedvr3b"
  "seedvr2_3b"
  "stage1_535_step10000"
  "stage1_usmgt_step3000"
  "stage2_v641_step6000"
  "stage3_v7d32_step2000"
)

run_one_job() {
  local gpu="$1" method="$2"
  local input_dir="${OUTPUT_ROOT}/_inputs/synthetic_89f"
  local output_dir="${OUTPUT_ROOT}/synthetic_89f/${method}"
  case "${method}" in
    flashvsr_official) run_flashvsr_official "${gpu}" "${input_dir}" "${output_dir}" ;;
    seedvr3b) run_seedvr "${gpu}" seedvr3b seedvr1 "${SEEDVR3B_MODEL_DIR}" "${input_dir}" "${output_dir}" ;;
    seedvr2_3b) run_seedvr "${gpu}" seedvr2_3b seedvr2 "${SEEDVR2_3B_MODEL_DIR}" "${input_dir}" "${output_dir}" ;;
    stage1_535_step10000) run_stage1 "${gpu}" stage1_535_step10000 "${STAGE1_535_CKPT}" "${input_dir}" "${output_dir}" ;;
    stage1_usmgt_step3000) run_stage1 "${gpu}" stage1_usmgt_step3000 "${STAGE1_USMGT_CKPT}" "${input_dir}" "${output_dir}" ;;
    stage2_v641_step6000) run_stage2_like "${gpu}" stage2_v641_step6000 "${STAGE2_641_CKPT}" 50 "${input_dir}" "${output_dir}" ;;
    stage3_v7d32_step2000) run_stage3 "${gpu}" stage3_v7d32_step2000 "${STAGE3_D32_CKPT}" "${input_dir}" "${output_dir}" ;;
    *) echo "unknown method=${method}" >&2; return 2 ;;
  esac
}

next_job=0
while [[ "${next_job}" -lt "${#JOBS[@]}" ]]; do
  declare -a pids=()
  for gpu in "${GPUS[@]}"; do
    if [[ "${next_job}" -ge "${#JOBS[@]}" ]]; then
      break
    fi
    run_one_job "${gpu}" "${JOBS[${next_job}]}" &
    pids+=("$!")
    next_job=$((next_job + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/all_outputs.txt"
cat "${OUTPUT_ROOT}"/logs/*.time | tee "${OUTPUT_ROOT}/timing_summary.txt"

if [[ "${SYNC_TO_S3}" == "1" ]]; then
  conductor s3 sync "${OUTPUT_ROOT}" "${OUTPUT_S3}"
fi

if [[ "${START_OCCUPY_AFTER}" == "1" ]]; then
  echo "[occupy] restarting normal occupy on GPUS=${FINAL_OCCUPY_GPUS}"
  pkill -f '[g]pu_stress_tc.py' || true
  pkill -f '[g]pu_stress_tc.sh' || true
  sleep 2
  tmux kill-session -t occupy_pptbench_closeup 2>/dev/null || true
  tmux kill-session -t occupy_pptbench50_spare 2>/dev/null || true
  tmux new-session -d -s occupy_pptbench50 "PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python GPUS='${FINAL_OCCUPY_GPUS}' bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh"
fi

echo "[done] output_root=${OUTPUT_ROOT}"
