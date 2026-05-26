#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_every100_20260506}"
CKPT_S3_DIR="${CKPT_S3_DIR:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output}"
SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
REAL_INPUT_DIR="${REAL_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_scan89_every100_20260506_by_dataset}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://lxh/data/test/stage2_v6_scan89_every100_20260506_by_dataset}"

mkdir -p "${CKPT_DIR}" "${OUTPUT_ROOT}/logs"

tmp_list="${OUTPUT_ROOT}/logs/stage2_v6_s3_ckpt_list.txt"
conductor s3 ls "${CKPT_S3_DIR}/" | awk '/step-[0-9]+\.safetensors/ {print $NF}' | sort -V > "${tmp_list}"
if [[ ! -s "${tmp_list}" ]]; then
  echo "[error] no remote step-*.safetensors under ${CKPT_S3_DIR}" >&2
  exit 1
fi

while read -r name; do
  step="${name#step-}"
  step="${step%.safetensors}"
  if [[ "${step}" =~ ^[0-9]+$ ]] && (( step >= 100 )) && (( step % 100 == 0 )); then
    if [[ ! -f "${CKPT_DIR}/${name}" ]]; then
      echo "[download] ${name}"
      conductor s3 cp "${CKPT_S3_DIR}/${name}" "${CKPT_DIR}/${name}"
    fi
  fi
done < "${tmp_list}"

export PYTHON_BIN
cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m py_compile \
  wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py \
  wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1_batch.py

env \
  ROOT_DIR="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  CKPT_DIR="${CKPT_DIR}" \
  SYNTHETIC_INPUT_DIR="${SYNTHETIC_INPUT_DIR}" \
  REAL_INPUT_DIR="${REAL_INPUT_DIR}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  S3_OUTPUT_DIR="${S3_OUTPUT_DIR}" \
  SYNC_TO_S3=1 \
  START_OCCUPY_AFTER=1 \
  GPU_LIST="0,1,2,3,4,5,6,7" \
  MAX_PARALLEL=8 \
  MIN_STEP=100 \
  STEP_MOD=100 \
  bash "${REPO_ROOT}/wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_every100_ckpts_20260506.sh"
