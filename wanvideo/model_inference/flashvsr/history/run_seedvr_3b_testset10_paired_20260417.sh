#!/usr/bin/env bash
set -euo pipefail

SEEDVR_ROOT="${SEEDVR_ROOT:-/mnt/task_runtime/SeedVR}"
MODEL_DIR="${MODEL_DIR:-/mnt/models/SeedVR-3B}"
TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_paired}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/seedvr_3b_testset10_paired_20260417}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
INPUT_SUBDIR="${INPUT_SUBDIR:-lq}"
CFG_SCALE="${CFG_SCALE:-6.5}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SEED="${SEED:-666}"
RES_H="${RES_H:-768}"
RES_W="${RES_W:-1280}"
SP_SIZE="${SP_SIZE:-1}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/task_runtime/lucidvsr/third_party_compat}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
LOCAL_RANK="${LOCAL_RANK:-0}"

mkdir -p "${OUTPUT_ROOT}"
cd "${SEEDVR_ROOT}"
mkdir -p "${SEEDVR_ROOT}/ckpts"
ln -sfn "${MODEL_DIR}/seedvr_ema_3b.pth" "${SEEDVR_ROOT}/ckpts/seedvr_ema_3b.pth"
ln -sfn "${MODEL_DIR}/ema_vae.pth" "${SEEDVR_ROOT}/ckpts/ema_vae.pth"

run_variant() {
  local variant="$1"
  local video_path="${TESTSET_ROOT}/${variant}/${INPUT_SUBDIR}"
  local output_dir="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${output_dir}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="${MASTER_PORT}" \
  RANK="${RANK}" \
  WORLD_SIZE="${WORLD_SIZE}" \
  LOCAL_RANK="${LOCAL_RANK}" \
  PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
  "${SEEDVR_PYTHON}" projects/inference_seedvr_3b.py \
    --video_path "${video_path}" \
    --output_dir "${output_dir}" \
    --cfg_scale "${CFG_SCALE}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --seed "${SEED}" \
    --res_h "${RES_H}" \
    --res_w "${RES_W}" \
    --sp_size "${SP_SIZE}" \
    2>&1 | tee "${output_dir}/run.log"
}

run_variant "testset10_17f"
run_variant "testset10_89f"
