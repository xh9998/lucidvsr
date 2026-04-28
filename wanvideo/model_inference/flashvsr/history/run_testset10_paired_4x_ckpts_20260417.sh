#!/usr/bin/env bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

export TESTSET_ROOT="${TESTSET_ROOT:-/mnt/task_wrapper/user_output/artifacts/input/testset10_paired}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/testset10_paired_4x_ckpts_20260417}"
export LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/inference/tmp_ckpts_20260417}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /mnt/task_runtime/lucidvsr
bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_testset10_paired_ckpts_20260417.sh
