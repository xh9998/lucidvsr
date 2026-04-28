#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "flashvsr" ]]; then
  conda activate flashvsr
fi
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

cd /mnt/task_runtime/lucidvsr

CFG="${CFG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v2_17f_alpha5_repro_from_m1_step1000.yaml}"
OUT_ROOT="${OUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/v2_block0_probe_$(date +%Y%m%d_%H%M%S)}"

python wanvideo/model_training/flashvsr/tests/check_v2_dit_probe.py \
  --config "${CFG}" \
  --output_dir "${OUT_ROOT}" \
  --variant single_eval

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --mixed_precision bf16 --num_processes 2 \
  wanvideo/model_training/flashvsr/tests/check_v2_dit_probe.py \
  --config "${CFG}" \
  --output_dir "${OUT_ROOT}" \
  --variant dist2_eval \
  --compare_to "${OUT_ROOT}/single_eval"

echo "${OUT_ROOT}"
