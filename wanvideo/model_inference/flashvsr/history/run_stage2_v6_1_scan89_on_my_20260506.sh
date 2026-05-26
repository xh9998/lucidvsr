#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_worker2_scan89_v61_20260506}"
RUN_LOG="${OUTPUT_ROOT}/run_rerun.log"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

echo "[info] stopping occupy jobs on non-baseline GPUs"
pkill -f "gpu_stress_tc.py" || true
sleep 2

echo "[info] keeping GPUs 6,7 occupied while v6.1 inference uses 2,3,4,5"
bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh 6,7 >/tmp/occupy_6_7_v61.log 2>&1 &
OCCUPY_PID=$!

status=0
(
  export FLASHVSR_PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
  export CUDA_DEVICE_LIST=2,3,4,5
  export OUTPUT_ROOT="${OUTPUT_ROOT}"
  export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
  bash /mnt/task_runtime/lucidvsr/wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_ckpts_20260506.sh
) 2>&1 | tee "${RUN_LOG}" || status=$?

echo "[info] restoring occupy jobs on GPUs 2-7"
kill "${OCCUPY_PID}" 2>/dev/null || true
pkill -f "gpu_stress_tc.py" || true
bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh 2,3,4,5,6,7 >/tmp/occupy_2_7_after_v61.log 2>&1 &

echo "[done] status=${status} output_root=${OUTPUT_ROOT}"
exit "${status}"
