#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
RUN_TS=$(date +%Y%m%d_%H%M%S)
BASE_DIR=/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/takano_parquet_roots_validate_${RUN_TS}
mkdir -p "${BASE_DIR}"

TAKANO1_ROOT="s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled"
TAKANO2_ROOT="s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled"

inspect_one() {
  local tag="$1"
  local parquet_url="$2"
  local out_dir="${BASE_DIR}/${tag}"
  local local_parquet="${out_dir}/$(basename "${parquet_url}")"
  mkdir -p "${out_dir}"
  echo "[inspect] tag=${tag} parquet_url=${parquet_url}" | tee "${out_dir}/run.log"
  conductor s3 cp "${parquet_url}" "${local_parquet}" 2>&1 | tee -a "${out_dir}/run.log"
  "${PYTHON_BIN}" wanvideo/data/flashvsr/tests/inspect_takano_parquet.py \
    --parquet_url "${local_parquet}" \
    2>&1 | tee "${out_dir}/inspect.txt"
}

inspect_one "takano1_00000" "${TAKANO1_ROOT}/00000.parquet"
inspect_one "takano2_00000" "${TAKANO2_ROOT}/00000.parquet"

echo "BASE_DIR=${BASE_DIR}"
