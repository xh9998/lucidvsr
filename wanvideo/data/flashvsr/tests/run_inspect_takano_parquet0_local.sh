#!/bin/bash
set -euo pipefail

REPO_ROOT="/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr"
PARQUET_URL="${1:-s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/00000.parquet}"
LOCAL_PARQUET="${2:-/tmp/takano_00000.parquet}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] parquet_url=${PARQUET_URL}"
echo "[info] local_parquet=${LOCAL_PARQUET}"
echo "[info] python_bin=${PYTHON_BIN}"

mkdir -p "$(dirname "${LOCAL_PARQUET}")"

conductor s3 cp "${PARQUET_URL}" "${LOCAL_PARQUET}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
"${PYTHON_BIN}" "${REPO_ROOT}/wanvideo/data/flashvsr/tests/inspect_takano_parquet.py" \
  --parquet_url "${LOCAL_PARQUET}"
