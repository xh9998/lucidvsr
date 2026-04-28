#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR=/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/image_row_count_${RUN_TS}
mkdir -p "$OUT_DIR"

METADATA_URL="${1:-s3://takano-assets/20231106/high_resolution/metadata_split_parquet/train/}"
MAX_FILES="${2:-}"

CMD=(
  "$PYTHON_BIN"
  wanvideo/data/flashvsr/tools/count_image_parquet_rows.py
  --metadata_url "$METADATA_URL"
  --output_path "$OUT_DIR/result.json"
)

if [[ -n "$MAX_FILES" ]]; then
  CMD+=(--max_files "$MAX_FILES")
fi

"${CMD[@]}" 2>&1 | tee "$OUT_DIR/run.log"
echo "OUT_DIR=$OUT_DIR"
