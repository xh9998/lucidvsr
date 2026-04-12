#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
echo "Using Python: $(which python)"

python wanvideo/data/flashvsr/tools/build_storymotion_manifest_from_parquet.py \
  --metadata_url "conductor://lucid-vr/datasets/dryrun_20m/metadata/storymotion-precut-batch1-dryrun_20m_1/1080p/" \
  --output_path /mnt/task_runtime/cloudfix/manifests/storymotion_1080p_manifest_small.jsonl \
  --max_records 1000
