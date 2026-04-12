#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
echo "Using Python: $(which python)"

python wanvideo/data/flashvsr/tests/test_streaming_dataset_minimal.py \
  --internal_url "conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier1-qwen3/1080p/,conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier2-qwen3/1080p/" \
  --height 768 \
  --width 1280 \
  --num_frames 89 \
  --max_source_frames 160 \
  --enable_degradation \
  --global_seed 20260407 \
  --num_samples 5 \
  --save_dir /mnt/task_wrapper/user_output/artifacts/exp/test_outputs/takano_shards
