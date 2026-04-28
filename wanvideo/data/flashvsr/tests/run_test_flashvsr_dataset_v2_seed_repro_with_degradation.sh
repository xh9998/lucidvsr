#!/bin/bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
source /mnt/task_runtime/bolt_lxh/use_active_python.sh

PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
RUN_TS=$(date +%Y%m%d_%H%M%S)
BASE_DIR=/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_seed_repro_with_degradation_${RUN_TS}
mkdir -p "$BASE_DIR"

TAKANO_METADATA_URL="s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/,s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/"
IMAGE_METADATA_URL="s3://takano-assets/20231106/high_resolution/metadata_split_parquet/train/apple_gen_20231117_full_metadata_6_5_0.parquet"
YUBARI_VIDEO_TAR_URL="s3://ve-t2222-datasets/projects/yubari/1.1/data/video/"

run_once() {
  local mode="$1"
  local seed="$2"
  local tag="$3"
  local out_dir="${BASE_DIR}/${tag}"
  mkdir -p "$out_dir"
  "$PYTHON_BIN" wanvideo/data/flashvsr/tests/test_flashvsr_dataset_v2_sources.py \
    --mode "$mode" \
    --takano_metadata_url "$TAKANO_METADATA_URL" \
    --image_metadata_url "$IMAGE_METADATA_URL" \
    --image_as_single_frame \
    --yubari_video_tar_url "$YUBARI_VIDEO_TAR_URL" \
    --height 768 \
    --width 1280 \
    --num_frames 17 \
    --max_source_frames 160 \
    --enable_degradation \
    --global_seed "$seed" \
    --num_samples 3 \
    --max_parquet_records 64 \
    --max_yubari_records 128 \
    --save_dir "$out_dir" \
    2>&1 | tee "$out_dir/run.log"
}

run_once takano 1 takano_seed1_run1
run_once takano 1 takano_seed1_run2
run_once takano 2 takano_seed2_run1

run_once image 1 image_seed1_run1
run_once image 1 image_seed1_run2
run_once image 2 image_seed2_run1

run_once yubari 1 yubari_seed1_run1
run_once yubari 1 yubari_seed1_run2
run_once yubari 2 yubari_seed2_run1

"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/takano_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/takano_seed1_run2/summary.json" \
  --output_path "$BASE_DIR/takano_seed1_vs_seed1.json"
"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/takano_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/takano_seed2_run1/summary.json" \
  --output_path "$BASE_DIR/takano_seed1_vs_seed2.json"

"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/image_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/image_seed1_run2/summary.json" \
  --output_path "$BASE_DIR/image_seed1_vs_seed1.json"
"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/image_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/image_seed2_run1/summary.json" \
  --output_path "$BASE_DIR/image_seed1_vs_seed2.json"

"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/yubari_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/yubari_seed1_run2/summary.json" \
  --output_path "$BASE_DIR/yubari_seed1_vs_seed1.json"
"$PYTHON_BIN" wanvideo/data/flashvsr/tests/compare_seed_runs_v2.py \
  --lhs "$BASE_DIR/yubari_seed1_run1/summary.json" \
  --rhs "$BASE_DIR/yubari_seed2_run1/summary.json" \
  --output_path "$BASE_DIR/yubari_seed1_vs_seed2.json"

echo "BASE_DIR=${BASE_DIR}"
