#!/usr/bin/env bash
set -euo pipefail

YUBARI_VIDEO_ROOT="${1:-s3://ve-t2222-datasets/projects/yubari/1.1/data/video/}"

echo "[info] yubari_video_root=${YUBARI_VIDEO_ROOT}"
echo "[info] expected_default_range=000000..006714"

conductor s3 ls "${YUBARI_VIDEO_ROOT}" \
  | grep 'part-.*\.tar$' \
  | awk '{print $4}' \
  | perl -ne 'if(/^part-(\d+)\.tar$/){print "$1\n"}' \
  | sort -n \
  | python3 -c '
import sys
nums = [int(x.strip()) for x in sys.stdin if x.strip()]
if not nums:
    raise SystemExit("No Yubari part-*.tar shards found.")
gaps = [(a + 1, b - 1, b - a - 1) for a, b in zip(nums, nums[1:]) if b - a > 1]
print(f"min={nums[0]:06d}")
print(f"max={nums[-1]:06d}")
print(f"count={len(nums)}")
print(f"has_gap={bool(gaps)}")
print(f"gap_count={len(gaps)}")
for item in gaps[:20]:
    print(f"gap={item[0]:06d}-{item[1]:06d} missing={item[2]}")
'
