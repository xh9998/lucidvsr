#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output_manifest.txt> <root1> [root2 ...]" >&2
  exit 1
fi

OUTPUT_PATH="$1"
shift

mkdir -p "$(dirname "$OUTPUT_PATH")"
TMP_PATH="${OUTPUT_PATH}.tmp.$$"
trap 'rm -f "$TMP_PATH"' EXIT
: > "$TMP_PATH"

for ROOT in "$@"; do
  echo "[manifest] listing $ROOT" >&2
  conductor s3 ls "$ROOT" \
    | sed -n 's#.*\(s3://[^ ]*\.tar\)$#\1#p' \
    >> "$TMP_PATH"
done

sort -u "$TMP_PATH" > "$OUTPUT_PATH"
COUNT="$(wc -l < "$OUTPUT_PATH" | tr -d ' ')"
echo "[manifest] wrote $COUNT tar paths to $OUTPUT_PATH" >&2
