#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/task_runtime/lucidvsr}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"

CKPT="${CKPT:-/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_4_1_89f_step200_20260512/step-1800.safetensors}"
CKPT_S3="${CKPT_S3:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-1800.safetensors}"
SOURCE_INPUT_DIR="${SOURCE_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
PADDED_INPUT_DIR="${PADDED_INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_tailpad85_20260512/lq}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_step1800_v61_tailpad85_20260512}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-s3://lxh/tmp/stage2_v6_4_1_step1800_v61_tailpad85_20260512}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${CKPT}")" "${PADDED_INPUT_DIR}" "${OUTPUT_ROOT}/logs"

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python_bin=${PYTHON_BIN}"
echo "[info] ckpt=${CKPT}"
echo "[info] source_input_dir=${SOURCE_INPUT_DIR}"
echo "[info] padded_input_dir=${PADDED_INPUT_DIR}"
echo "[info] output_root=${OUTPUT_ROOT}"

if [[ ! -s "${CKPT}" ]]; then
  echo "[download] ${CKPT_S3}"
  conductor s3 cp "${CKPT_S3}" "${CKPT}"
fi

echo "[info] stopping occupy / stale inference"
tmux kill-session -t occupy 2>/dev/null || true
pkill -f gpu_stress_tc.py || true
pkill -f infer_flashvsr_stage2_v6_1_batch || true
sleep 2

echo "[info] building 85+tail4 padded inputs"
export SOURCE_INPUT_DIR
export PADDED_INPUT_DIR
"${PYTHON_BIN}" - <<'PY'
import os
import imageio.v3 as iio
from pathlib import Path

src_dir = Path(os.environ["SOURCE_INPUT_DIR"])
dst_dir = Path(os.environ["PADDED_INPUT_DIR"])
dst_dir.mkdir(parents=True, exist_ok=True)

for src in sorted(src_dir.glob("*.mp4")):
    frames = list(iio.imiter(src))
    if len(frames) < 85:
        raise RuntimeError(f"{src} has only {len(frames)} frames, need >=85")
    out = frames[:85] + [frames[84]] * 4
    dst = dst_dir / src.name
    iio.imwrite(dst, out, fps=8, codec="libx264", pixelformat="yuv420p")
    print(f"[pad85] {src.name}: {len(frames)} -> {len(out)}")
PY

input_count="$(find "${PADDED_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ')"
if [[ "${input_count}" -le 0 ]]; then
  echo "[error] no padded input mp4 found" >&2
  exit 1
fi

echo "[info] launching step-1800 tailpad85 inference, input_count=${input_count}"
rm -rf "${OUTPUT_ROOT}/_input_splits"
mkdir -p "${OUTPUT_ROOT}/_input_splits"/part{0,1,2,3} "${OUTPUT_ROOT}/step-1800_tailpad85"
idx=0
while IFS= read -r input; do
  part=$((idx % 4))
  ln -sf "${input}" "${OUTPUT_ROOT}/_input_splits/part${part}/$(basename "${input}")"
  idx=$((idx + 1))
done < <(find "${PADDED_INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort)

run_part() {
  local gpu="$1"
  local part="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${PYTHON_BIN}" -m wanvideo.model_inference.flashvsr.infer_flashvsr_stage2_v6_1_batch \
      --checkpoint_path "${CKPT}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${OUTPUT_ROOT}/_input_splits/part${part}" \
      --output_dir "${OUTPUT_ROOT}/step-1800_tailpad85" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --num_inference_steps 50 \
      --lq_proj_scale 1.0 \
      --stage2_attention_mode block_sparse_chunk_causal \
      --stage2_topk_ratio 2.0 \
      --stage2_local_num -1 \
      --stage2_kv_ratio 3.0 \
      --input_bicubic_upscale 4.0 \
      --color_fix_method adain \
      > "${OUTPUT_ROOT}/logs/step-1800_tailpad85_part${part}.log" 2>&1
}

pids=()
for part in 0 1 2 3; do
  run_part "${part}" "${part}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" != "0" ]]; then
  echo "[error] one or more tailpad85 inference workers failed" >&2
  exit "${status}"
fi

find "${OUTPUT_ROOT}" -type f -name '*.mp4' | sort > "${OUTPUT_ROOT}/outputs.txt"
find "${OUTPUT_ROOT}/step-1800_tailpad85" -maxdepth 1 -type f -name '*.mp4' | wc -l | tr -d ' ' > "${OUTPUT_ROOT}/summary_counts.txt"

echo "[info] syncing output to ${S3_OUTPUT_DIR}"
conductor s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_DIR}"

echo "[info] restoring occupy"
tmux kill-session -t occupy 2>/dev/null || true
pkill -f gpu_stress_tc.py || true
tmux new-session -d -s occupy "${PYTHON_BIN} /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.py --fp16 --size 147456 --repeat 50"

echo "[done] output_root=${OUTPUT_ROOT}"
