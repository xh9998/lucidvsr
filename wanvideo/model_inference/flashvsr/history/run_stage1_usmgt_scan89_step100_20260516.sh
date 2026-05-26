#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
REPO_DIR="${REPO_DIR:-/mnt/task_runtime/lucidvsr}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/mnt/models/Wan2.1-T2V-1.3B}"
PROMPT_TENSOR_PATH="${PROMPT_TENSOR_PATH:-/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth}"
CKPT_DIR="${CKPT_DIR:-/mnt/task_wrapper/user_output/artifacts/ckpts/usmgt_stage1_20260516}"
INPUT_DIR="${INPUT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
GT_DIR="${GT_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/gt}"
INPUT_S3_DIR="${INPUT_S3_DIR:-s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq}"
GT_S3_DIR="${GT_S3_DIR:-s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/gt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step100_20260516}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MIN_STEP="${MIN_STEP:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "restore INPUT_DIR from ${INPUT_S3_DIR}"
  mkdir -p "${INPUT_DIR}"
  zsh -lc "conductor s3 sync ${INPUT_S3_DIR} ${INPUT_DIR}"
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "missing INPUT_DIR=${INPUT_DIR}" >&2
  exit 2
fi
if [[ ! -d "${GT_DIR}" ]]; then
  echo "restore GT_DIR from ${GT_S3_DIR}"
  mkdir -p "${GT_DIR}"
  zsh -lc "conductor s3 sync ${GT_S3_DIR} ${GT_DIR}" || true
fi

if [[ -d "${GT_DIR}" ]]; then
  mkdir -p "${OUTPUT_ROOT}/synthetic_89f/gt"
  rsync -a --ignore-existing "${GT_DIR}/" "${OUTPUT_ROOT}/synthetic_89f/gt/"
fi
mkdir -p "${OUTPUT_ROOT}/synthetic_89f/lq"
rsync -a --ignore-existing "${INPUT_DIR}/" "${OUTPUT_ROOT}/synthetic_89f/lq/"

IFS=',' read -r -a GPUS <<< "${GPU_IDS}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

mapfile -t CKPTS < <("${PYTHON_BIN}" - <<PY
import glob, os, re
ckpt_dir = "${CKPT_DIR}"
min_step = int("${MIN_STEP}")
items = []
for p in glob.glob(os.path.join(ckpt_dir, "step-*.safetensors")):
    m = re.search(r"step-(\d+)\.safetensors$", os.path.basename(p))
    if not m:
        continue
    step = int(m.group(1))
    if step >= min_step:
        items.append((step, p))
for _, p in sorted(items):
    print(p)
PY
)

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  echo "no ckpts >= ${MIN_STEP} under ${CKPT_DIR}" >&2
  exit 2
fi

echo "input_dir=${INPUT_DIR}"
echo "ckpt_dir=${CKPT_DIR}"
echo "output_root=${OUTPUT_ROOT}"
printf 'ckpts:\n'; printf '  %s\n' "${CKPTS[@]}"

running=0
for idx in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$idx]}"
  step="$(basename "${ckpt}" .safetensors | sed 's/^step-//')"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  out_dir="${OUTPUT_ROOT}/synthetic_89f/usmgt_step${step}"
  log="${OUTPUT_ROOT}/logs/usmgt_step${step}.log"
  mkdir -p "${out_dir}"
  echo "launch step=${step} gpu=${gpu}"
  (
    cd "${REPO_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned_batch.py \
      --checkpoint_path "${ckpt}" \
      --base_model_dir "${BASE_MODEL_DIR}" \
      --prompt_tensor_path "${PROMPT_TENSOR_PATH}" \
      --input_dir "${INPUT_DIR}" \
      --output_dir "${out_dir}" \
      --output_suffix "_sr" \
      --height 768 \
      --width 1280 \
      --num_frames 89 \
      --fps 8 \
      --seed 0 \
      --num_inference_steps 50 \
      --lq_proj_temporal_mode nonstreaming_aligned \
      --input_bicubic_upscale 4.0 \
      --torch_dtype bfloat16 \
      --device cuda \
      --color_fix_method adain \
      --save_input_lq \
      2>&1 | tee "${log}"
  ) &
  running=$((running + 1))
  if [[ "${running}" -ge "${MAX_PARALLEL}" ]]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

find "${OUTPUT_ROOT}" -type f \( -name '*.mp4' -o -name '*.json' \) | sort > "${OUTPUT_ROOT}/manifest_files.txt"
echo "done output_root=${OUTPUT_ROOT}"
