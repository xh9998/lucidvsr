#!/usr/bin/env bash
set -euo pipefail

SEEDVR_ROOT="${SEEDVR_ROOT:-/mnt/task_runtime/SeedVR}"
MODEL_KIND="${MODEL_KIND:?need MODEL_KIND(seedvr1|seedvr2)}"
MODEL_DIR="${MODEL_DIR:?need MODEL_DIR}"
INPUT_DIR="${INPUT_DIR:?need INPUT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?need OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/run.log}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}"
PYTHONPATH_PREFIX="${PYTHONPATH_PREFIX:-/mnt/task_runtime/lucidvsr/third_party_compat}"
CFG_SCALE="${CFG_SCALE:-6.5}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SEED="${SEED:-666}"
OUT_FPS="${OUT_FPS:-8}"
RES_H="${RES_H:-768}"
RES_W="${RES_W:-1280}"
SP_SIZE="${SP_SIZE:-1}"
INPUT_BICUBIC_UPSCALE="${INPUT_BICUBIC_UPSCALE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29531}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
LOCAL_RANK="${LOCAL_RANK:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${SEEDVR_ROOT}"
mkdir -p "${SEEDVR_ROOT}/ckpts"

if [[ "${MODEL_KIND}" == "seedvr1" ]]; then
  INFER_PY="projects/inference_seedvr_3b.py"
  CKPT_LINK="ckpts/seedvr_ema_3b.pth"
  CKPT_SRC="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'seedvr_ema_3b.pth*' | head -1)"
  EXTRA_ARGS=(
    --cfg_scale "${CFG_SCALE}"
    --sample_steps "${SAMPLE_STEPS}"
  )
elif [[ "${MODEL_KIND}" == "seedvr2" ]]; then
  INFER_PY="projects/inference_seedvr2_3b.py"
  CKPT_LINK="ckpts/seedvr2_ema_3b.pth"
  CKPT_SRC="$(find "${MODEL_DIR}" -maxdepth 1 -type f -name 'seedvr2_ema_3b.pth*' | head -1)"
  EXTRA_ARGS=()
else
  echo "unknown MODEL_KIND=${MODEL_KIND}" >&2
  exit 1
fi

if [[ -z "${CKPT_SRC}" ]]; then
  echo "checkpoint not found in MODEL_DIR=${MODEL_DIR}" >&2
  exit 1
fi

ln -sfn "${CKPT_SRC}" "${SEEDVR_ROOT}/${CKPT_LINK}"
ln -sfn "${MODEL_DIR}/ema_vae.pth" "${SEEDVR_ROOT}/ckpts/ema_vae.pth"
if [[ -f "${MODEL_DIR}/pos_emb.pt" ]]; then
  ln -sfn "${MODEL_DIR}/pos_emb.pt" "${SEEDVR_ROOT}/pos_emb.pt"
fi
if [[ -f "${MODEL_DIR}/neg_emb.pt" ]]; then
  ln -sfn "${MODEL_DIR}/neg_emb.pt" "${SEEDVR_ROOT}/neg_emb.pt"
fi

WORK_INPUT_DIR="${INPUT_DIR}"
if [[ "${INPUT_BICUBIC_UPSCALE}" != "1" ]]; then
  WORK_INPUT_DIR="${OUTPUT_DIR}/_input_upx${INPUT_BICUBIC_UPSCALE}"
  mkdir -p "${WORK_INPUT_DIR}"
  shopt -s nullglob
  for input_video in "${INPUT_DIR}"/*.mp4; do
    name="$(basename "${input_video}")"
    out_video="${WORK_INPUT_DIR}/${name}"
    read -r width height < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "${input_video}" | tr 'x' ' ')
    out_w=$(python - <<PY
width=${width}
scale=${INPUT_BICUBIC_UPSCALE}
print(max(1, int(round(width * float(scale)))))
PY
)
    out_h=$(python - <<PY
height=${height}
scale=${INPUT_BICUBIC_UPSCALE}
print(max(1, int(round(height * float(scale)))))
PY
)
    ffmpeg -y -v error -i "${input_video}" -vf "scale=${out_w}:${out_h}:flags=bicubic" -an "${out_video}"
  done
  shopt -u nullglob
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
RANK="${RANK}" \
WORLD_SIZE="${WORLD_SIZE}" \
LOCAL_RANK="${LOCAL_RANK}" \
PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH:-}" \
"${SEEDVR_PYTHON}" "${INFER_PY}" \
  --video_path "${WORK_INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --res_h "${RES_H}" \
  --res_w "${RES_W}" \
  --sp_size "${SP_SIZE}" \
  --out_fps "${OUT_FPS}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
