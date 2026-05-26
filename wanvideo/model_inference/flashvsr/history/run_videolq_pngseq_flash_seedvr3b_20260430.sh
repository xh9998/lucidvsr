#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/task_runtime/lucidvsr}"
RAW_ROOT="${RAW_ROOT:-/mnt/task_wrapper/user_output/artifacts/data/inference/videolq_raw_pngseq_20260430}"
VIDEO_DIR="${VIDEO_DIR:-/mnt/task_wrapper/user_output/artifacts/data/inference/videolq_17f_fps8_native_20260430_fixed}"
OUT_ROOT="${OUT_ROOT:-/mnt/task_wrapper/user_output/artifacts/inference/compare_videolq_native17_fps8_flash_seedvr3b_20260430_fixed}"
TARBALL_S3="${TARBALL_S3:-s3://lxh/data/test/videolq_17f_pngseq_20260430.tar.gz}"
TARBALL_LOCAL="${TARBALL_LOCAL:-/tmp/videolq_17f_pngseq_20260430.tar.gz}"

mkdir -p "${RAW_ROOT}" "${OUT_ROOT}/logs"

if [[ ! -d "${RAW_ROOT}/videolq" ]]; then
  conductor s3 cp "${TARBALL_S3}" "${TARBALL_LOCAL}"
  rm -rf "${RAW_ROOT}"
  mkdir -p "${RAW_ROOT}"
  tar -C "${RAW_ROOT}" --exclude='._*' --exclude='*/._*' -xzf "${TARBALL_LOCAL}"
fi

find "${RAW_ROOT}/videolq" -type f -name '._*.png' -delete
rm -rf "${VIDEO_DIR}" "${OUT_ROOT}/flashvsr_official" "${OUT_ROOT}/seedvr3b"
mkdir -p "${VIDEO_DIR}" "${OUT_ROOT}/logs"

count=0
bad=0
shopt -s nullglob
for dir in "${RAW_ROOT}"/videolq/*; do
  [[ -d "${dir}" ]] || continue
  name="$(basename "${dir}")"
  list_file="/tmp/videolq_fixed_${name}.txt"
  : > "${list_file}"
  while IFS= read -r frame; do
    printf "file '%s'\n" "${frame}" >> "${list_file}"
  done < <(find "${dir}" -maxdepth 1 -type f -iname '*.png' ! -name '._*' | sort | head -17)

  frame_count="$(wc -l < "${list_file}" | tr -d ' ')"
  if [[ "${frame_count}" != "17" ]]; then
    echo "bad_input_count ${name} ${frame_count}" >&2
    bad=$((bad + 1))
    continue
  fi

  out_video="${VIDEO_DIR}/${name}_17f_fps8.mp4"
  if ! ffmpeg -y -v error -r 8 -f concat -safe 0 -i "${list_file}" -frames:v 17 -pix_fmt yuv420p "${out_video}"; then
    echo "ffmpeg_fail ${name}" >&2
    bad=$((bad + 1))
    continue
  fi
  out_frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "${out_video}")"
  if [[ "${out_frames}" != "17" ]]; then
    echo "bad_output_frames ${name} ${out_frames}" >&2
    rm -f "${out_video}"
    bad=$((bad + 1))
    continue
  fi
  count=$((count + 1))
done
shopt -u nullglob

{
  echo "prepared_videos=${count}"
  echo "bad=${bad}"
  echo "video_dir=${VIDEO_DIR}"
  first_video="$(find "${VIDEO_DIR}" -maxdepth 1 -type f -name '*.mp4' | sort | head -1)"
  if [[ -n "${first_video}" ]]; then
    ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of csv=p=0 "${first_video}"
  fi
} | tee "${OUT_ROOT}/settings.txt"

if [[ "${count}" -eq 0 ]]; then
  echo "No valid videos prepared" >&2
  exit 2
fi

cd "${ROOT}"

run_flash() {
  CUDA_DEVICE="${FLASH_GPU:-0}" \
  INPUT_DIR="${VIDEO_DIR}" \
  OUTPUT_DIR="${OUT_ROOT}/flashvsr_official" \
  MODEL_DIR="${FLASHVSR_MODEL_DIR:-/mnt/models/FlashVSR-v1.1}" \
  SCALE=4 \
  SEED=0 \
    bash "${ROOT}/wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh" \
    2>&1 | tee "${OUT_ROOT}/logs/flashvsr.log"
}

run_seedvr3() {
  CUDA_DEVICE="${SEEDVR_GPU:-2}" \
  MODEL_KIND=seedvr1 \
  MODEL_DIR="${SEEDVR3B_MODEL_DIR:-/mnt/models/SeedVR-3B}" \
  INPUT_DIR="${VIDEO_DIR}" \
  OUTPUT_DIR="${OUT_ROOT}/seedvr3b" \
  SEEDVR_PYTHON="${SEEDVR_PYTHON:-/mnt/conda_envs/seedvr/bin/python}" \
  RES_H=768 \
  RES_W=1280 \
  OUT_FPS=8 \
  SEED=0 \
  MASTER_PORT="${MASTER_PORT:-29732}" \
  LOG_FILE="${OUT_ROOT}/logs/seedvr3b.inner.log" \
    bash "${ROOT}/wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh" \
    2>&1 | tee "${OUT_ROOT}/logs/seedvr3b.log"
}

run_flash > "${OUT_ROOT}/logs/flash_all.log" 2>&1 &
flash_pid=$!
run_seedvr3 > "${OUT_ROOT}/logs/seedvr3_all.log" 2>&1 &
seed_pid=$!
wait "${flash_pid}" "${seed_pid}"

find "${OUT_ROOT}" -type f -name '*.mp4' | sort > "${OUT_ROOT}/all_mp4.txt"
conductor s3 sync "${VIDEO_DIR}" "s3://lxh/data/test/$(basename "${VIDEO_DIR}")"
conductor s3 sync "${OUT_ROOT}" "s3://lxh/data/test/$(basename "${OUT_ROOT}")"
echo "DONE_VIDEO_LQ_FIXED ${OUT_ROOT}"
