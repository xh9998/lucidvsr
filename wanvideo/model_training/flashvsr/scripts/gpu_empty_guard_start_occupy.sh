#!/usr/bin/env bash
set -euo pipefail

# Monitor a GPU subset and start the normal occupy program for any GPU that stays
# truly empty. "Empty" is intentionally defined by both low memory and low util
# to avoid occupying a rank that is only waiting at a distributed barrier.

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
CHECK_INTERVAL_SECONDS="${GPU_EMPTY_GUARD_CHECK_INTERVAL_SECONDS:-30}"
IDLE_SECONDS="${GPU_EMPTY_GUARD_IDLE_SECONDS:-180}"
STARTUP_GRACE_SECONDS="${GPU_EMPTY_GUARD_STARTUP_GRACE_SECONDS:-600}"
MEM_THRESHOLD_MB="${GPU_EMPTY_GUARD_MEM_THRESHOLD_MB:-1024}"
UTIL_THRESHOLD="${GPU_EMPTY_GUARD_UTIL_THRESHOLD:-5}"
OCCUPY_SCRIPT="${GPU_EMPTY_GUARD_OCCUPY_SCRIPT:-/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh}"
LOG_PREFIX="${GPU_EMPTY_GUARD_LOG_PREFIX:-[gpu-empty-guard]}"
STATE_DIR="${GPU_EMPTY_GUARD_STATE_DIR:-/tmp/flashvsr_gpu_empty_guard}"
OCCUPY_SESSION_PREFIX="${GPU_EMPTY_GUARD_OCCUPY_SESSION_PREFIX:-gpu_guard_occupy}"

mkdir -p "${STATE_DIR}"

started_at="$(date +%s)"
IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

gpu_state_file() {
  local gpu="$1"
  printf '%s/gpu_%s.idle_since' "${STATE_DIR}" "${gpu//[^0-9A-Za-z_.-]/_}"
}

query_gpu() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1
}

start_occupy_for_gpus() {
  local gpus="$1"
  local safe_gpus="${gpus//,/ _}"
  safe_gpus="${safe_gpus// /}"
  local session="${OCCUPY_SESSION_PREFIX}_${safe_gpus}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "${LOG_PREFIX} occupy session already exists: ${session} gpus=${gpus}"
    return 0
  fi

  echo "${LOG_PREFIX} starting occupy session=${session} gpus=${gpus}"
  tmux new-session -d -s "${session}" \
    "cd /mnt/task_runtime/lucidvsr 2>/dev/null || cd /mnt/task_runtime; CUDA_VISIBLE_DEVICES='${gpus}' bash '${OCCUPY_SCRIPT}'"
}

echo "${LOG_PREFIX} start gpu_ids=${GPU_IDS} idle_seconds=${IDLE_SECONDS} grace=${STARTUP_GRACE_SECONDS} mem_threshold_mb=${MEM_THRESHOLD_MB} util_threshold=${UTIL_THRESHOLD}"

while true; do
  now="$(date +%s)"
  idle_gpus=()

  for raw_gpu in "${gpu_array[@]}"; do
    gpu="$(trim "${raw_gpu}")"
    [ -n "${gpu}" ] || continue

    line="$(query_gpu "${gpu}" || true)"
    if [ -z "${line}" ]; then
      echo "${LOG_PREFIX} warn unable to query gpu=${gpu}"
      rm -f "$(gpu_state_file "${gpu}")"
      continue
    fi

    util="$(trim "${line%%,*}")"
    mem="$(trim "${line#*,}")"
    util="${util:-999}"
    mem="${mem:-999999}"
    state_file="$(gpu_state_file "${gpu}")"

    if [ "${mem}" -le "${MEM_THRESHOLD_MB}" ] && [ "${util}" -le "${UTIL_THRESHOLD}" ]; then
      if [ ! -s "${state_file}" ]; then
        printf '%s\n' "${now}" > "${state_file}"
      fi
      idle_since="$(cat "${state_file}" 2>/dev/null || printf '%s' "${now}")"
      idle_for=$((now - idle_since))
      since_start=$((now - started_at))
      echo "${LOG_PREFIX} gpu=${gpu} empty util=${util} mem=${mem}MB idle_for=${idle_for}s since_start=${since_start}s"
      if [ "${since_start}" -ge "${STARTUP_GRACE_SECONDS}" ] && [ "${idle_for}" -ge "${IDLE_SECONDS}" ]; then
        idle_gpus+=("${gpu}")
      fi
    else
      rm -f "${state_file}"
      echo "${LOG_PREFIX} gpu=${gpu} busy util=${util} mem=${mem}MB"
    fi
  done

  if [ "${#idle_gpus[@]}" -gt 0 ]; then
    joined="$(IFS=','; echo "${idle_gpus[*]}")"
    start_occupy_for_gpus "${joined}"
    for gpu in "${idle_gpus[@]}"; do
      rm -f "$(gpu_state_file "${gpu}")"
    done
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
