#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
INTERVAL_SECONDS="${WANDB_SYNC_INTERVAL_SECONDS:-3600}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
WANDB_BIN="${WANDB_BIN:-/mnt/conda_envs/flashvsr/bin/wandb}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/wandb_offline_sync.log}"
TRAIN_PROCESS_PATTERN="${TRAIN_PROCESS_PATTERN:-train_flashvsr_stage3_v7_c_lora.py}"
PERSIST_WANDB_DIR="${PERSIST_WANDB_DIR:-${RUN_DIR}/wandb}"
TRANSIENT_WANDB_DIR="${TRANSIENT_WANDB_DIR:-/mnt/task_runtime/lucidvsr/wandb}"
WANDB_SYNC_TIMEOUT_SECONDS="${WANDB_SYNC_TIMEOUT_SECONDS:-600}"

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${PERSIST_WANDB_DIR}"

is_training_alive() {
  pgrep -f "${TRAIN_PROCESS_PATTERN}" >/dev/null 2>&1
}

sync_once() {
  local synced_any=0
  if [ -d "${TRANSIENT_WANDB_DIR}" ]; then
    while IFS= read -r -d '' run_path; do
      local persist_path="${PERSIST_WANDB_DIR}/$(basename "${run_path}")"
      mkdir -p "${persist_path}"
      rsync -a --delete "${run_path}/" "${persist_path}/" 2>&1 | tee -a "${LOG_FILE}" || true
      echo "[$(date '+%F %T')] mirrored transient wandb run to ${persist_path}" | tee -a "${LOG_FILE}"
    done < <(find "${TRANSIENT_WANDB_DIR}" -maxdepth 1 -type d -name 'offline-run-*' -print0 2>/dev/null)
  fi

  while IFS= read -r -d '' run_path; do
    synced_any=1
    echo "[$(date '+%F %T')] syncing ${run_path}" | tee -a "${LOG_FILE}"
    timeout "${WANDB_SYNC_TIMEOUT_SECONDS}" "${WANDB_BIN}" sync "${run_path}" 2>&1 | tee -a "${LOG_FILE}" || true
  done < <(find "${PERSIST_WANDB_DIR}" -maxdepth 1 -type d -name 'offline-run-*' -print0 2>/dev/null)

  if [ "${synced_any}" -eq 0 ]; then
    echo "[$(date '+%F %T')] no offline-run directories found in: ${PERSIST_WANDB_DIR} or ${TRANSIENT_WANDB_DIR}" | tee -a "${LOG_FILE}"
  fi
}

echo "[$(date '+%F %T')] wandb offline sync loop start run_dir=${RUN_DIR} interval=${INTERVAL_SECONDS}s python=${PYTHON_BIN}" | tee -a "${LOG_FILE}"

while true; do
  sync_once
  if ! is_training_alive; then
    echo "[$(date '+%F %T')] training process not found; running final sync and exiting" | tee -a "${LOG_FILE}"
    sync_once
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
