#!/usr/bin/env bash
set -euo pipefail

export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:${PATH}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

RUN_DIR="${RUN_DIR:?RUN_DIR must be set}"
S3_URI="${WANDB_OFFLINE_S3_URI:?WANDB_OFFLINE_S3_URI must be set}"
INTERVAL_SECONDS="${WANDB_PACKAGE_INTERVAL_SECONDS:-3600}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/wandb_offline_package.log}"
TRAIN_PROCESS_PATTERN="${TRAIN_PROCESS_PATTERN:-train_flashvsr_stage3_v7_d_stable.py}"
PERSIST_WANDB_DIR="${PERSIST_WANDB_DIR:-${RUN_DIR}/wandb}"
TRANSIENT_WANDB_DIR="${TRANSIENT_WANDB_DIR:-/mnt/task_runtime/lucidvsr/wandb}"

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${PERSIST_WANDB_DIR}"

is_training_alive() {
  pgrep -f "${TRAIN_PROCESS_PATTERN}" >/dev/null 2>&1
}

mirror_transient_runs() {
  if [ ! -d "${TRANSIENT_WANDB_DIR}" ]; then
    return 0
  fi
  while IFS= read -r -d '' run_path; do
    local persist_path="${PERSIST_WANDB_DIR}/$(basename "${run_path}")"
    mkdir -p "${persist_path}"
    rsync -a --delete "${run_path}/" "${persist_path}/" 2>&1 | tee -a "${LOG_FILE}" || true
    echo "[$(date '+%F %T')] mirrored transient wandb run to ${persist_path}" | tee -a "${LOG_FILE}"
  done < <(find "${TRANSIENT_WANDB_DIR}" -maxdepth 1 -type d -name 'offline-run-*' -print0 2>/dev/null)
}

package_once() {
  mirror_transient_runs

  local run_count
  run_count="$(find "${PERSIST_WANDB_DIR}" -maxdepth 1 -type d -name 'offline-run-*' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${run_count}" = "0" ]; then
    echo "[$(date '+%F %T')] no offline-run directories found in ${PERSIST_WANDB_DIR}" | tee -a "${LOG_FILE}"
    return 0
  fi

  local tmp_tar
  tmp_tar="/tmp/$(basename "${RUN_DIR}")_wandb_offline.tar.gz"
  echo "[$(date '+%F %T')] packaging ${run_count} offline runs to ${tmp_tar}" | tee -a "${LOG_FILE}"
  (cd "${RUN_DIR}" && tar -czf "${tmp_tar}" wandb wandb_offline_package.log 2>/tmp/wandb_package_tar.err) 2>&1 | tee -a "${LOG_FILE}" || {
    echo "[$(date '+%F %T')] tar failed; stderr follows" | tee -a "${LOG_FILE}"
    cat /tmp/wandb_package_tar.err 2>/dev/null | tee -a "${LOG_FILE}" || true
    return 0
  }

  echo "[$(date '+%F %T')] uploading ${tmp_tar} to ${S3_URI}" | tee -a "${LOG_FILE}"
  conductor s3 cp "${tmp_tar}" "${S3_URI}" 2>&1 | tee -a "${LOG_FILE}" || true
}

echo "[$(date '+%F %T')] wandb offline package loop start run_dir=${RUN_DIR} s3_uri=${S3_URI} interval=${INTERVAL_SECONDS}s" | tee -a "${LOG_FILE}"

while true; do
  package_once
  if ! is_training_alive; then
    echo "[$(date '+%F %T')] training process not found; running final package and exiting" | tee -a "${LOG_FILE}"
    package_once
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
