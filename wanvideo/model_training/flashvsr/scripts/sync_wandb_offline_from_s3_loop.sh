#!/usr/bin/env bash
set -euo pipefail

export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:${PATH}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

S3_URI="${WANDB_OFFLINE_S3_URI:?WANDB_OFFLINE_S3_URI must be set}"
INTERVAL_SECONDS="${WANDB_SYNC_INTERVAL_SECONDS:-3600}"
WORKDIR="${WANDB_SYNC_WORKDIR:-/tmp/wandb_offline_sync_from_s3}"
WANDB_BIN="${WANDB_BIN:-/mnt/conda_envs/flashvsr/bin/wandb}"
CONDUCTOR_BIN="${CONDUCTOR_BIN:-conductor}"
WANDB_SYNC_TIMEOUT_SECONDS="${WANDB_SYNC_TIMEOUT_SECONDS:-600}"
LOG_FILE="${LOG_FILE:-${WORKDIR}/wandb_sync_from_s3.log}"
RUN_ONCE="${RUN_ONCE:-0}"

mkdir -p "${WORKDIR}"
touch "${LOG_FILE}"

sync_once() {
  local package_path="${WORKDIR}/wandb_offline_package.tar.gz"
  local extract_dir="${WORKDIR}/extract"

  echo "[$(date '+%F %T')] downloading ${S3_URI}" | tee -a "${LOG_FILE}"
  if ! "${CONDUCTOR_BIN}" s3 cp "${S3_URI}" "${package_path}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[$(date '+%F %T')] download failed; will retry later" | tee -a "${LOG_FILE}"
    return 0
  fi

  rm -rf "${extract_dir}"
  mkdir -p "${extract_dir}"
  if ! tar -xzf "${package_path}" -C "${extract_dir}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[$(date '+%F %T')] extract failed; will retry later" | tee -a "${LOG_FILE}"
    return 0
  fi

  local synced_any=0
  while IFS= read -r -d '' run_path; do
    synced_any=1
    echo "[$(date '+%F %T')] wandb sync ${run_path}" | tee -a "${LOG_FILE}"
    timeout "${WANDB_SYNC_TIMEOUT_SECONDS}" "${WANDB_BIN}" sync "${run_path}" 2>&1 | tee -a "${LOG_FILE}" || true
  done < <(find "${extract_dir}" -type d -name 'offline-run-*' -print0 2>/dev/null)

  if [ "${synced_any}" -eq 0 ]; then
    echo "[$(date '+%F %T')] no offline-run directories found in package" | tee -a "${LOG_FILE}"
  fi
}

echo "[$(date '+%F %T')] wandb s3 sync loop start s3_uri=${S3_URI} interval=${INTERVAL_SECONDS}s workdir=${WORKDIR}" | tee -a "${LOG_FILE}"

while true; do
  sync_once
  if [ "${RUN_ONCE}" = "1" ]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
