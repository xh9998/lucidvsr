#!/usr/bin/env bash
set -euo pipefail

cd /mnt/task_runtime/lucidvsr
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/conda_envs/flashvsr/bin/python}"
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export FLASHVSR_IO_MAX_PARALLEL="${FLASHVSR_IO_MAX_PARALLEL:-2}"
export FLASHVSR_IO_NODE_LIMIT_DIR="${FLASHVSR_IO_NODE_LIMIT_DIR:-/tmp/flashvsr_io_limiter}"
export CONDUCTOR_VERBOSITY="${CONDUCTOR_VERBOSITY:-1}"
export CONDUCTOR_METRICS_INTERVAL="${CONDUCTOR_METRICS_INTERVAL:-3600000}"
export CONDUCTOR_CACHE_MAX_BYTES="${CONDUCTOR_CACHE_MAX_BYTES:-214748364800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export FLASHVSR_STAGE3_DEBUG_LOSS="${FLASHVSR_STAGE3_DEBUG_LOSS:-1}"
export FLASHVSR_STAGE3C_NO_GATHER_LOG="${FLASHVSR_STAGE3C_NO_GATHER_LOG:-1}"
export FLASHVSR_STAGE3_TIMING_DEBUG="${FLASHVSR_STAGE3_TIMING_DEBUG:-1}"
export FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG="${FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG:-0}"
export FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG="${FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG:-0}"
export FLASHVSR_STAGE3_VAL_FROM_TRAIN_BATCH="${FLASHVSR_STAGE3_VAL_FROM_TRAIN_BATCH:-1}"
export FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH="${FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH:-1}"
export FLASHVSR_STAGE3_DS_CONFIG="${FLASHVSR_STAGE3_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json}"
export FLASHVSR_STAGE3_FAKE_DS_CONFIG="${FLASHVSR_STAGE3_FAKE_DS_CONFIG:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload_fake_stable.json}"
export FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED="${FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED:-2026052301}"
export TORCH_HOME="${TORCH_HOME:-/mnt/torch_cache}"
export NOTARY_CONFIG_FILE="${NOTARY_CONFIG_FILE:-/turibolt_k8s_mounts/task_configmap/notary-config}"
export AWS_CA_BUNDLE="${AWS_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"

: "${OF_ID:?OF_ID is required, e.g. OF-A-fast}"
: "${OF_KIND:?OF_KIND is required, e.g. full/recononly/flowonly/dmdfakeonly}"
: "${CONFIG_PATH:?CONFIG_PATH is required}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29541}"
NEED_STAGE3_CRITIC="${NEED_STAGE3_CRITIC:-0}"
START_OCCUPY_ON_EXIT="${START_OCCUPY_ON_EXIT:-1}"
START_EMPTY_GPU_GUARD="${START_EMPTY_GPU_GUARD:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
SAFE_OF_ID="${OF_ID//-/_}"
OUTPUT_TAG="${OUTPUT_TAG:-stage3_${SAFE_OF_ID}_4gpu_overfit4_${OF_KIND}_v7_d4_4}"
ACCEL_YAML="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_1node4gpu_nooffload.yaml"
TRAIN_PY="${TRAIN_PY:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_lora.py}"

STAGE2_EXP="train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100"
RESUME_STAGE2_CHECKPOINT="${RESUME_STAGE2_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"
RESUME_STAGE2_S3="${RESUME_STAGE2_S3:-s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/${STAGE2_EXP}/output/step-6000.safetensors}"

STAGE1_EXP="train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn"
STAGE3_REAL_CHECKPOINT="${STAGE3_REAL_CHECKPOINT:-/mnt/task_wrapper/user_output/artifacts/exp/${STAGE1_EXP}/output/step-3000.safetensors}"
STAGE3_REAL_S3="${STAGE3_REAL_S3:-s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors}"
STAGE3_FAKE_CHECKPOINT="${STAGE3_FAKE_CHECKPOINT:-${STAGE3_REAL_CHECKPOINT}}"
STAGE3_FAKE_S3="${STAGE3_FAKE_S3:-${STAGE3_REAL_S3}}"

RUN_TS="${RUN_TS_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${OUTPUT_TAG}_${RUN_TS}"
RUN_DIR="/mnt/task_wrapper/user_output/artifacts/exp/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${RUN_DIR}/snapshot"
if [ "${ENABLE_DMD_TENSOR_DUMP:-0}" = "1" ]; then
  export FLASHVSR_STAGE3_DMD_TENSOR_DUMP_DIR="${FLASHVSR_STAGE3_DMD_TENSOR_DUMP_DIR:-${RUN_DIR}/dmd_tensor_dumps}"
  export FLASHVSR_STAGE3_DMD_TENSOR_DUMP_STEPS="${FLASHVSR_STAGE3_DMD_TENSOR_DUMP_STEPS:-1,2,5,10,20,50,100,200,500,1000}"
  mkdir -p "${FLASHVSR_STAGE3_DMD_TENSOR_DUMP_DIR}"
fi
export WANDB_DIR="${WANDB_DIR:-${RUN_DIR}}"
mkdir -p "${WANDB_DIR}/wandb"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

finish() {
  rc=$?
  if [ "${START_OCCUPY_ON_EXIT}" = "1" ]; then
    echo "[guard] ${OF_ID} exited rc=${rc}; starting occupy on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh || true
  fi
  exit "${rc}"
}
trap finish EXIT

if [ "${START_EMPTY_GPU_GUARD}" = "1" ]; then
  GPU_EMPTY_GUARD_SESSION="gpu_empty_guard_${SAFE_OF_ID}"
  GPU_EMPTY_GUARD_LOG="${RUN_DIR}/gpu_empty_guard.log"
  tmux kill-session -t "${GPU_EMPTY_GUARD_SESSION}" 2>/dev/null || true
  tmux new-session -d -s "${GPU_EMPTY_GUARD_SESSION}" \
    "cd /mnt/task_runtime/lucidvsr && GPU_IDS='${GPU_IDS}' GPU_EMPTY_GUARD_LOG_PREFIX='[gpu-empty-guard ${OF_ID}]' bash /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/gpu_empty_guard_start_occupy.sh 2>&1 | tee -a '${GPU_EMPTY_GUARD_LOG}'"
  echo "[guard] started empty GPU guard session=${GPU_EMPTY_GUARD_SESSION} gpu_ids=${GPU_IDS} log=${GPU_EMPTY_GUARD_LOG}"
fi

WANDB_OFFLINE_S3_URI="${WANDB_OFFLINE_S3_URI:-s3://lxh/tmp/wandb_offline/${RUN_NAME}.tar.gz}"
tmux kill-session -t "wandb_package_${SAFE_OF_ID}" 2>/dev/null || true
tmux new-session -d -s "wandb_package_${SAFE_OF_ID}" \
  "cd /mnt/task_runtime/lucidvsr && RUN_DIR='${RUN_DIR}' TRAIN_PROCESS_PATTERN='train_flashvsr_stage3_v7_d4_4_overfit_lora.py' WANDB_OFFLINE_S3_URI='${WANDB_OFFLINE_S3_URI}' WANDB_PACKAGE_INTERVAL_SECONDS=3600 bash /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/package_wandb_offline_to_s3_loop.sh"

TAKANO_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
OVERFIT_MANIFEST="/mnt/task_wrapper/user_output/artifacts/data/manifests/stage3_overfit4_takano_fixed_20260523.txt"
if [ ! -s "${TAKANO_MANIFEST}" ]; then
  mkdir -p "$(dirname "${TAKANO_MANIFEST}")"
  conductor s3 cp "s3://lxh/data/mainfest/takano_video_train_all.txt" "${TAKANO_MANIFEST}"
fi
mkdir -p "$(dirname "${OVERFIT_MANIFEST}")"
awk 'NF && $1 !~ /^#/ { print; n += 1; if (n == 4) exit 0 }' "${TAKANO_MANIFEST}" > "${OVERFIT_MANIFEST}"
if [ "$(wc -l < "${OVERFIT_MANIFEST}")" -ne 4 ]; then
  echo "[error] overfit manifest should contain exactly 4 lines: ${OVERFIT_MANIFEST}" >&2
  exit 1
fi

VGG16_PATH="${TORCH_HOME}/hub/checkpoints/vgg16-397923af.pth"
if [ ! -s "${VGG16_PATH}" ]; then
  mkdir -p "$(dirname "${VGG16_PATH}")"
  conductor s3 cp "s3://lxh/models/SR/vgg16-397923af.pth" "${VGG16_PATH}"
fi
if [ ! -s "${RESUME_STAGE2_CHECKPOINT}" ]; then
  mkdir -p "$(dirname "${RESUME_STAGE2_CHECKPOINT}")"
  conductor s3 cp "${RESUME_STAGE2_S3}" "${RESUME_STAGE2_CHECKPOINT}"
fi
if [ "${NEED_STAGE3_CRITIC}" = "1" ]; then
  if [ ! -s "${STAGE3_REAL_CHECKPOINT}" ]; then
    mkdir -p "$(dirname "${STAGE3_REAL_CHECKPOINT}")"
    conductor s3 cp "${STAGE3_REAL_S3}" "${STAGE3_REAL_CHECKPOINT}"
  fi
  if [ ! -s "${STAGE3_FAKE_CHECKPOINT}" ]; then
    mkdir -p "$(dirname "${STAGE3_FAKE_CHECKPOINT}")"
    conductor s3 cp "${STAGE3_FAKE_S3}" "${STAGE3_FAKE_CHECKPOINT}"
  fi
fi

echo "of_id=${OF_ID}"
echo "of_kind=${OF_KIND}"
echo "gpu_ids=${GPU_IDS}"
echo "overfit_manifest=${OVERFIT_MANIFEST} lines=$(wc -l < "${OVERFIT_MANIFEST}")"
echo "fixed_sample_seed=${FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED}"
echo "resume_stage2_checkpoint=${RESUME_STAGE2_CHECKPOINT}"
echo "need_stage3_critic=${NEED_STAGE3_CRITIC}"
echo "stage3_real_checkpoint=${STAGE3_REAL_CHECKPOINT}"
echo "stage3_fake_checkpoint=${STAGE3_FAKE_CHECKPOINT}"
echo "config=${CONFIG_PATH}"
echo "dmd_tensor_dump_dir=${FLASHVSR_STAGE3_DMD_TENSOR_DUMP_DIR:-}"
echo "dmd_tensor_dump_steps=${FLASHVSR_STAGE3_DMD_TENSOR_DUMP_STEPS:-}"
echo "wandb_offline_package_tmux=wandb_package_${SAFE_OF_ID} run_dir=${RUN_DIR} s3_uri=${WANDB_OFFLINE_S3_URI}"

cp "${TRAIN_PY}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py" "${RUN_DIR}/snapshot/" || true
cp "${CONFIG_PATH}" "${RUN_DIR}/snapshot/" || true
cp "${OVERFIT_MANIFEST}" "${RUN_DIR}/snapshot/" || true
cp "/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/gpu_empty_guard_start_occupy.sh" "${RUN_DIR}/snapshot/" || true
cp "$0" "${RUN_DIR}/snapshot/" || true
cp "${ACCEL_YAML}" "${RUN_DIR}/snapshot/" || true

CMD=(/mnt/conda_envs/flashvsr/bin/accelerate launch
  --config_file "${ACCEL_YAML}"
  --num_machines 1
  --num_processes 4
  --main_process_port "${MASTER_PORT}"
  --deepspeed_multinode_launcher standard
  "${TRAIN_PY}"
  --config "${CONFIG_PATH}"
  --output_path "${RUN_DIR}/output"
  --wandb_name "${RUN_NAME}"
  --resume_stage2_checkpoint "${RESUME_STAGE2_CHECKPOINT}"
  --debug_dump_training_batch_dir "${RUN_DIR}/debug_batch"
  --debug_dump_training_batch_limit 1
  --debug_dump_training_batch_fps 8
  --zero_init_lq_proj_in false
)
if [ "${NEED_STAGE3_CRITIC}" = "1" ]; then
  CMD+=(--stage3_real_checkpoint "${STAGE3_REAL_CHECKPOINT}")
  CMD+=(--stage3_fake_checkpoint "${STAGE3_FAKE_CHECKPOINT}")
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
  # Space-separated simple CLI overrides, e.g.
  # EXTRA_ARGS='--max_train_steps 1000 --save_steps 100'.
  read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch_command.sh"; printf '\n' >> "${RUN_DIR}/launch_command.sh"
"${CMD[@]}"
