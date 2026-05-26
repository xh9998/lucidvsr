#!/usr/bin/env bash
set -euo pipefail

# Thin launcher for isolated GateY hack-probe variants.
# It delegates environment/bootstrap to the existing OF 4GPU launcher but
# switches TRAIN_PY to the hack-probe wrapper.  Production D44 code is not
# modified by this script.

: "${HACK_PROBE_VARIANT:?Set HACK_PROBE_VARIANT=fake_x0_equal_real|dmd_grad_scale0p1_clipnear|color_match_fake_x0_to_real|fake_score_percentile_clip|weight_factor_rms_detach|freeze_fake_lq_proj|fake_lr0p1|fake_lr0p01}"

case "${HACK_PROBE_VARIANT}" in
  fake_x0_equal_real)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_fake_x0_equal_real_4gpu_dmdonly_dfake5.yaml"
    ;;
  dmd_grad_scale0p1_clipnear)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_dmd_grad_scale0p1_clipnear_4gpu_dmdonly_dfake5.yaml"
    ;;
  color_match_fake_x0_to_real)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_color_match_fake_x0_to_real_4gpu_dmdonly_dfake5.yaml"
    ;;
  fake_score_percentile_clip)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_fake_score_percentile_clip_4gpu_dmdonly_dfake5.yaml"
    ;;
  weight_factor_rms_detach)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_weight_factor_rms_detach_4gpu_dmdonly_dfake5.yaml"
    ;;
  freeze_fake_lq_proj)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_freeze_fake_lq_proj_4gpu_dmdonly_dfake5.yaml"
    ;;
  fake_lr0p1)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_fake_lr0p1_4gpu_dmdonly_dfake5.yaml"
    ;;
  fake_lr0p01)
    DEFAULT_CONFIG="/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_gateY_hack_probe_fake_lr0p01_4gpu_dmdonly_dfake5.yaml"
    ;;
  *)
    echo "[gateY_hack_probe] unknown HACK_PROBE_VARIANT=${HACK_PROBE_VARIANT}" >&2
    exit 2
    ;;
esac

export FLASHVSR_STAGE3_HACK_PROBE_VARIANT="${HACK_PROBE_VARIANT}"
export TRAIN_PY="${TRAIN_PY:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_hack_probe_lora.py}"
export CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG}}"
export OF_ID="${OF_ID:-gateY-${HACK_PROBE_VARIANT}}"
export OF_KIND="${OF_KIND:-hack_probe}"
export NEED_STAGE3_CRITIC="${NEED_STAGE3_CRITIC:-1}"
export ENABLE_DMD_TENSOR_DUMP="${ENABLE_DMD_TENSOR_DUMP:-1}"
export OUTPUT_TAG="${OUTPUT_TAG:-stage3_gateY_hack_probe_${HACK_PROBE_VARIANT}_4gpu_dmdonly_dfake5}"

echo "[gateY_hack_probe] variant=${HACK_PROBE_VARIANT}"
echo "[gateY_hack_probe] train_py=${TRAIN_PY}"
echo "[gateY_hack_probe] config=${CONFIG_PATH}"
echo "[gateY_hack_probe] output_tag=${OUTPUT_TAG}"

exec bash /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-OF-Fast-4GPU-v7-D4-4.sh
