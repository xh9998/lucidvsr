#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb.yaml}"
export OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_release_48gpu_v7_d_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb}"

bash "${SCRIPT_DIR}/FlashVSR-Stage3-Release-48GPU-v7-D-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-641Data.sh"
