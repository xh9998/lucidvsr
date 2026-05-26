#!/usr/bin/env bash
set -euo pipefail

export CONFIG_PATH="${CONFIG_PATH:-/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml}"
export OUTPUT_TAG="${OUTPUT_TAG:-train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb}"
exec /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-Clean-OfflineWandb.sh
