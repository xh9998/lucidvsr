#!/bin/bash
set -euo pipefail

source /mnt/task_runtime/bolt_lxh/use_active_python.sh

/mnt/conda_envs/flashvsr/bin/python /mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/scripts/plot_flashvsr_loss_from_log.py \
  --log_path /mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_nostartval_20260412_042600/run.log \
  --output_dir /mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_nostartval_20260412_042600/loss_plot \
  --title "FlashVSR 16GPU 17f takano bs24 lr1e-5 alpha5"
