# FlashVSR 稳定实验登记表

这个文档只记录已经进入下一阶段、或可作为正式对照/初始化来源的稳定实验。  
它不是 worklog，不记录日常 debug 细节；每条记录必须能回答：用什么代码、什么配置、什么脚本、在哪台机器、实验目录在哪里、最终可用 checkpoint 是哪个。

## 2026-05-10：当前稳定主线

### Stage1 89f 稳定母本：`v5.3.5 nonstreaming aligned23`

用途：

- 当前 Stage2 `v6` 的初始化来源；
- 89 帧 Stage1 主线母本；
- 解决旧版 streaming projector 的 `22/23` latent-time 不对齐问题。

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300
```

主机与节点：

```text
主机：b8gkuie2ns
六节点 48GPU：
  b8gkuie2ns
  wfnwbym4v6
  kh5idf7f98
  hj65iqg9rh
  zhki5rrddw
  xwk6qjuej5
```

训练代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py
```

启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned23-ResumeStep3000-Seed20260501.sh
```

配置文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_resume_step3000_seed20260501.yaml
```

关键设置：

```text
dataset_mode: tar_v53
num_frames: 89
image_branch_num_frames: 5
height/width: 768/1280
batch_size: 1
learning_rate: 1e-5
dataset_num_workers: 1
yubari_video_prob: 0.5
takano_video_prob: 0.5
image_tar_url: /mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt
takano_video_tar_url: /mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt
degradation_config_path: params_aliyun_video_compression_v1.yaml
lq_proj_temporal_mode: nonstreaming_aligned
lq_proj_scale: 1.0
lq_proj_layer_num: 1
freeze_lq_proj_in: false
zero_init_lq_proj_in: false
lora_rank: 384
max_train_steps: 10000
validation_num_samples: 3
use_gradient_checkpointing: true
use_gradient_checkpointing_offload: false
```

初始化/恢复关系：

```text
resume_training_state_dir:
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_175800/output/training_state/step-3000

resume_reset_rng_with_global_seed: true
global_seed: 2026050101
```

稳定 checkpoint：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

validation：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/validation/step-10000
```

云端备份目标：

```text
s3://lxh/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/
```

备份状态：

```text
2026-05-10 已完成 conductor s3 sync，并已通过 conductor s3 ls 验证。
```

备份范围：

```text
run.log
launch_command.sh
snapshot/
output/step-10000.safetensors
output/validation/step-10000/
```

不备份：

```text
output/training_state/
DeepSpeed optimizer/rng/scheduler resume 状态
```

## 2026-05-10：Clean 版代码入口

这一节记录在稳定母本基础上整理出的 clean 入口。clean 版不替代上面已经完成训练和备份的稳定实验；用途是后续新实验优先从更干净的代码、配置、启动脚本出发，避免继续叠加历史 debug 分支。

### Stage1 clean：`v5.3.5 clean nonstreaming aligned23`

对应稳定母本：

```text
Stage1 89f 稳定母本：v5.3.5 nonstreaming aligned23
```

clean 训练代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_clean_lora.py
```

clean 启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Clean-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-NonStreamProj-Aligned23.sh
```

clean 配置文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23.yaml
```

smoke 脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-Smoke-2GPU-v5-3-5-Clean-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-NonStreamProj-Aligned23.sh
```

smoke 配置：

```text
wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23.yaml
```

clean 整理内容：

```text
保留原 Stage1 v5.3.5 的核心训练逻辑、数据设置、nonstreaming_aligned projector、LoRA 设置和 checkpoint 导入关系。
关闭/移除用于排查的 tensor dump、训练 debug dump、tensor preview、GC/branch 调试打印等旁路输出。
保留必要的参数摘要和模型加载日志，方便确认实验设置。
```

smoke 验证：

```text
机器：b8gkuie2ns
RUN_TS_OVERRIDE: 20260510_clean_stage1_smoke2
实验目录：
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23_20260510_clean_stage1_smoke2

结果：
step=1 loss=0.034188
step=2 loss=0.329382
```

备注：

```text
第一次 smoke 失败是因为 b8 上 8 张卡仍有占卡 Python 进程，每张约 166GB 显存；清空占卡进程后，同一套 clean smoke 配置成功跑通。
```

### Stage2 clean：`v6 clean block-sparse video-only`

对应稳定训练：

```text
Stage2 89f 稳定训练：v6 block-sparse video-only worker2
```

clean 训练代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_clean_lora.py
diffsynth/models/wan_video_dit_stage2_v6_clean.py
```

clean 启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Clean-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh
```

clean 配置文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml
```

smoke 脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Clean-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh
```

smoke 配置：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml
```

clean 整理内容：

```text
Stage2 clean 训练入口改为依赖 Stage1 clean 模块和 Stage2 clean DiT patch 文件。
保留 Stage2 v6 的核心逻辑：video-only、Stage1 step-10000 初始化 lq_proj_in 和 LoRA、block_sparse_chunk_causal、topk_ratio=2.0。
保留必要的 Stage2 加载确认日志：lq_proj_in keys、LoRA keys、attention mode、topk_ratio。
```

smoke 验证：

```text
机器：b8gkuie2ns
RUN_TS_OVERRIDE: 20260510_clean_stage2_smoke
实验目录：
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_smoke_2gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260510_clean_stage2_smoke

结果：
Stage2 v6 loaded lq_proj_in ... keys=8, missing=0, unexpected=0
Stage2 v6 loaded LoRA ... keys=480, missing=825, unexpected=0
Stage2 v6 attention mode: block_sparse_chunk_causal
Stage2 v6 topk_ratio: 2.0
step=1 loss=0.270946
step=2 loss=0.008904
```

### Stage2 89f 稳定训练：`v6 block-sparse video-only worker2`

用途：

- 当前 Stage2 89 帧主线；
- 从 Stage1 `v5.3.5 step-10000` 热启动 `lq_proj_in` 与 DiT LoRA；
- 用于探索 block-sparse causal attention 和后续 Stage3。

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2
```

主机与节点：

```text
主机：b8gkuie2ns
六节点 48GPU：
  b8gkuie2ns
  wfnwbym4v6
  kh5idf7f98
  hj65iqg9rh
  zhki5rrddw
  xwk6qjuej5
```

训练代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py
diffsynth/models/wan_video_dit_stage2_v6.py
```

启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh
```

配置文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml
```

关键设置：

```text
dataset_mode: stage2_video_only
num_frames: 89
height/width: 768/1280
batch_size: 1
learning_rate: 1e-5
dataset_num_workers: 2
yubari_video_prob: 0.5
takano_video_prob: 0.5
degradation_config_path: params_aliyun_video_compression_v1.yaml
stage2_attention_mode: block_sparse_chunk_causal
stage2_topk_ratio: 2.0
stage2_local_num: -1
zero_init_lq_proj_in: false
lora_rank: 384
max_train_steps: 10000
validation_num_samples: 0
use_gradient_checkpointing: true
use_gradient_checkpointing_offload: false
```

初始化来源：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

该 checkpoint 同时初始化：

```text
lq_proj_in
DiT LoRA
```

稳定 checkpoint：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output/step-10000.safetensors
```

validation：

```text
无。该训练配置 validation_num_samples: 0，因此训练目录内没有 validation 输出。
```

云端备份目标：

```text
s3://lxh/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/
```

备份状态：

```text
2026-05-10 已完成 conductor s3 sync，并已通过 conductor s3 ls 验证。
```

备份范围：

```text
run.log
launch_command.sh
snapshot/
output/step-10000.safetensors
```

不备份：

```text
output/training_state/
DeepSpeed optimizer/rng/scheduler resume 状态
```
