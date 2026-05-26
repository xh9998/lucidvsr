# FLASHVSR STANDARD INDEX

更新时间：2026-05-18

用途：这是当前 FlashVSR 三阶段标准版本索引。它不是日常 worklog，而是给后续复现/接手时一眼确认“应该用哪条线、文件在哪里、checkpoint 在哪里、有没有 S3 备份”的总表。

已有旧索引：

- `doc/flashvsr_stable_experiment_registry.md`
- 旧索引主要记录 2026-05-10 前后的 Stage1/Stage2 稳定实验；本文补齐当前 Stage1 USMGT、Stage2 641、Stage3 D4.2/D4.4。

## 当前标准结论

- Stage1 稳定母本：`v5.3.5 nonstreaming_aligned23 step-10000`
- Stage1 当前给 Stage3 使用的 sharper/USMGT teacher：`v5.3.5 USMGT Takano20250205 step-3000`
- Stage2 当前标准 pretrain：`v6.4.1 step-6000`
- Stage3 当前复现/备用标准：`v7-D4.2 single-runner dfake5`
- Stage3 当前正在跑/更正规候选：`v7-D4.4 dual Accelerator + dual DeepSpeedPlugin dfake5`
- 不再新增 `D3.3`：如果要“front22 + dfake5”，直接用 D4.2。

## Stage1 标准母本：v5.3.5 step-10000

用途：

- Stage1 89f 稳定母本。
- Stage2 v6/v6.4/v6.4.1 的初始化来源。
- Stage3 D3/D3.2 旧 authorweights teacher 来源。
- Stage1 USMGT/sharpness 实验的 warm-start 来源。

训练代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

配置：

- `wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_resume_step3000_seed20260501.yaml`

启动脚本：

- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned23-ResumeStep3000-Seed20260501.sh`

实验目录：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300`

机器记录：

- 主机：`b8gkuie2ns`
- 6 节点 48GPU：`b8gkuie2ns`, `wfnwbym4v6`, `kh5idf7f98`, `hj65iqg9rh`, `zhki5rrddw`, `xwk6qjuej5`

关键 checkpoint：

- 远端路径：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- S3 备份：
  `s3://lxh/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- 备份核对：
  `2026-05-10 16:25:06`, `1141980688` bytes。

validation：

- 远端：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/validation/step-10000`
- S3：
  `s3://lxh/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/validation/step-10000/`
- S3 已看到：`sample_000/`, `sample_001/`, `sample_002/`

关键设置：

- `num_frames: 89`
- `lq_proj_temporal_mode: nonstreaming_aligned`
- `89f -> 23 latent positions`
- `lora_rank: 384`
- `learning_rate: 1e-5`
- `height/width: 768/1280`
- `dataset_mode: tar_v53`
- `yubari_video_prob: 0.5`
- `takano_video_prob: 0.5`

## Stage1 当前 Stage3 Teacher：v5.3.5 USMGT step-3000

用途：

- 当前 D4.2/D4.4 Stage3 的默认 `G_real/G_fake` 初始化。
- 用户称的 sharper/sharpness 实验，对 GT 做 USM sharpen 后训练。
- D4.2 以后不要再默认用旧 `step-10000` authorweights teacher，除非明确做旧版对照。

训练代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`

专用 dataset：

- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53_usmgt.py`

配置：

- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml`

启动脚本：

- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh`

实验目录：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`

机器记录：

- 16GPU 两节点：`bfs6vaz4d6`, `i6hf4scd4y`
- 最新只读核对：`bfs6vaz4d6` 上该目录存在，约 `123G`。

关键 checkpoint：

- 远端路径：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- S3 fallback 1：
  `s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors`
- S3 fallback 2：
  `s3://lxh/tmp/usmgt_stage1_takano20250205_step3000_20260517/step-3000.safetensors`
- S3 核对：
  - `2026-05-18 00:49:53`, `1141980688` bytes
  - `2026-05-17 13:20:50`, `1141980688` bytes

当前远端已看到的 checkpoint：

- `step-10, 20, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000`

关键设置：

- warm-start：Stage1 v5.3.5 `step-10000`
- `learning_rate: 5e-6`
- `gt_sharpen: true`
- `gt_sharpen_backend: torch`
- `gt_sharpen_device: auto`
- `degradation_device: auto`
- `dataset_num_workers: 2`
- `dataloader_multiprocessing_context: spawn`
- `lq_proj_temporal_mode: nonstreaming_aligned`

评测输出：

- step100：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step100_20260516`
  - S3：`s3://lxh/artifacts/inference/stage1_usmgt_takano20250205_step100_20260516`
  - 本机：`/Users/lixiaohui/Desktop/stage1_usmgt_takano20250205_step100_20260516`
- step1500/2500/3000：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
  - S3：`s3://lxh/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
  - 本机：`/Users/lixiaohui/Desktop/stage1_usmgt_takano20250205_step1500_2500_20260517`
- S3 已核对该目录存在：`logs/`, `synthetic_89f/`, `manifest_files.txt`

## Stage2 标准：v6.4.1 step-6000

用途：

- 当前 Stage3 student 初始化标准 checkpoint。
- 后续 Stage3 D4.2/D4.4 都默认加载它。
- 用户之前确认 `641` 效果较好，Stage3 先沿用 `641` 规则。

训练代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
- `diffsynth/models/wan_video_dit_stage2_v6_1.py`

配置：

- `wanvideo/model_training/flashvsr/configs/history/stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml`

启动脚本：

- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-1-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`

实验目录：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100`

机器记录：

- 5 节点 40GPU。
- 启动脚本使用 `accelerate_zero2_flashvsr_5node40gpu_nooffload.template.yaml`。
- 当前 t5 只读核对该目录存在，约 `1.1G`，可见 `output/step-6000.safetensors`。

初始化来源：

- Stage1 v5.3.5 `step-10000`

当前标准 checkpoint：

- 远端路径：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- S3 备份 1：
  `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- S3 备份 2：
  `s3://lxh/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- S3 核对：
  - bolt-prod：`2026-05-13 12:28:37`, `1141980688` bytes
  - lxh：`2026-05-14 01:25:03`, `1141980688` bytes

关键设置：

- `num_frames: 89`
- `dataset_mode: stage2_video_only`
- `stage2_attention_mode: block_sparse_chunk_causal`
- `stage2_topk_ratio: 2.0`
- `stage2_local_num: -1`
- `target: first_85_frames`
- 官方对齐 chunk-grouped top-k 版本。
- 训练稳定速度记录：约 `14.2-15.0s/step`。

评测输出：

- step200 扫描：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_scan89_v61_step200_20260512`
  - S3：`s3://lxh/tmp/stage2_v6_4_1_scan89_v61_step200_20260512`
  - 本机：`/Users/lixiaohui/Desktop/stage2_v6_4_1_scan89_v61_step200_20260512`
- official85f from 1800：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_official85f_from1800_20260512`
  - S3：`s3://lxh/tmp/stage2_v6_4_1_official85f_from1800_20260512`
  - 本机：`/Users/lixiaohui/Desktop/stage2_v6_4_1_official85f_from1800_20260512`

## Stage3 标准复现线：v7-D4.2 single-runner dfake5

用途：

- 目前建议的“论文语义复现/备用训练”标准线。
- 不再新开 D3.3；front22 + dfake5 直接用 D4.2。
- D4.2 比 D4.1 更稳，避免 turn-isolated/DDP fake 路径暴露的 rank skew 和大 collective 问题。

训练代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`

配置：

- release：
  `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_singlerunner_dfake5_offlinewandb.yaml`
- smoke：
  `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5.yaml`

启动脚本：

- release：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-SingleRunner-Dfake5-OfflineWandb.sh`
- smoke：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-2-SingleRunner-Dfake5.sh`

加载的 Stage2 pretrain：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- S3 fallback：
  `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`

加载的 Stage1 real/fake teacher：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- S3 fallback：
  `s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors`

语义：

- `stage3_dfake_gen_update_ratio: 5`
- runner 0：student/generator + fake
- runner 1-4：fake-only
- runner 5：student/generator + fake
- fake loss 使用当前 `student_z_pred.detach()`，不回传 student。
- student backward 只包含 `student_loss + dmd_student_loss`。
- fake backward 只包含 fake FM loss，显式 fake grad averaging，只 step fake optimizer。
- teacher 对齐目标：保留 teacher 前 22 `[0,22)`，丢弃最后一个 `[22,23)`。

注意：

- D4.2 release 脚本里有一行 echo 仍残留 `teacher_lq_trim_front_to_match` 字样；这是旧日志文案，不代表当前代码语义。代码和验证文档已经按前 22 的 `trim_tail_to_match` 修正。

smoke 结果：

- 机器：`6ikhpjzv3z`
- run：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5_20260518_d42_smoke_g01c_takano`
- 只读核对：该目录在 `6ikhpjzv3z` 存在，约 `35G`。
- 已有：
  - `run.log`
  - `output/resolved_args.yaml`
  - `output/step-1.safetensors`
  - `output/step-2.safetensors`
- 结论：完整通过到 runner 5 第二个 generator turn。

正式 48 卡：

- 截至本文，未发现 D4.2 正式 48 卡 run 作为当前主训练；D4.4 才是当前 48 卡正式 run。

## Stage3 当前 48 卡候选：v7-D4.4 dual Accelerator

用途：

- 当前正在跑的 48 卡 Stage3 候选。
- 更接近“两个模型/两个 optimizer/两个 DeepSpeedPlugin”的组织。
- 目标是比 D4.2 更正规，但速度和 fake backward/sync 仍需持续观察。

训练代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`

配置：

- release：
  `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- smoke：
  `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml`

启动脚本：

- release：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- smoke：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`

加载的 Stage2 pretrain：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`

加载的 Stage1 real/fake teacher：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`

正式 run：

- run name：
  `train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
- run dir：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
- 机器：
  6 节点 48GPU：`t5qdtykjsw`, `a9suya6gxe`, `67dxkwcb7m`, `ui9n6p293s`, `g48bd6x4h7`, `gx2intv5rk`
- master：
  `t5qdtykjsw`
- master address/port：
  `240.12.138.137:29547`

当前只读核对：

- 在 `t5qdtykjsw` 上 run dir 存在，约 `102G`。
- `run.log` 最新 stat：`2026-05-17 22:06:16 -0700`, size `756564`。
- 已有 checkpoint：
  - `output/step-1.safetensors`
  - `output/step-2.safetensors`
  - `output/step-5.safetensors`
  - `output/step-10.safetensors`
  - `output/step-20.safetensors`
  - `output/step-50.safetensors`
- 最近日志已到：
  - `runner_step=359`
  - `step=72`
  - `dfake_gen_update_ratio=5`
  - `generator_update=1` 出现在 `runner_step=355`, `step=72`

W&B/offline 同步：

- offline package S3：
  `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1.tar.gz`
- S3 核对：
  `2026-05-18 12:51:19`, size `825971`
- 本地/网页同步记录：
  W&B run URL 曾记录为 `https://wandb.ai/veralee/flashvsr/runs/yid6lzvt`

smoke/验证：

- 2GPU timing smoke 机器：`bfs6vaz4d6`
- smoke run：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_validate_bfs_2gpu`
- 只读核对：该目录在 `bfs6vaz4d6` 存在，约 `28G`。
- 已有：
  - `run.log`
  - `output/resolved_args.yaml`
  - `output/step-1.safetensors`

语义：

- 双 Accelerator + 双 DeepSpeedPlugin。
- G_fake 每个 runner step 更新。
- student/generator 每 `stage3_dfake_gen_update_ratio=5` 个 runner step 更新一次。
- fake loss 使用当前 `z_pred.detach()`。
- teacher 前 22 对齐：`trim_tail_to_match`。
- fake 只训练 `lq_proj_in` 和 LoRA，不训练完整 WAN body。
- dense_full fake/teacher 已确认走 flash-attn。

已知边界：

- D4.4 更正规，但未证明结果一定优于 D4.2。
- fake backward/sync 仍是主要耗时。
- 当前 generator step 仍远小于 D3.2 的 `1800+`，不能直接用 loss 曲线做同阶段对比。

## 旧 Stage3 对照：D3.2

用途：

- 旧结果对照。
- `step-2000` 视觉比 `step-1500` 好很多，值得保留评测结果。
- 但 D3.2 teacher 对齐是旧的后 22，不是当前标准前 22。

训练入口：

- wrapper：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-2-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-OfflineWandb.sh`
- 实际 train：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_1_lora.py`
- config：
  `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml`

run dir：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`

评测结果：

- step-1500 本机：
  `/Users/lixiaohui/Desktop/stage3_v7_d3_2_step1500_synthetic_20260518`
- step-2000 本机：
  `/Users/lixiaohui/Desktop/stage3/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01/step-2000`
- step-2000 S3：
  `s3://lxh/data/test/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01/step-2000/`
- S3 核对：`10/10` mp4 已存在。

注意：

- D3.2 可以作为旧视觉对照，不作为当前标准复现线。
- 如果目标是“front22 + dfake5”，不要派生 D3.3，直接用 D4.2。

## 数据与公共资源

Takano video manifest：

- 远端：`/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt`
- S3：`s3://lxh/data/mainfest/takano_video_train_all.txt`

Takano 20250205 4K video manifest：

- 远端：`/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_20250205_test_4k_tar_manifest.txt`
- S3：`s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt`
- repo 生成文件：
  `wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt`

Takano image manifest：

- 远端：`/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt`
- S3：`s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt`

Yubari video source：

- `conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/`

VGG16 LPIPS cache：

- 远端：`/mnt/torch_cache/hub/checkpoints/vgg16-397923af.pth`
- S3：`s3://lxh/models/SR/vgg16-397923af.pth`

## 复现时优先使用的文件组合

Stage1 old mother run:

- code: `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
- sh: `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned23-ResumeStep3000-Seed20260501.sh`
- config: `wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_resume_step3000_seed20260501.yaml`
- ckpt: Stage1 v5.3.5 `step-10000`

Stage1 current teacher:

- code: `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`
- dataset: `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53_usmgt.py`
- sh: `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh`
- config: `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml`
- ckpt: Stage1 USMGT `step-3000`

Stage2 current pretrain:

- code: `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
- model patch: `diffsynth/models/wan_video_dit_stage2_v6_1.py`
- sh: `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-1-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`
- config: `wanvideo/model_training/flashvsr/configs/history/stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml`
- ckpt: Stage2 v6.4.1 `step-6000`

Stage3 D4.2:

- code: `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
- sh: `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-SingleRunner-Dfake5-OfflineWandb.sh`
- config: `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_singlerunner_dfake5_offlinewandb.yaml`
- Stage2 ckpt: v6.4.1 `step-6000`
- Stage1 real/fake ckpt: USMGT `step-3000`

Stage3 D4.4:

- code: `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- sh: `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- config: `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- Stage2 ckpt: v6.4.1 `step-6000`
- Stage1 real/fake ckpt: USMGT `step-3000`

## 未完全关闭/需注意

- D4.2 release 脚本 echo 文案有旧 `trim_front` 字样；代码语义以 `train_flashvsr_stage3_v7_d4_2_lora.py` 和验证文档为准。
- D4.4 正式 run 的 W&B 是 offline package + 本机 sync，不是训练进程直接在线 W&B。
- Stage1 teacher deterministic full forward 数值等价不是本文重新验证的内容。
- D4.4 当前还在早期 generator step，不能直接和 D3.2 `step-2000` 的视觉成熟度比较。
- 严禁使用模糊 `pkill`/空变量 kill 停实验；停实验必须进 tmux Ctrl-C 或先精确确认 PID。
