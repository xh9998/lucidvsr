# FlashVSR Stage3 Overfit Debug Plan 2026-05-23

## 背景

D4.4 40 卡实验在早期 checkpoint 视觉效果较好，但 100 student step 之后出现细节消失 / 变糊。由于日志里的 loss 并没有同步表现为明显崩溃，当前不能只靠总 loss 判断训练正确。

这次先做小样本过拟合，目标是把问题从“大数据训练波动”缩小到“代码和训练目标是否能把固定样本学住”。

## 当前主要怀疑点

- 多 loss 的标量值健康，不代表它们对 LoRA / projector 的梯度方向健康。Flow、DMD、MSE、LPIPS 的 mean 维度不同，梯度贡献可能和 loss 数值不一致。
- DMD / G_fake 更新逻辑可能在某些阶段把 student 拉向过平滑方向，尤其是 fake critic 还没稳定时。
- `scheduler.step(..., to_final=True)` 的 one-step latent `z_pred` 如果尺度或方向不对，pixel / LPIPS 会给错误监督。
- Wan decoder 的 full-prefix detach decode 虽然语义上更接近作者，但仍可能存在首帧、窗口或 GT 对齐问题。
- 训练时 Stage2 sparse causal mask 与推理时 streaming KV-cache 需要保持一致；如果 student 在训练中没有学到稳定流式行为，外部推理会快速暴露。
- 在线退化和随机裁剪会掩盖问题，所以 overfit 需要固定 source、固定裁剪和固定退化 seed。

## 本轮 Overfit-16 设计

实验名：

`train_stage3_release_16gpu_v7_d4_4_overfit16_full_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb`

代码：

`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_lora.py`

配置：

`wanvideo/model_training/flashvsr/configs/history/stage3_release_16gpu_v7_d4_4_overfit16_full_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`

启动脚本：

`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-16GPU-v7-D4-4-Overfit16-Full-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`

关键设定：

- 使用 16 卡，每个 global rank 固定 1 个 Takano 视频，共 16 个视频。
- manifest 由 `takano_video_train_all.txt` 前 16 条生成到：
  `/mnt/task_wrapper/user_output/artifacts/data/manifests/stage3_overfit16_takano_fixed_20260523.txt`
- `max_source_frames=90`，让 89f clip 的 start 固定为 1，避免随机裁剪扰动。
- `FLASHVSR_OVERFIT_FIXED_SAMPLE_SEED=2026052301`，通过独立 wrapper 固定在线退化 sample seed，不改生产训练源码。
- 使用 D4.4 full loss：
  - `stage3_flow_weight=1.0`
  - `stage3_mse_weight=1.0`
  - `stage3_lpips_weight=2.0`
  - `stage3_dmd_weight=1.0`
  - `stage3_fake_fm_weight=1.0`
  - `stage3_dfake_gen_update_ratio=5`
- 保存 `1,2,5,10,20,50,100`。
- v3 起打开内部 validation：`validation_num_samples=1`、`validation_num_inference_steps=1`，用于快速追踪固定样本视觉变化。注意当前训练内 validation 仍是 Stage3 one-step direct decode，不是完整外部 streaming KV-cache 推理；真正判定流式视觉效果仍以后续外部 one-step streaming inference 为准。
- W&B 使用 offline 模式，写在实验目录 `wandb/` 下，避免 artifacts 之外的记录丢失。

## 实验矩阵与停机标准

### A. Full D4.4 Overfit-16

目的：验证最接近 40 卡 D4.4 主训练的完整代码能否在固定小数据上记住细节。

设置：

- `flow=1, mse=1, lpips=2, dmd=1, fake_fm=1`
- `dfake_gen_update_ratio=5`
- 固定 16 个 Takano 视频、固定 89f crop、固定在线退化 seed
- 保存并验证 `step 1,2,5,10,20,50,100`

停机标准：

- `step10/20` 已明显变糊或不学习：停止，进入 B/C/D 拆分实验。
- `step10/20` 变清晰：继续到 `step50/100`，检查是否复现“早期好、后期糊”。
- `step100` 仍能稳定记住：主代码基本能 overfit，问题更可能在大数据动态、loss 权重、dfake schedule 或训练时长。

### B. Recon-only Overfit

目的：单独验证 `z_pred -> Wan decoder -> pixel/LPIPS -> student` 反传链路。

设置：

- 保留 `mse + lpips`
- 关闭 `flow/dmd/fake`
- 先跑到 `step20`，必要时延长到 `step50`

判断：

- B 能记住但 A 不能：DMD/fake/flow 分支在拉坏 student。
- B 也不能记住：优先查 `scheduler.step(..., to_final=True)`、decoder prefix detach、GT/window 对齐。

### C. Flow-only Overfit

目的：检查原始 FM/flow 目标是否会保持二阶段行为，还是把模型推向过平滑。

设置：

- 只保留 `flow`
- 跑 `step20/50`

### D. DMD/Fake 分支对照

目的：判断 fake critic 和 DMD student loss 是否方向正确。

设置：

- 先跑 `DMD+fake` 小步
- 再跑 `DMD+recon`
- 默认先看 `step20`

## 判断标准

- 如果固定 16 个样本的外部推理从 `step-1` 到 `step-20/50/100` 明显变清晰，说明训练链路至少能 overfit，问题更可能是大数据目标、权重比例或训练时长动态。
- 如果固定样本也快速变糊，优先排查 `z_pred`、MSE/LPIPS decode 对齐、DMD 梯度方向和 fake critic 训练。
- 如果 recon 指标下降但视觉变差，说明 loss 的平均方式或 LPIPS/DMD/flow 的组合目标与视觉目标不一致。
- 如果训练 loss 下降但固定样本输出不变，说明 trainable 参数、optimizer ownership、LoRA/projector 更新路径存在问题。

## 下一步分支

- Overfit-16 full D4.4 失败：拆成 recon-only、DMD-only、flow-only、fake-only 四个小实验。
- Overfit-16 full D4.4 成功但大数据失败：重点做 loss weight / dfake schedule / 数据退化强度的 ablation。
- 如果早期好、后期糊：重点比较保存点的 `flow/mse/lpips/dmd_student/fake_loss/dmd_grad/dmd_skip` 与外部视觉，找出哪个 loss 分支开始反向主导。

## OF-fast 4GPU 并行验证矩阵

2026-05-23 追加：16 卡 Full Overfit-16 虽然最接近 40 卡正式训练，但普通 runner step 仍然是数分钟级，不能作为快速定位工具。因此新增一批 4 卡并行实验，目标是在同一批 16 卡上同时跑 4 个互补假设。

通用设置：

- 每个实验单机 4 卡，两个节点各跑两个实验。
- 每个实验固定 4 个 Takano 视频：
  `/mnt/task_wrapper/user_output/artifacts/data/manifests/stage3_overfit4_takano_fixed_20260523.txt`
- 固定 `89f / 768x1280 / bs=1 / lr=1e-5 / LoRA rank=384`。
- 保存并 validation：`step 1,2,5,10,20`。
- validation 不再启动前额外扫 dataset。overfit 实验设置
  `FLASHVSR_STAGE3_VAL_FROM_TRAIN_BATCH=1`，rank0 从第一批训练 batch 固定取样做 validation，确保“测训练集本身”，也避免启动阶段卡在 dataset scan。
- 数据读取不再每步重复扫同 4 个视频。overfit 实验设置
  `FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH=1`，每个 rank 首次取到本地训练 batch 后缓存到内存，后续 runner step 直接复用同一 batch。这个逻辑只在
  `train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py` 中存在，通用 D4.4 训练代码不受影响。
- W&B offline 写入各自实验目录，并后台每小时打包到 S3。
- 每个实验退出后，只在自己的 `CUDA_VISIBLE_DEVICES` 范围内启动占卡，避免空卡。

代码与启动：

- 通用启动脚本：
  `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-OF-Fast-4GPU-v7-D4-4.sh`
- overfit 入口：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_lora.py`
- overfit 专用训练实现：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py`
  只在这里加入 train-batch validation；通用 `train_flashvsr_stage3_v7_d4_4_lora.py` 保持不动，避免影响正在跑的 D4.4 正式实验。
- 4GPU accelerate 配置：
  `wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_1node4gpu_nooffload.yaml`

实验编号：

| 编号 | 名称 | 目的 | Loss 设置 | 是否加载 G_real/G_fake |
| --- | --- | --- | --- | --- |
| OF-A-fast | D4.4-4v-Full | 最小数据上复现完整 D4.4 目标，观察是否仍出现早期好后期糊 | `flow=1, mse=1, lpips=2, dmd=1, fake_fm=1` | 是 |
| OF-B-fast | D4.4-4v-ReconOnly | 验证 `z_pred -> Wan decoder -> MSE/LPIPS` 反传链路是否能单独记住样本 | `flow=0, mse=1, lpips=2, dmd=0, fake_fm=0` | 否 |
| OF-C-fast | D4.4-4v-FlowOnly | 验证原始 flow/FM 目标是否会保留二阶段行为或快速抹细节 | `flow=1, mse=0, lpips=0, dmd=0, fake_fm=0` | 否 |
| OF-D-fast | D4.4-4v-DMDFakeOnly | 验证 DMD + fake critic 分支自身是否方向正常，是否单独导致模糊 | `flow=0, mse=0, lpips=0, dmd=1, fake_fm=1` | 是 |

判断方式：

- OF-B-fast 能快速记住但 OF-A-fast 变糊：DMD/fake 或 flow 分支在组合后拉坏 student。
- OF-C-fast 单独就变糊：flow/FM 目标本身可能在 one-step stage3 中过强或不适合继续优化。
- OF-D-fast 单独异常：优先检查 DMD 梯度方向、fake critic 训练频率、fake/student shared timestep 和 `dmd_loss_max`。
- OF-A-fast 正常而 40 卡 D4.4 异常：问题更可能来自大数据分布、训练时长、dfake schedule 或在线退化动态，而不是基本代码路径。

### OF-fast v5 启动状态

2026-05-23 追加：v4 仍然每个 runner step 重新走 DataLoader，`data` 经常到几十到几百秒，导致过拟合验证本身太慢。v5 已改成真正的 per-rank first-batch cache：

- 代码只改 overfit 专用实现：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py`
- 标准 D4.4 训练源码保持不动：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- 启动脚本默认设置：
  `FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH=1`
- 实验目录：
  - OF-A-fast: `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_A_fast_4gpu_overfit4_full_v7_d4_4_20260523_of_a_fast_v5`
  - OF-B-fast: `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_B_fast_4gpu_overfit4_recononly_v7_d4_4_20260523_of_b_fast_v5`
  - OF-C-fast: `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_C_fast_4gpu_overfit4_flowonly_v7_d4_4_20260523_of_c_fast_v5`
  - OF-D-fast: `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_D_fast_4gpu_overfit4_dmdfakeonly_v7_d4_4_20260523_of_d_fast_v5`

启动后检查结果：

- 四组均已打印 `Caching the first DataLoader batch per rank`，并进入训练。
- 首个 runner step 仍需读取/准备首 batch，`data` 约 `175s / 269s / 515s / 213s`，其中 OF-C 首次最慢。
- 缓存后 `data` 已稳定降到约 `0.10-0.15s`，证明慢点主要来自重复数据读取/退化/视频准备，不是核心 forward/backward。
- 缓存后的核心耗时大致为：
  - recon-only：`student` 约 `8-15s`；
  - flow-only：`student` 约 `7s`；
  - DMD/fake：`student` 约 `7s`，`dmd` 约 `17.6s`，`fake + fake_backward_sync` 约 `22s`；
  - full：generator turn 加上 `dmd/fake/student_backward/save_sched` 仍较重，但不再被 data stall 主导。
- GPU 利用率在稳定 runner step 里基本恢复到 `97-100%`，rank0 或保存 validation 阶段会短暂偏低。

## 2026-05-25 OF-medium-long 重做计划

旧 OF-fast 只跑到 `20` student steps，且使用完整 `params_aliyun_video_compression_v1.yaml` 双阶段退化。用户复查后指出两个问题：

- 退化偏重，训练输入本身作为测试集也难以判断视觉改善；
- 20 step 太短，只能验证代码是否跑通，不能验证“早期清晰、后期变糊、再震荡或 NaN”的 Stage3 动态。

本轮重做目标：

- 固定小样本，真正看 overfit 曲线；
- 退化改为中度单阶段配置：
  `wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_medium_x4test.yaml`；
- 训练拉长到 `220` student steps；
- 保存与验证：
  `1,2,5,10,20,50,100,150,200,220`；
- 四组统一使用同一组数据随机性：
  `degradation_seed=2026052500`、`global_seed=2026052500`。
  旧 OF-fast 使用不同 seed，导致四组 LQ 退化强度不同，视觉对比不干净；本轮必须保证同一批固定视频的 LQ 输入一致，再比较不同 loss 分支。
- validation 仍直接使用训练 batch：
  `FLASHVSR_STAGE3_VAL_FROM_TRAIN_BATCH=1`；
- DataLoader 仍缓存首 batch：
  `FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH=1`；
- 优先用 4 卡 4 视频，而不是 16 卡 16 视频。原因是 overfit 验证要快速、明确地看固定样本是否能记住；16 卡会强制至少 16 个样本，通信更重、样本更多，反而更慢、更不利于定位。

新配置：

| 编号 | 配置 | Loss |
| --- | --- | --- |
| OF-A-medium-long | `stage3_of_a_medium_long_4gpu_overfit4_full_v7_d4_4.yaml` | `flow=1, mse=1, lpips=2, dmd=1, fake_fm=1` |
| OF-B-medium-long | `stage3_of_b_medium_long_4gpu_overfit4_recononly_v7_d4_4.yaml` | `flow=0, mse=1, lpips=2, dmd=0, fake_fm=0` |
| OF-C-medium-long | `stage3_of_c_medium_long_4gpu_overfit4_flowonly_v7_d4_4.yaml` | `flow=1, mse=0, lpips=0, dmd=0, fake_fm=0` |
| OF-D-medium-long | `stage3_of_d_medium_long_4gpu_overfit4_dmdfakeonly_v7_d4_4.yaml` | `flow=0, mse=0, lpips=0, dmd=1, fake_fm=1` |
| OF-E-fixedlqgt | `stage3_of_e_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4.yaml` | `flow=1, mse=1, lpips=2, dmd=0, fake_fm=0` |

监控重点：

- OF-B 是否能稳定把 4 个固定训练样本记住。若不能，先查 `z_pred`、Wan decoder prefix detach、GT/window 对齐，不再优先讨论 DMD；
- OF-C 是否单独造成模糊。若 flow-only 越训越糊，说明 one-step stage3 里 flow 权重/目标可能在拉向平滑；
- OF-D 是否出现 DMD spike、fake_loss 震荡或 NaN。若 D 单独异常，优先查 fake critic 和 DMD guard；
- OF-A 是否复现完整 D4.4 的早期好、后期糊。如果 B 正常但 A 异常，问题更集中在 flow/DMD/fake 的组合与 schedule。
- OF-E 是“完整主链路但去掉 DMD/fake”的对照；它决定后续是否把主要精力集中在 DMD/fake，而不是 pixel/flow 主链。
