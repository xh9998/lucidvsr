# FlashVSR 组会进展（2026-04-27）

## 1. 本周主线结论

本周主要完成三件事：

1. 将 `v5.3` 系列实验整理成稳定对照线，明确每个版本的变量。
2. 完成多机迁移、阶段一到阶段二的恢复训练，并重新跑标准测试集。
3. 开始设计 Stage 2 / `v6`，把结构改进、causal、streaming 三件事拆开，避免直接把推理逻辑硬塞进训练。

当前判断：

- `v5.3 / v5.3.1 / v5.3.2` 是当前主要对照实验。
- `v5.3.3` 随机 projector 对照已停止，原因是效果相对 FlashVSR projector 初始化不明显。
- `v5.3.4` 已启动，主要为后续 89 帧和二阶段训练准备。
- Stage 2 不能简单理解成“加 causal mask”，它至少包含 self-attention 结构替换、causal mask、streaming / KV cache 三层问题。

## 2. v5.3 系列实验矩阵

| 版本 | 主要目的 | 数据构成 | 退化 | projector 初始化 | 图像 branch | 当前状态 | 结论 |
|---|---|---|---|---|---|---|---|
| `v5.3` | 主实验线 | Takano video + Yubari video + image | Aliyun full | FlashVSR projector init，先 freeze 后 unfreeze | 外部 image 构造 pseudo-video | phase2 已启动 | 用作标准主线，观察 full degradation 的最终效果 |
| `v5.3.1` | 退化强度对照 | 与 `v5.3` 相同 | Aliyun half | 同 `v5.3` | 同 `v5.3` | phase2 已启动 | 对照 full/half degradation，判断退化是否过强 |
| `v5.3.2` | 图像来源对照 | Yubari video + Yubari frame image | Aliyun full | 同 `v5.3` | 从 Yubari 视频中抽一帧构造 pseudo-video | phase2 已启动 | 排除外部 image 数据读取和分布差异的影响 |
| `v5.3.3` | projector 初始化对照 | 与 `v5.3` 对齐 | Aliyun full | random projector init | 同 `v5.3` | 已停止 | 随机初始化早期学习明显更难，效果优势不明显 |
| `v5.3.4` | 89 帧 / 二阶段准备 | 与 `v5.3` 对齐 | Aliyun full | random projector init | 固定 5 帧 pseudo-video | 已启动 | 为长帧数和后续 Stage 2 训练验证路径 |

### 2.1 三条主要实验线

`v5.3 / v5.3.1 / v5.3.2` 的共同点是：

- 都采用 author-style paired sample；
- 一个 batch item 内包含一条 video branch 和一条 image pseudo-video branch；
- 先 freeze `LQ_proj_in`，只训练 LoRA；
- 后续从满意 checkpoint 进入 phase2，打开 `LQ_proj_in` 和 LoRA 一起训练。

这样做的原因是：直接随机训练 projector 时，模型前期需要先学会使用 LQ 条件支路，早期训练效率较低。FlashVSR 官方 projector 初始化能让模型一开始就看到更稳定的 LQ 表征。

### 2.2 关键实验目录

| 实验 | 阶段一目录 | 阶段二目录 / 当前测试 ckpt |
|---|---|---|
| `v5.3` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427_20260427_001300/output/step-600.safetensors` |
| `v5.3.1` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427_20260427_001300/output/step-600.safetensors` |
| `v5.3.2` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426_20260426_142230/output/step-1200.safetensors` |
| `v5.3.3` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_20260427_130300` | 已停止 |
| `v5.3.4` | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_20260428_040800` | 89 帧实验已修正 image branch 帧数后重启，已出 loss |

## 3. 测试设置

本周重新整理了固定测试流程，后续 checkpoint 对比都按照这个格式放结果。

| 测试集 | 内容 | 输入尺寸 | 帧数 / FPS | 生成方式 | 目录 |
|---|---|---|---|---|---|
| synthetic | 3 个 Takano + 3 个 Yubari | LQ `320x192`，GT `1280x768` | 17 帧 / 8fps | Aliyun 退化，去掉最后 bicubic 上采样，保留 1/4 LQ | `/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq_20260427` |
| real | challenging 真实输入 | `320x192` | 17 帧 / 8fps | 本机真实视频按比例 resize 后 center crop | `/mnt/task_wrapper/user_output/artifacts/inference/challenging_test_lxh_17f_320x192` |

对比方法：

| 方法 | 权重 / 模型 | 说明 |
|---|---|---|
| `v5.3 phase2` | `step-600.safetensors` | full degradation 主线 |
| `v5.3.1 phase2` | `step-600.safetensors` | half degradation 对照 |
| `v5.3.2 phase2` | `step-1200.safetensors` | Yubari frame-image 对照 |
| FlashVSR official | `/mnt/models/FlashVSR-v1.1` | 官方模型 |
| SeedVR3B | `/mnt/models/SeedVR-3B` | 常用外部 baseline |

输出目录：

`/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset`

目录结构：

```text
synthetic/<method>/*.mp4
real/<method>/*.mp4
logs/*.log
```

本周还修复了推理效率问题：LoRA 分支从“每个视频重新加载 Wan/VAE/projector”改成“每个 checkpoint / dataset 加载一次”，后续测试会明显更快。

## 4. 换卡和恢复训练

本周原来的母机过期后，实验迁移到新的多机环境。这里最重要的经验是：16 卡 DeepSpeed resume 不能只搬主机目录。

| 问题 | 现象 | 修正 |
|---|---|---|
| 只恢复主机 artifacts | resume 卡在 barrier / TCPStore / 状态加载附近 | 主机和从机的训练状态都要恢复 |
| 从机状态缺失 | 看起来像 NCCL 通信问题 | 从旧从机对应的 `s3://bolt-prod-.../tasks/<task_id>/artifacts/...` 同步到新从机 |
| freeze 到 unfreeze 阶段切换 | 需要使用阶段一满意 checkpoint 作为阶段二初始化 | 阶段二不再从 FlashVSR projector 单独初始化，而是从阶段一 ckpt 导入 projector + LoRA |

这部分属于工程流程修正，后续会作为固定 resume 流程执行。

### 4.1 GPU 利用率和 DataLoader worker

训练过程中出现过 GPU 利用率周期性 `100% -> 0%` 的现象，主要瓶颈不在模型 forward，而在数据读取、解码和退化。原始设置 `dataset_num_workers=0` 时，取数据与 GPU 训练串行，容易让 GPU 等数据。

本周尝试打开 DataLoader worker 后，定位到两个问题：

| 问题 | 现象 | 处理 |
|---|---|---|
| worker 退化默认落到 `cuda:0` | 0 卡显存异常升高，其他卡缓慢增长 | 退化模块改成 lazy init，并在 worker 内按 `LOCAL_RANK` 绑定 CUDA device |
| `worker=2` 会额外占显存 | `v5.3.1` 在 `bs12 + worker=2` 下 backward 阶段 OOM | 当前判断不是显存泄露，而是每个 rank 多出 worker CUDA/退化上下文后余量不足 |

当前结论：

- `worker=2` 已经证明能跑通 4 卡 smoke，并能改善取数据串行问题。
- 但在 16 卡正式训练里，`bs12` 本身已经接近显存上限，`worker=2` 不是所有实验都稳。
- 如果需要严格 DeepSpeed optimizer/state resume，最稳妥是不改 `bs`，优先把 worker 降到 `1` 或 `0`。
- 如果要保留 `worker=2` 并降低 `bs`，更适合从已有 safetensors 导入 LoRA + projector warm-start，重新开一个新实验，而不是 optimizer 原地 resume。

## 5. Stage 2 / v6 设计

Stage 2 的目标不是继续改数据，而是改 Wan DiT 的 self-attention。当前理解可以拆成三层。

### 5.1 结构改进

目标：

```text
Wan DiT 30 层 self-attention:
dense full attention -> block-sparse attention
```

原则上不动：

- VAE；
- patch embedding；
- cross-attention；
- FFN；
- output head。

需要改的是 self-attention 本身。论文里的 block size 是 `(2, 8, 8)`，也就是在 latent time、height、width 三个维度上做 block。

### 5.2 causal

causal 的目标是让当前 latent-time 只能看当前和过去，不能看未来。

这里不能直接用普通 Transformer 的 `is_causal=True`，因为视频 token flatten 后，同一帧内有很多空间 token。普通 causal 会让同一帧内后面的空间 token 不能看前面的空间 token，破坏同一帧空间建模。

正确逻辑应该是：

- 同一个 latent-time 内，空间 token 互相可见；
- 当前 latent-time 可以看过去 latent-time；
- 当前 latent-time 不能看未来 latent-time。

### 5.3 streaming

streaming 是另一层问题，不等价于 causal。

streaming 相关内容包括：

- `LR Proj-In` 按 4 帧一组流式处理；
- DiT 推理时可能是首段 `f=6`，后续 `f=2`；
- self-attention 需要 KV cache；
- 需要处理 overlap / buffer；
- 训练 loss 是否只算新输出部分还不确定。

也就是说：

| 概念 | 作用 | 是否必须同时做 |
|---|---|---|
| block-sparse | 降低 self-attention 计算量 | Stage 2 核心 |
| causal mask | 防止看未来帧 | Stage 2 核心 |
| streaming / KV cache | 支持长视频流式推理 | 论文 Stage 2/3 边界还需确认 |

### 5.4 v6 当前路线

当前 `v6` 先按分层方式推进：

1. video-only；
2. causal LR Proj-In；
3. time-aware causal attention；
4. dense fallback 先对齐 correctness；
5. 再接 FlashVSR 官方 block-sparse kernel；
6. 最后再决定是否在训练中引入 streaming chunk + KV cache。

已新增初始代码：

| 文件 | 作用 |
|---|---|
| `diffsynth/models/wan_video_dit_stage2_v6.py` | Stage 2 attention patch / dense causal baseline / block-sparse 接口 |
| `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py` | v6 video-only 训练入口 |
| `wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_densecausal.yaml` | 2 卡 smoke config |
| `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-DenseCausal.sh` | 2 卡 smoke 启动脚本 |

### 5.5 需要确认的问题

当前最大的未决问题是：论文 Stage 2 训练到底是不是也按照官方推理那样走 streaming chunk。

需要向作者确认：

| 问题 | 为什么重要 |
|---|---|
| Stage 2 训练是否使用 `f=6,2,2,...` 的 streaming chunk？ | 决定训练 forward 是整段 causal 还是 chunked causal |
| KV cache 是 Stage 2 训练就使用，还是主要用于 Stage 3 / inference？ | 决定训练代码复杂度和 loss 对齐方式 |
| 89 帧对应 23 个 latent-time，而 block temporal size 是 2，如果整段训练如何处理？ | 决定是否需要 padding、裁剪，或必须走 streaming chunk |
| 如果训练也走 streaming，overlap / buffer 的 loss 怎么算？ | 决定 flow matching loss 是否只算新 chunk |

已整理问题文档：

`/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/flashvsr_stage2_questions_for_authors_20260427.md`

## 6. 下一步

1. 继续看 `v5.3 / v5.3.1 / v5.3.2` phase2 测试结果。
2. 观察 `v5.3.4` 89 帧 random projector 实验，为后续长帧数 / Stage 2 做准备。
3. 向作者确认 Stage 2 训练是否使用 streaming chunk + KV cache。
4. 确认后推进 `v6` 的 block-sparse causal attention 训练实现。
