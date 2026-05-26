# FlashVSR / LucidVSR 实验版本总览

日期：2026-05-18

这份文档回答一个问题：从过去到现在，每条主要实验线做了什么、为什么做、结论是什么、哪些版本还值得继续用。

它不是完整流水账。完整逐日操作记录看：

```text
doc/FLASHVSR_WORKLOG.md
```

稳定母本的代码 / config / sh / checkpoint / 机器目录看：

```text
doc/flashvsr_stable_experiment_registry.md
```

## 1. 当前主线一句话

当前项目已经从 Stage1 full-attention SR teacher，推进到 Stage2 sparse-causal streaming teacher，并进入 Stage3 DMD one-step distillation。

主线关系：

```text
Stage1 v5.3.5 / 535
  -> 训练 89f full-attention restoration teacher
  -> 提供 Stage2 初始化

Stage2 v6.4.1 / 641
  -> 从 Stage1 继续训 89f sparse-causal streaming model
  -> 提供 Stage3 student 初始化

Stage3 v7-D4.4
  -> 从 Stage2 继续训 one-step streaming model
  -> G_real / G_fake 使用 Stage1 teacher
  -> 加 DMD + fake FM + pixel / LPIPS
```

## 2. 当前最重要的稳定版本

| 阶段 | 当前关键版本 | 作用 | 当前结论 |
|---|---|---|---|
| Stage1 89f | `v5.3.5 / 535` | 89 帧 full-attention teacher，Stage2 初始化 | 稳定母本，已用于 Stage2 |
| Stage1 17f | `v5.3.6 / 536` | 17 帧 teacher，对照和短测试 | 可用，但不是当前 89f 主线 |
| Stage1 fine-tune | `USMGT Takano 20250205 step-3000` | 用更清晰 GT 微调 Stage1，供 Stage3 的 G_real/G_fake 使用 | 当前 Stage3 D4.2+ 默认指定这个 Stage1 teacher |
| Stage2 89f | `v6.4.1 / 641 step-6000` | sparse-causal streaming teacher，Stage3 student 初始化 | 当前 Stage2 主线，效果比早期 v6 更对齐 |
| Stage3 | `v7-D4.4` | dual Accelerator + dual DeepSpeedPlugin 的 DMD 训练线 | 48 卡已跑起来，仍在验证 fake backward 性能和 loss 归属 |

## 3. Stage1 之前：v2 / v3 / v4 探索

### v2 / v3：早期视频 LoRA 和训练系统稳定性

定位：

- 更早期的纯视频 LoRA / full finetune 探索；
- 主要目标是把 Wan/FlashVSR 风格训练链路跑起来；
- 后面没有作为当前正式主线继续展开。

结论：

- 它们是后续 v4/v5 的工程基础；
- 当前不建议从 v2/v3 继续开新实验。

### v4：多源数据 + image/video joint 的桥梁版本

目标：

- 把 Takano / Yubari / image 数据接进同一训练线；
- 尝试 image/video packed attention；
- 定位 image 混训后 loss 异常抬高的问题。

代表文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_fullb.py
diffsynth/models/wan_video_dit_joint_v1.py
```

做过的 probe：

| Probe | 设置 | 目的 |
|---|---|---|
| A | Takano only, packed off | 确认纯视频是否稳定 |
| B | Takano + Yubari, packed off | 多视频源是否稳定 |
| C | Takano + image, packed off | image 是否导致 loss 异常 |
| D | 三源一起, packed off | 多源完整组合 |
| E/F/G | packed attention 系列 | 验证 image/video mask 与 packed 组织 |

结论：

- 纯视频相对稳定；
- image 一加入，loss 更容易异常抬高；
- packed / mask 思路可行，但 v4 的 image/video joint 语义不够干净；
- v4 不再作为正式主线，后续进入 v5。

详细文档：

```text
doc/flashvsr_v4_iteration.md
doc/flashvsr_v4_loss_and_data_investigation.md
doc/FLASHVSR_V4_REFBIG_GAPS.md
```

## 4. Stage1 v5：正式 teacher 训练线

v5 的核心目标：把 image/video joint training 从 v4 的混乱探索，拆成更明确的版本。

### v5.1 / v5.2：grouped image sample 试验

定位：

- grouped image sample；
- 尝试用不同 image 权重理解处理 image pseudo-video。

结论：

- 工程表达复杂；
- grouped image 带来 padding / token 对齐和显存压力；
- 没有成为主线。

### v5.3：author-style paired sample

定位：

- 一个 sample 明确包含：
  - 一个真实 video branch；
  - 一个 image pseudo-video branch；
- 样本语义最清楚，是后续 Stage1 主线基础。

关键文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py
wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py
diffsynth/models/wan_video_dit_joint_v5.py
```

结论：

- v5.3 成为 Stage1 正式主线；
- 后续 v5.3.5 / v5.3.6 都基于这条线继续发展。

### v5.3.5 / 535：89f 稳定 Stage1 teacher

目标：

- 训练 89 帧 full-attention restoration teacher；
- 使用 `nonstreaming_aligned` projector；
- 不把 streaming / causal 逻辑提前放进 Stage1；
- 作为 Stage2 89f 初始化来源。

关键规则：

```text
89 raw frames -> 23 latent-time
LQ projector 和 GT / DiT latent-time 完全对齐
```

核心设置：

| 项 | 设置 |
|---|---|
| 帧数 | 89f |
| 规模 | 48GPU |
| 数据 | Takano + Yubari + image pseudo-video |
| 退化 | aliyun degradation |
| projector | random init, `nonstreaming_aligned` |
| 学习率 | `1e-5` |
| 训练步数 | 到 `step-10000` |

稳定实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300
```

稳定 checkpoint：

```text
.../output/step-10000.safetensors
```

结论：

- 当前 89f Stage1 稳定母本；
- Stage2 `v6 / v6.4.1` 都从它初始化；
- 后续若要找 89f Stage1 teacher，优先找这个。

详细登记：

```text
doc/flashvsr_stable_experiment_registry.md
```

### v5.3.6 / 536：17f Stage1 teacher

目标：

- 训练 17 帧版本；
- 用于短视频 teacher、短测试和对照。

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200
```

结论：

- 17f 线可用；
- 当前大部分 Stage2 / Stage3 89f 主线不再以它为核心。

### Stage1 clean 版

目标：

- 保留 v5.3.5 核心逻辑；
- 去掉历史 debug flag、tensor dump、冗余打印；
- 给后续新实验一个更干净入口。

代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_clean_lora.py
```

结论：

- 2GPU smoke 成功；
- 原稳定母本不被覆盖，clean 版只是后续开新线的推荐入口。

## 5. Stage2 v6：从 full attention 到 streaming sparse-causal

Stage2 目标：把 Stage1 full-attention teacher 继续适配成适合长视频流式推理的 sparse-causal DiT。

核心变化：

| 模块 | Stage1 | Stage2 |
|---|---|---|
| 数据 | video + image joint | video only |
| LQ projector | nonstreaming aligned | streaming causal |
| DiT attention | full attention | chunk causal + block sparse |
| 推理 | 整段 / full attention | streaming + KV cache |

### v6：第一版 Stage2 sparse-causal

目标：

- 从 `v5.3.5 step-10000` 初始化；
- video-only；
- block sparse chunk causal；
- 先跑通 48 卡 Stage2。

结论：

- 训练跑通；
- 推理出现 chunk-level temporal artifact / 跳变；
- 需要进一步区分是 KV cache、top-k、mask、colorfix、学习率还是 target 对齐导致。

### v6.1：接近官方 streaming + KV cache 推理

目标：

- 尽量贴近 FlashVSR 官方流式推理；
- chunk 输入，前面 chunk 的 K/V 缓存给后续 chunk 用；
- 用于检查推理路径是否对。

结论：

- 仍有 chunk 边界跳变；
- 说明问题不是单纯“没用 cache”。

### v6.2：full sequence + mask，无 KV cache

目标：

- 让推理更接近训练 mask；
- 整段送入 DiT，用 causal mask 表达流式关系，不使用 KV cache。

结论：

- 仍有跳变；
- 说明问题不是 KV cache 独立导致。

### A-H probe：定位 Stage2 跳变

固定测试：

- 89f synthetic；
- 重点看 `takano04`；
- 对比 v6.1、v6.2、dense full、FlashVSR official。

关键结论：

| Probe | 设置 | 结果 |
|---|---|---|
| A | v6.2, topk=2 | 跳变 |
| B | v6.2, topk=4 | 跳变 |
| C | dense full attention | 不跳 |
| D | v6.2, topk=8 | 跳变 |
| F | colorfix / no-colorfix | 主结论不变 |
| G | projector stats dump | projector 数值稳定 |
| H | 官方式 local mask + chunk grouped top-k | 仍跳 |
| FlashVSR official | 官方 baseline | 也有轻微边界，但小很多 |

总判断：

- sparse / chunk causal 路径是跳变核心条件；
- 不是简单 top-k 太小；
- 不是 colorfix 单独导致；
- 不是 projector 数值爆炸；
- dense full attention 不跳。

详细文档：

```text
doc/flashvsr_stage2_v6_jump_probe_20260509.md
```

### v6.3：小学习率对照

目标：

- 把 Stage2 学习率从 `1e-5` 降到 `3e-6`；
- 验证跳变是否由学习率过大导致。

结论：

- 仍有跳变；
- 学习率不是主因。

### v6.4：首帧 / target 对齐修正

旧 v6 风险：

```text
LQ projector: 89 -> 22
GT target:    89 -> VAE 23 -> drop z0 -> 22
```

风险是 drop 掉首 latent 后，原 `z1` 被放到首位，但 Wan decoder / DiT 首 latent 有特殊语义。

v6.4 新假设：

```text
LQ projector: 89 -> 22
GT target:    first 85 frames -> VAE -> 22
```

目的：

- 保留首 latent 的语义；
- 修复偶发首帧模糊 / 首帧不稳定。

结论：

- 方向更合理；
- 后续 v6.4.1 在此基础上继续修 sparse top-k。

### v6.4.1 / 641：当前 Stage2 主线

目标：

- 保留 v6.4 的首帧 target 对齐；
- 修正 sparse top-k 组织方式，让它更接近作者 chunk-grouped 口径。

核心修正：

```text
先应用 chunk causal / temporal allowed mask
再在每个 query chunk 内选 top-k block pair
```

而不是在更全局的 block pair 上选 top-k。

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100
```

关键 checkpoint：

```text
.../output/step-6000.safetensors
```

结论：

- 当前 Stage2 最重要版本；
- Stage3 student 初始化默认用它；
- 仍需要通过 Stage3 / loss 约束继续压 chunk artifact。

## 6. 固定测试集和 baseline

固定测试集：

| 数据集 | 数量 | 帧数 | 用途 |
|---|---:|---:|---|
| synthetic | 10 | 89f / 17f | 合成轻退化，Takano/Yubari |
| real | 11 | 89f / 17f | 真实 challenging videos |

常用本地目录：

```text
/Users/lixiaohui/Desktop/baselines_flash_seedvr3b_17f89f_20260506_by_dataset
```

已经做过的 baseline：

| 方法 | 说明 |
|---|---|
| FlashVSR official | 官方 baseline |
| SeedVR3B | 生成型强 baseline |
| SeedVR2-3B | 2026-05-18 补测到同一目录 |
| Stage1 / Stage2 / Stage3 ckpt scan | 用同一批 synthetic / real 看随训练变化 |

重要结论：

- SeedVR 类方法细节强，但慢、偏生成；
- FlashVSR / LucidVSR 路线更快，更强调保真和少过度生成；
- Stage2/Stage3 的质量判断必须看视频连续性，不只看单帧。

## 7. Stage3 v7：DMD one-step distillation

Stage3 目标：把 Stage2 sparse-causal model 蒸馏成 one-step streaming VSR model。

论文 loss：

```text
L = DMD + fake FM + pixel MSE + lambda * LPIPS
lambda = 2
```

角色：

| 模型 | 来源 | 是否训练 | 作用 |
|---|---|---|---|
| Student / G_one | Stage2 v6.4.1 | 训练 LoRA + projector | one-step 输出 |
| G_real | Stage1 teacher | 冻结 | real distribution reference |
| G_fake | Stage1 teacher copy | 训练 | fake distribution model |

### v7-A：one-step + decoder + pixel / LPIPS 链路

目标：

- 先验证 Stage2 checkpoint 能否一步得到 `z_pred`；
- Wan decoder 解码后能否算 pixel MSE / LPIPS；
- 不先接完整 DMD。

结论：

- 基础链路跑通；
- 但不符合论文 random latent decode / previous detach 的完整设定；
- 不能作为最终 Stage3。

### v7-B：random latent decode 和显存语义

目标：

- 对齐论文“随机选 2 个 latents decode，previous ones detached”；
- 解决 decoder / LPIPS 显存问题；
- 检查 CPU/GPU 退化、输入 LQ/GT 是否正常。

结论：

- 确认 Stage3 显存主要卡在 Wan decoder selected-window backward；
- CPU 退化方向可行，但不能影响训练输入质量；
- v7-B 是完整 DMD 前的语义收口，不是最终版本。

### v7-C：接入完整 DMD runner

目标：

- 接入 `G_real / G_fake`；
- 支持 fake model 独立 optimizer；
- 明确 student loss 和 fake loss 的参数归属。

走过的问题：

- `G_fake` 和 student 同时放入当前 runner 会撞到单 optimizer / DeepSpeed 组织限制；
- 需要拆双 optimizer / 双训练目标；
- `v7-C6` 48 卡可启动并出 loss，但权重、validation、spike guard 和 fake 更新语义仍不够最终。

结论：

- v7-C 证明完整 DMD 工程骨架可跑；
- 但还不是最终作者对齐版本。

### v7-D / D3.2：作者权重 + datafix 后的 48 卡线

目标：

- 修正 Stage3 的权重设置；
- 加 spike guard；
- 用更接近 Stage2 的 streaming one-step validation；
- offline wandb 规避集群网络问题。

问题：

- 早期 D3.2 / D4 前后仍出现 ghost / residual artifact；
- 检查发现 D3.2 target trimming 仍有旧逻辑风险；
- 需要进一步对齐 Stage2 v6.4 的前 22 latent 规则。

结论：

- D3.2 可用于观察 Stage3 初步趋势；
- 不再视作最终正确实现。

### v7-D4.1：turn-isolated / dfake=5 语义

目标：

- 更靠近 DMD2 的 generator / fake 交替更新；
- fake loss 不回传 student；
- student loss 不更新 G_fake；
- fake optimizer 和 student optimizer 分开。

结论：

- 梯度归属思路正确；
- 但远端 DDP / DeepSpeed rank 对齐、fake backward 组织上有工程阻塞；
- 不作为当前 48 卡主线。

### v7-D4.2：single-runner dfake=5 折中线

目标：

- 不再强行彻底 turn-isolated；
- 在 single runner 内实现更接近 DMD2 的 detach、fake optimizer、dfake=5；
- 使用 USMGT Stage1 teacher；
- 修正 teacher 前 22 对齐。

结论：

- 2GPU smoke 到第二个 generator turn；
- 比 D4.1 更稳；
- 但它仍是 single-runner 工程折中，不是最干净的 dual-engine 方案。

### v7-D4.3：dual DeepSpeed engine

目标：

- 让 `G_fake` 真正由独立 DeepSpeed engine 管理；
- fake loss 通过 fake engine backward/step。

结论：

- 工程结构更干净；
- 但 fake backward / sync 很慢；
- 证明慢点不在 optimizer offload，主要在 fake full-attention backward/sync 或 rank 等待；
- 不直接上 48 卡长训。

### v7-D4.4：dual Accelerator + dual DeepSpeedPlugin 当前主验证线

目标：

- 使用两个 Accelerator + 两个 DeepSpeedPlugin；
- 更接近 Accelerate 官方多模型 DeepSpeed 形态；
- 保持 DMD2-style dual optimizer / dfake=5 语义；
- 使用 Stage2 `v6.4.1 step-6000` 初始化 student；
- 使用 USMGT Stage1 `step-3000` 初始化 G_real/G_fake。

关键文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh
```

48 卡 run：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1
```

当前结论：

- 48 卡已真正跑起来，跨节点 GPU 利用率健康；
- fake backward 仍是主要耗时；
- 当前没发现把整个 WAN body 错放进 fake optimizer 的问题；
- 还需要继续做 loss ownership / grad norm / fake 参数范围验证；
- 这是当前 Stage3 最值得继续验证的主线。

详细文档：

```text
doc/flashvsr_stage3_v7d44_validation_plan_20260518.md
doc/flashvsr_stage3_v7d_dmd_review_20260515.md
doc/flashvsr_stage3_loss_validation_plan_20260517.md
```

## 8. Stage1 USMGT Takano 微调线

目标：

- 使用新 Takano 20250205 4K 视频源；
- 对 GT 加 USM sharpness；
- 从 Stage1 `v5.3.5 step-10000` warm start；
- 学习率降到 `5e-6`；
- 作为“更清晰 Stage1 teacher”供 Stage3 的 G_real/G_fake 使用。

关键 checkpoint：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors
```

结论：

- 当前 Stage3 D4.2+ 默认指定这个作为 Stage1 teacher / G_real / G_fake 初始化；
- 数据和 USMGT 逻辑需要独立文件，不应污染通用 `streaming_dataset.py`。

## 9. 已经废弃或不建议继续的线

| 版本 / 路线 | 不建议继续的原因 |
|---|---|
| v4 packed / joint 早期线 | image/video joint 语义不够干净，loss 异常多 |
| v5.1 / v5.2 grouped image | 工程复杂、没有成为主线 |
| Stage1 streaming projector | Stage1 会产生 22/23 latent 不对齐风险 |
| Stage2 v6.1 / v6.2 作为最终推理 | 可用于诊断，但跳变仍明显 |
| Stage2 v6.3 小学习率 | 证明学习率不是主因，不继续扩展 |
| Stage3 v7-A / v7-B | 只是链路验证，不是完整 DMD |
| Stage3 v7-C6 / D3.2 | 可跑但仍有权重、target trimming、validation 对齐风险 |
| Stage3 D4.1 | 语义清楚但工程 rank/DeepSpeed 阻塞，不作为当前主线 |
| Stage3 D4.3 | 结构干净但 fake backward 太慢，暂不长训 |

## 10. 现在要看什么文档

| 想查的问题 | 文档 |
|---|---|
| 每天具体做过什么 | `doc/FLASHVSR_WORKLOG.md` |
| 稳定母本代码 / config / sh / ckpt | `doc/flashvsr_stable_experiment_registry.md` |
| v4 为什么废弃 | `doc/flashvsr_v4_iteration.md` |
| v5 各版本关系 | `doc/flashvsr_v5_iteration.md` |
| Stage2 唯一执行版设计 | `doc/flashvsr_stage2_v6_design.md` |
| Stage2 跳变 A-H probe | `doc/flashvsr_stage2_v6_jump_probe_20260509.md` |
| Stage3 总计划 | `doc/flashvsr_stage3_dmd_plan_20260511.md` |
| Stage3 显存 / decoder 对齐 | `doc/flashvsr_stage3_memory_alignment_plan_20260514.md` |
| Stage3 DMD 评审对话 | `doc/flashvsr_stage3_v7d_dmd_review_20260515.md` |
| Stage3 loss 验证计划 | `doc/flashvsr_stage3_loss_validation_plan_20260517.md` |
| Stage3 D4.4 当前验证 | `doc/flashvsr_stage3_v7d44_validation_plan_20260518.md` |
| 给 leader 的阶段汇报 | `doc/flashvsr_training_group_meeting_20260508.md`、`doc/flashvsr_training_group_meeting_20260511.md`、`doc/flashvsr_training_group_meeting_20260514.md` |

## 11. 当前状态和后续优先级

当前已经明确：

- Stage1 `v5.3.5 / 535` 是 89f 稳定 teacher；
- Stage2 `v6.4.1 / 641` 是当前 streaming sparse-causal teacher；
- Stage3 当前主验证线是 `v7-D4.4`；
- SeedVR / FlashVSR official baseline 已有统一测试目录；
- Stage2 的 chunk artifact 不能只靠调学习率解决；
- Stage3 的关键风险是 DMD/fake 更新语义、loss 权重、one-step streaming validation、fake backward 性能和 ghost artifact。

优先级：

| 优先级 | 任务 |
|---|---|
| P0 | 继续观察 `v7-D4.4` 48 卡 loss、wandb、validation |
| P0 | 做 D4.4 loss ownership / grad norm 验证 |
| P1 | 用固定 10 synthetic / 11 real 测 Stage3 checkpoint |
| P1 | 对比 Stage2 `641`、Stage3 `D3.2`、Stage3 `D4.4` 的 ghost / residual |
| P2 | 如果 D4.4 fake backward 仍过慢，继续拆 fake LoRA-only / projector-only / Stage1 backward timing |

