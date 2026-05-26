# FlashVSR / LucidVSR 组会汇报（2026-05-18）

## 1. 本次汇报主线

5 月 14 日上次汇报时，Stage3 已经完成 `v7-A / v7-B` 的基础链路：one-step student、random latent reconstruction、pixel / LPIPS、显存与数据侧问题基本拆清楚。当时的下一步计划是 `v7-C`：把 FlashVSR 论文里的 `G_real / G_fake` 接进来，并尝试 DMD / trainable fake diffusion critic。

这次汇报按实际推进顺序整理：

| 阶段 | 核心问题 | 当前结论 |
|---|---|---|
| `v7-C0` 到 `v7-C6` | 能不能把 `G_real / G_fake` 真的塞进 Stage3 runner | 已接入，C6 已有 trainable `G_fake`，但 validation / W&B / 显存与长训稳定性仍有问题 |
| `v7-D` 到 `v7-D3.2` | 从能跑的 C6，推进到更接近论文语义的正式训练 | D3.2 能 48 卡长训并出结果，但仍有 teacher 对齐、训练慢、视觉鬼影/波动问题 |
| 并行 Stage1 USMGT | Stage3 的 teacher 是否足够清晰 | 新 Stage1 USMGT `step-3000` 已产出，作为后续 D4 的 `G_real/G_fake` 初始化 |
| `v7-D4.1 / D4.2 / D4.4` | 按 DMD2 语义修 dfake=5、optimizer 归属和双 DeepSpeed 管理 | D4.2 是稳妥备用线，D4.4 是当前 48 卡正式长训线 |

## 2. 从 v7-C 计划到实际落地

### 2.1 v7-C 的目标

5 月 14 日汇报里的 `v7-C` 不是已经完成的实验，而是下一步计划：在 Stage3 one-step student 的基础上加入完整 DMD 角色。

| 角色 | 来源 | 状态 | 用途 |
|---|---|---|---|
| student / `G_one` | Stage2 v6.4.1 `step-6000` | 训练 | one-step streaming VSR |
| `G_real` | Stage1 full-attention teacher | 冻结 | DMD real reference |
| `G_fake` | Stage1 teacher copy | 训练 | 学习 student fake distribution |
| Wan VAE / LPIPS | 预训练冻结模块 | 冻结 | reconstruction / perceptual loss |

这里最关键的是：`G_fake` 不是普通 loss head，而是一个需要单独 optimizer 的 trainable fake score model。因此 C 线的第一目标不是视觉效果，而是把多模型、多 optimizer、DMD score 路径跑通。

### 2.2 v7-C 实际做了什么

`v7-C` 是分步推进的，不是一次把所有东西塞进去：

| 版本 | 做了什么 | 结论 |
|---|---|---|
| `C0` | dual optimizer runner 骨架 | 证明 student optimizer + fake optimizer 的工程入口可行 |
| `C2 / C3` | 接 frozen `G_real / G_fake` probe | 可以加载 Stage1-like reference 模型做 DMD probe |
| `C4 / C5` | 接 DMD direction / DMD student loss | DMD2-style student loss 能进入 student graph 并 backward |
| `C6` | 把 `G_fake` 变成 trainable fake diffusion critic | 终于从 frozen probe 推进到真正 trainable `G_fake` |

因此，回答“C 最后怎么样了”：`G_real / G_fake` 确实被接进去了，C6 已经有 trainable `G_fake`，并且做过 2GPU 全尺寸 smoke、short run、resume 和 48GPU 启动。

### 2.3 C6 暴露的问题

C6 的意义是把结构接上，但它还不是干净的最终版本。主要问题有三类：

| 问题 | 影响 |
|---|---|
| validation 在保存点过重 | C6 常驻 `G_real/G_fake`，validation 如果没有严格 inference/no-grad，会显著放大显存和耗时 |
| W&B 在线同步不稳定 | 远端网络 `ConnectTimeout`，后来改成 offline W&B + 后台同步 |
| 代码快照和命名混乱 | C 系列一路 debug，日志和 snapshot 容易混入旧命名，需要干净线 |

C6 仍有价值：`step-100 / step-200` 做过 10 个 synthetic one-step 推理，约 `16.2s/video`，证明 Stage3 one-step 的速度潜力已经出现。但 C6 不是最后主线，因为它主要解决“塞进去能跑”，还没有充分解决 DMD2 语义、teacher wrapper、长训稳定性。

## 3. 从 C6 到 D 系列

### 3.1 为什么从 C6 切到 D

`v7-D` 不是另起炉灶，而是从 C6 的稳定快照出发，清理 validation / W&B / 代码命名，并开始把 C6 的“能跑 DMD”推进为“更接近论文和 DMD2 训练方式的 DMD”。

### 3.2 D stable / D1

早期 D 线主要解决长训工程问题：

| 版本 | 目的 | 结果 |
|---|---|---|
| `D stable` | 从 C6-stable 快照回到较稳训练形态 | 可作为短期稳定基线 |
| `D1` | 尝试训练 batch cache validation / 2-video validation | 2GPU smoke 可跑，但 48GPU step 很重，rank0 显存到 150GB+，暂停 |

结论：D1 的 validation 方向太重，不适合作为当前主线。后面继续回到更稳定的训练主干。

### 3.3 D2：shared-noise 修正，但发现 teacher wrapper 不对

D2 修了一个重要语义问题：DMD student loss 里 `G_real` 和 `G_fake` 应该在同一个 noisy latent / timestep 上比较 score，而不是各自采样不同 noise。这个修正和 DMD2 / OSEDiff 的 score difference 语义一致。

但 D2 同时暴露出更大的问题：虽然加载的是 Stage1 checkpoint，`G_real/G_fake` 的 forward 仍然套在 Stage2/Stage3 streaming wrapper 里。也就是说，D2 更准确地说是：

```text
Stage1 权重 + Stage2/Stage3 streaming wrapper
```

而不是 FlashVSR 论文目标里的：

```text
Stage1 full-attention teacher / Stage1 teacher copy
```

因此 D2 只能作为 shared-noise DMD ablation，不能作为严格 paper-aligned Stage3。

### 3.4 D3 / D3.1 / D3.2：修 teacher wrapper 并开始长训

D3 之后的主目标是把 `G_real/G_fake` 拉回 Stage1 non-streaming/full-attention teacher 语义。

| 版本 | 作用 |
|---|---|
| `D3` | 修 `G_real/G_fake` 的 projector temporal mode，改为 Stage1 `nonstreaming_aligned`，并做 shape / temporal map / optimizer ownership smoke |
| `D3.1` | 从 D3 复制干净正式线，去掉仅用于 debug 的 checksum / smoke 代码 |
| `D3.2` | 修 48GPU 数据分片 / 低利用率问题，成为可长训版本 |

D3.2 最终能 48 卡稳定训练，并且做了多轮 checkpoint 测试：

| checkpoint | 观察 |
|---|---|
| `step-1500` | 可用，但视觉仍有明显波动和残影/鬼影 |
| `step-2000` | 用户观察明显好于 step-1500，说明 D3.2 长训后会改善 |

D3.2 的问题也比较明确：

| 问题 | 说明 |
|---|---|
| 保存间隔太宽 | 一开始 `save_steps=500`，中间视觉变化不好追踪，后来后续配置改成 100 |
| 训练慢 / 利用率波动 | fake backward/sync、checkpoint 保存和数据侧都会造成周期性低利用 |
| 视觉鬼影严重 | 早期 checkpoint 尤其明显，后来 step-2000 好很多，但仍不是完全可接受 |
| teacher 对齐仍不是最终目标 | D3/D3.2 当时使用 teacher 后 22，即丢 position 0、保留 `[1,23)`；后来确认应改为 teacher 前 22 |

因此 D3.2 的定位是：很有价值的长训对照和可复刻 ablation，但不是最终 paper-aligned 版本。

## 4. D3.2 长训同时推进的 Stage1 USMGT teacher

在 D3.2 训练和排查的同时，另开了一条 Stage1 teacher 优化线。原因是：Stage3 的 `G_real/G_fake` 来自 Stage1 teacher，如果 teacher 本身偏软，Stage3 的 DMD/pixel/LPIPS 也容易把 student 往偏软的目标拉。

### 4.1 Stage1 USMGT 做了什么

从原 Stage1 v5.3.5 `step-10000` warm-start，用新的 Takano 20250205 4K 视频源继续 fine-tune，并对 GT 侧加入 Real-ESRGAN 风格 USM/sharpness 处理。

| 项 | 设定 |
|---|---|
| warm-start | Stage1 v5.3.5 `step-10000` |
| 新视频源 | Takano 20250205 4K |
| 学习率 | `5e-6` |
| 帧数 | 89f |
| projector | Stage1 `nonstreaming_aligned` |
| 目标 | 让 teacher 更锐、更适合作为 Stage3 real/fake 参考 |

当前 checkpoint：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors
```

### 4.2 Stage1 USMGT 当前结果

| 结果 | 说明 |
|---|---|
| `step-3000` 已产出 | 后续 D4.2 / D4.4 默认用它初始化 `G_real/G_fake` |
| 数据可视化通过 | `gt_raw / gt_usm / lq_degraded` 检查没有发现 LQ 异常发灰 |
| 推理速度不变 | 架构不变，Stage1 仍约 210s/video；优化目标是 teacher 质量，不是速度 |
| 风险 | sharpness 过强可能带 halo，所以先作为 Stage3 teacher 候选，不直接宣称最终产品模型 |

这条线的意义是：D3.2 暴露出视觉波动和鬼影后，后续 D4 不只是改 runner，也换用了更适合作为 teacher 的 Stage1 USMGT `step-3000`。

## 5. 从 D3.2 到 D4.4：修正目标与当前版本

D3.2 能长训并且 `step-2000` 视觉明显好于 `step-1500`，但它仍暴露出 teacher latent 对齐、fake update 语义和 `G_fake` 分布式管理的问题。因此 D4 系列的目标不是重新做一条 unrelated 训练线，而是把 D3.2 修成更接近 FlashVSR / DMD2 语义的正式版本。

| 版本 / 问题 | 主要改动 | 当前结论 |
|---|---|---|
| D3.2 遗留问题 | teacher 用后 22 `[1,23)`；fake update ratio 语义不清；fake loss 需要确认不回传 student | D3.2 保留为有价值长训对照，但不是最终 paper-aligned 版本 |
| D4.1 | 尝试把 student phase 和 fake phase 物理分开，更接近 DMD2 runner | 明确了梯度归属，但低卡 smoke 中 rank 等待严重，不适合作为正式长训 |
| D4.2 | 回到单 runner 组织；teacher 改前 22 `[0,22)`；fake 每步更新，student 每 5 步更新；fake loss 使用 `z_pred.detach()` | 当前稳妥备用线，dfake=5 和梯度归属清楚 |
| D4.3 | 尝试 student 一个 DeepSpeed engine、`G_fake` 一个 raw DeepSpeed engine | 证明 fake engine 可独立更新，但 fake backward 在小规模 smoke 中很慢 |
| D4.4 | 两个 Accelerator + 两个 DeepSpeedPlugin，让 student 和 `G_fake` 都交给 DeepSpeed 管理 | 当前 48 卡正式长训线；更正规，但工程成本和 fake sync 开销更高 |

D4.4 已确认：`G_fake` 只训练 LoRA 和 `lq_proj_in`，没有误训 full WAN body；fake full-attention 走 `flash_attn_2`；正式 run 默认 no-offload；W&B 已确认可在 t5 远端直接同步。

当前 D4.4 48 卡 run：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1
```

当前仍需保留的风险：

| 风险 | 说明 |
|---|---|
| fake backward/sync 仍慢 | 主要来自 trainable Stage1-like `G_fake` 的大 backward / ZeRO2 sync |
| D4.4 视觉还没充分评测 | 需要等更多 checkpoint 后与 D3.2 step-2000 对比 |
| D4.4 不一定优于 D4.2 | D4.4 工程上更正规，但成本更高；D4.2 仍是稳妥备用线 |

## 6. Loss / 梯度归属验证

这次也专门做了代码和小实验验证，防止 Stage3 只是“能跑”但 loss 目标串线。

| 验证点 | 当前结论 |
|---|---|
| fake FM 是否回传 student | 不回传，fake loss 内部使用 `z_pred.detach()` |
| DMD student 是否误更新 `G_real/G_fake` | 不更新，real/fake probe 在 `torch.no_grad()` 内，DMD target detach |
| `G_real` 是否被冻结 | ownership smoke 中 `real_delta=0` |
| DMD 是否真的生效 | D4.4-DMDOnly 16 卡对照中，pixel/recon 全关，runner5/10 仍出现非零 `dmd_student/dmd_grad` |

DMD-only 对照不是正式训练线，因为没有 validation / wandb，且 16 卡 fake sync 波动很大。它的价值是证明：D4.4 里 DMD branch 不是只打印，确实能在 fake critic 更新后产生 student-side loss / gradient。

## 7. 推理评测与 benchmark

### 7.1 D3.2 step-1500 / step-2000

用正确 Stage3 一步 streaming / KV-cache 推理代码测了 D3.2：

| checkpoint | 状态 |
|---|---|
| D3.2 `step-1500` | 10 个 synthetic 已测，视觉仍有明显 ghost / 波动 |
| D3.2 `step-2000` | 10 个 synthetic 已测，用户观察明显好于 `step-1500` |

这个结果说明：D3.2 虽然不是最终目标，但长训确实会改善视觉，后续 D4 也需要给足训练步数后再比较。

### 7.2 20 synthetic + 11 real benchmark

为了组会 / PPT，重新整理了标准 benchmark：

| 数据 | 数量 |
|---|---:|
| Takano20250205 synthetic light x4 | 20 |
| real LQ | 11 |

对比方法：

```text
FlashVSR official
SeedVR2-3B
SeedVR3B
Stage1 v5.3.5
Stage1 USMGT
Stage2 v6.4.1
Stage3 v7-D3.2 step-2000
```

平均端到端速度：

| 方法 | synthetic 秒/视频 | real 秒/视频 |
|---|---:|---:|
| FlashVSR official | 42.8 | 41.9 |
| SeedVR2-3B | 50.7 | 53.3 |
| SeedVR3B | 228.7 | 230.2 |
| Stage1 v5.3.5 | 209.5 | 210.5 |
| Stage1 USMGT | 209.9 | 210.3 |
| Stage2 v6.4.1 | 149.3 | 141.7 |
| Stage3 D3.2 | 18.3 | 18.5 |

关键解读：

| 观察 | 说明 |
|---|---|
| FlashVSR official 约 42s/video | 与之前 89f full end-to-end 40s 量级一致 |
| Stage3 D3.2 约 18s/video | one-step 确实带来明显速度优势 |
| 当前瓶颈不是推理速度 | 主要矛盾是 Stage3 训练目标和视觉稳定性 |

## 8. 当前主要工程问题

| 问题 | 当前处理 |
|---|---|
| fake backward / sync 慢 | 已排除 full WAN 误训、缺 flash-attn、optimizer offload 等原因；更可能是 trainable Stage1-like fake critic 本身重 |
| rank 等待 / 低利用率 | 低效 smoke 不再长时间占卡；正式 D4.4 48 卡目前比 2GPU/16GPU smoke 健康 |
| W&B 同步 | 确认 t5 正常 zsh 环境可直接 sync，D4.4 已挂远端直接同步 |
| 安全停实验 | 不再用模糊 `pkill`；只用 tmux Ctrl-C 或确认完整 PID 后处理 |

## 9. 当前结论与下一步

当前结论：

| 结论 | 说明 |
|---|---|
| C 线已经完成从计划到 trainable `G_fake` 的接入 | C6 证明结构能跑，但不是最终主线 |
| D3.2 是有价值长训对照 | step-2000 变好，但 teacher 对齐和 ghost 问题让它不能作为最终版本 |
| Stage1 USMGT `step-3000` 已成为新 teacher 候选 | 后续 D4 默认用它初始化 `G_real/G_fake` |
| D4.2 是稳妥备用线 | dfake=5、front-22 teacher 对齐、梯度归属清楚 |
| D4.4 是当前正式长训线 | 双 Accelerator + 双 DeepSpeedPlugin，更正规但更重 |

下一步：

| 优先级 | 内容 |
|---|---|
| 1 | 继续观察 D4.4，等 `step-500 / step-1000` 后做同一套 Stage3 推理评测 |
| 2 | 对比 D4.4 与 D3.2 `step-2000` 的残影、细节和时序稳定性 |
| 3 | 如果 D4.4 视觉不如 D3.2，优先回到 D4.2 做更稳的 48 卡复现 |
| 4 | 保留 fake backward/sync profiling，但不让低效 smoke 长时间占 16 卡 |
| 5 | 对最终候选补做更轻量的 loss / optimizer ownership 验证，避免正式训练代码继续堆 debug |
