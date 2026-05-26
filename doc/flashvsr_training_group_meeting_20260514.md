# FlashVSR / LucidVSR 组会汇报（2026-05-14）

## 1. 本周目标：把 Stage3 从想法拆成可验证的工程路线

上次汇报后，主线从 Stage2 sparse-causal 训练，推进到 Stage3 one-step distillation。Stage3 不是直接加一个 loss，而是要把 Stage2 的 streaming sparse-causal model 蒸馏成 one-step model，同时加入 DMD、pixel loss 和 LPIPS。

当前采用的拆分方式：

| 阶段 | 目标 | 状态 |
|---|---|---|
| `v7-A` | 先验证 one-step + decoder + pixel / LPIPS 链路 | 已完成基础闭环 |
| `v7-B` | 修正 random latent decode、previous detach、显存和数据侧问题 | 已完成进入 C 前的收口 |
| `v7-C` | 接入完整 DMD，包含 `G_real / G_fake` 和 dual optimizer | 下一步开始 |

核心思路：先把每个不确定点拆开验证，不直接把 DMD、decoder、LPIPS、`G_fake` 全部一次性堆进训练，否则很难判断问题来自哪里。

## 2. Stage3 总体设计

Stage3 论文目标是：

```text
Stage2 sparse-causal model -> one-step streaming VSR model
```

Loss 组成：

| Loss | 作用 |
|---|---|
| DMD | 让 one-step 输出分布接近真实高质量视频分布 |
| Flow Matching | 保留 diffusion / flow 训练约束 |
| Pixel MSE | 直接约束像素细节和时序稳定性 |
| LPIPS | 约束感知质量 |

模型角色：

| 角色 | 来源 | 是否训练 | 用途 |
|---|---|---|---|
| `G_one` / student | Stage2 `v6.4.1 / 641` | 训练 | 最终 one-step streaming model |
| `G_real` | Stage1 full-attention teacher | 冻结 | DMD 的 real reference |
| `G_fake` | Stage1 teacher copy | 单独训练 | DMD 的 fake distribution model |
| Wan VAE decoder | Wan pretrained | 冻结 | 把 latent decode 到 pixel space |
| LPIPS / VGG | pretrained VGG | 冻结 | perceptual loss |

其中 `G_fake` 不是普通 loss 分支，它需要独立 optimizer。因此完整 DMD 不能直接塞进现有单 optimizer runner。

## 3. `v7-A`：先验证 one-step + pixel / LPIPS 基础链路

`v7-A` 的目标不是完整 Stage3，而是先回答一个问题：

```text
Stage2 checkpoint 能不能一步生成 z_pred，并 decode 到 pixel space 后计算 MSE / LPIPS？
```

### 3.1 `v7-A` 做了什么

| 模块 | 实现 |
|---|---|
| student 初始化 | 从 Stage2 checkpoint 导入 LoRA 和 `lq_proj_in` |
| one-step | 用 `scheduler.step(..., to_final=True)` 得到 `z_pred` |
| pixel branch | `z_pred -> Wan decoder -> x_pred` |
| loss | `flow + MSE + 2 * LPIPS` |
| validation | one-step validation，不再沿用 50-step |

代码入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py
```

### 3.2 `v7-A` 遇到的问题

| 问题 | 为什么关键 | 处理结果 |
|---|---|---|
| 不能 decode 整段 89 帧算 pixel / LPIPS | Wan decoder 显存过大，直接整段 decode 不现实 | 先固定 decode temporal prefix 做 smoke |
| LPIPS/VGG 破坏 DeepSpeed state 保存 | frozen LPIPS 被错误注册进 model tree，导致 training state 保存失败 | 改成 loss-only cache，不注册为子模块 |
| 首帧 pixel 权重不明确 | Wan VAE 首帧不压缩，首帧监督需要特殊处理 | pixel 首帧乘 `4`，后续 LPIPS 也统一乘 `4` |

### 3.3 `v7-A` 验证结果

| 验证项 | 结果 |
|---|---|
| one-step forward/backward | 通过 |
| MSE / LPIPS 参与 loss | 通过 |
| checkpoint 保存 | 通过 |
| training state 保存 | 修复后通过 |
| 从 training state resume | 通过 |

代表性 smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_smoke_statefix
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_true_resume_from_step10
```

结论：`v7-A` 证明基础 one-step + decoder reconstruction 链路可行，但它不符合论文里 random latent decode 的设计，也没有 DMD。

## 4. `v7-B`：修 random latent decode 和显存语义

进入 `v7-B` 后，重点不是马上接 `G_fake`，而是先修 `v7-A` 的语义缺口。

### 4.1 为什么必须做 `v7-B`

论文 Stage3 里有一个关键约束：

```text
Due to memory constraints, two latents are randomly selected per iteration for decoding,
with previous ones detached from gradients.
```

这句话决定了三个关键问题：

| 问题 | 为什么关键 |
|---|---|
| 随机选 2 个 latent | 不能永远只训前缀，否则 pixel / LPIPS 监督不覆盖全视频 |
| previous detach | Wan decoder 是因果/上下文相关的，不能完全无视前面的 latent |
| 只让 selected latent 反传 | 否则随机到后部 latent 时等价整段视频反传，显存不可控 |

### 4.2 `v7-B` 最终采用的语义

| 情况 | 当前做法 |
|---|---|
| 抽到开头 latent | 直接 decode `[0:recon_end]`，首帧 pixel / LPIPS 权重乘 `4` |
| 抽到中间 / 后部 latent | prefix `[0:recon_start)` no-grad forward，只建立 decoder context |
| 反传 | 只对选中的连续 `2` 个 latent 对应帧反传 |
| 不采用 | 不使用自定义 spatial tile 作为主线，避免改变 Wan decoder 语义 |

代码入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
```

## 5. `v7-B` 遇到的不确定性与解决结果

### 5.1 显存问题：到底是谁把显存打满

最开始的问题是：Stage3 加 decoder / LPIPS 后，显存明显升高。为避免误判，做了分解 probe。

| Probe | 开启内容 | 观测 | 结论 |
|---|---|---|---|
| flow-only | Stage2 / flow 底座 | 约 `40GB` 级 | DiT / block sparse 底座不是主因 |
| z_pred-only | one-step student，不 decode | 约 `40GB` 级 | one-step latent 生成不是主因 |
| MSE-only | 加 Wan decoder selected-window backward | 约 `119GB` 级 | Wan decoder reconstruction 是显存大头 |
| MSE + LPIPS | 再加 LPIPS | 约 `119GB` 级 | LPIPS 串行后不是主要峰值来源 |

结论：Stage3 显存主要卡在 Wan decoder selected-window backward。后续优化重点应该是 graph 生命周期、decoder offload、串行执行，而不是盲目调 DataLoader。

### 5.2 DataLoader worker 为什么会影响 GPU 显存

之前 worker 开大时 GPU 显存会波动，怀疑数据退化仍在占 GPU。排查后发现：

| 问题 | 发现 |
|---|---|
| DataLoader worker 是否创建 CUDA context | 是，旧 worker init 会触发 CUDA 相关逻辑 |
| 在线退化是否上 GPU | 旧逻辑可能在 worker 内创建 GPU 中间 tensor |
| 现象 | 多 worker 时出现一串约 `614MB` 的 worker CUDA context |

修复：

| 修复项 | 结果 |
|---|---|
| DataLoader worker init 改成 CPU-only seed | worker 不再创建 CUDA context |
| 在线退化强制 CPU | 退化不再抢 GPU 显存 |
| `worker=8` 复测 | 能跑，但没有稳定快于 `worker=2` |

当前建议：

| 设置 | 结论 |
|---|---|
| `dataset_num_workers=2` | 当前最稳妥默认 |
| `worker=8` | 可用但不作为默认 |
| `GC offload on` | 更稳，约 `120GB` |
| `GC offload off` | 稍快，约 `140GB`，显存余量更小 |

### 5.3 CPU 退化是否把 LQ 弄烂

外部 CPU/GPU ablation 出现不稳定结果：有的 `lq_gpu` 或 repeat 结果变灰，不能直接证明训练数据坏了。

因此改成更直接的检查：在训练代码里、模型接收前 dump DataLoader batch。

训练入口 dump：

```text
/Users/lixiaohui/Desktop/stage3_v7b_training_input_dump_cpu_degra_17f_20260514/before_model
```

重点文件：

```text
sample_000/gt_before_model.mp4
sample_000/lq_before_model.mp4
sample_000/meta.json
```

结论：训练入口 dump 中的 `lq_before_model.mp4` 视觉正常。因此当前更可信的判断是：CPU 在线退化送进模型前没有把 LQ 弄烂。

### 5.4 89 帧到底怎么监督

Stage3 继承 Stage2 `641` 规则。当前固定：

| 项 | 规则 |
|---|---|
| 输入 | 正常 `89` 帧 |
| LQ projector | `89 -> 22 latent` |
| GT target | 只取前 `85` 帧 |
| Wan VAE target | `85 -> 22 latent` |
| loss / inference | 对齐 `22 latent / 85 effective frames` |

这个规则来自当前效果较好的 `v6.4.1 / 641`，后续 `v7-B / v7-C` 先沿用，不再默认做额外 `85 + repeat4` 预处理。

## 6. `v7-B` 收口结论

| 项 | 当前结论 |
|---|---|
| one-step student | 能跑 |
| random latent decode | 已按 full-prefix detach 语义实现 |
| pixel / LPIPS | 已接入，首帧权重统一为 `4` |
| 数据退化 | CPU 在线退化可用，训练入口 LQ 正常 |
| 显存主因 | Wan decoder selected-window backward |
| block sparse | CUDA extension 可用，不会 silent fallback |
| training state | `v7-A` 已验证保存 / resume；`v7-B` 基础 smoke 可跑 |
| 完整 DMD | 尚未接入，需要 `v7-C` |

`v7-B` 的意义：它不是最终 Stage3，但已经把完整 DMD 前最容易混乱的几件事拆清楚了。

## 7. 为什么下一步必须进入 `v7-C`

完整 DMD 需要 `G_real / G_fake`，其中 `G_fake` 需要独立 optimizer 和交替更新。当前 DiffSynth runner 是单 optimizer 结构：

```text
optimizer = AdamW(model.trainable_modules(), ...)
```

这不能正确表达 DMD2 的训练逻辑。

| 需求 | 当前 runner 是否支持 | 处理 |
|---|---|---|
| student optimizer | 支持 | 保留 |
| `G_fake` optimizer | 不支持 | `v7-C` 新写 |
| student / fake 交替更新 | 不支持 | `v7-C` 新写 |
| `G_real` frozen no-grad | 可实现 | `v7-C` 显式管理 |
| 两套 state 保存 / resume | 不支持 | `v7-C` 新写 |

因此不能在 `v7-B` 里硬打开 `stage3_fake_fm_weight`。这样虽然可能“跑起来”，但语义会错。

## 8. `v7-C` 计划

### 8.1 模型角色

| 角色 | 初始化 | 是否训练 | 用途 |
|---|---|---|---|
| Student / `G_one` | `v6.4.1 / 641 step-6000` | 训练 LoRA + `lq_proj_in` | one-step 输出 |
| `G_real` | Stage1 full-attention teacher | 冻结 | DMD real reference |
| `G_fake` | 从 `G_real` copy | 单独训练 | DMD fake distribution |
| Wan VAE decoder | Wan pretrained | 冻结 | selected latent decode |
| LPIPS / VGG | pretrained VGG | 冻结 | perceptual loss |

### 8.2 单 step 计划

| 步骤 | 操作 | 显存原则 |
|---|---|---|
| 1 | 读 89 帧 video batch，沿用 `641` 规则 | CPU degradation |
| 2 | Student one-step forward 得到 `z_pred` | 保留 student graph |
| 3 | 随机选 2 latent decode，算 MSE / LPIPS | prefix no-grad，selected grad |
| 4 | `G_real` frozen forward | no-grad，立刻释放 |
| 5 | `G_fake` forward / update | 单独 optimizer |
| 6 | student optimizer step | 只更新 student |
| 7 | 保存 student / fake / optimizer state | 支持 resume |

### 8.3 验收顺序

| 阶段 | 开启内容 | 验收标准 |
|---|---|---|
| C0 | dual optimizer skeleton | 两套 optimizer / scheduler 能保存和 resume |
| C1 | student reconstruction loss | random window、首帧权重、pixel / LPIPS 正常 |
| C2 | `G_real` frozen forward | `G_real` 无梯度，显存可控 |
| C3 | `G_fake` forward，不更新 | DMD loss 能记录 |
| C4 | 打开 `G_fake` optimizer | fake grad norm 和 student grad norm 独立 |
| C5 | 完整 DMD | 20 step smoke 可保存完整 state |

## 9. 当前汇报结论

| 结论 | 价值 |
|---|---|
| Stage3 已经从概念拆成 `A/B/C` 三段 | 后续不会把所有风险混在一起 |
| `v7-A` 证明 one-step + decoder reconstruction 可行 | pixel / LPIPS 链路跑通 |
| `v7-B` 解决 random latent decode、显存分解、CPU 退化和数据入口问题 | 完整 DMD 前的基础已收口 |
| 当前显存主因明确是 Wan decoder selected-window backward | 后续优化方向明确 |
| CPU 退化训练入口 LQ 正常 | 数据入口不是当前主要风险 |
| 完整 DMD 必须进入 `v7-C` 独立 runner | 避免语义错误的“假 DMD” |

## 10. 下一步

| 优先级 | 动作 | 目的 |
|---|---|---|
| P0 | 新建 `v7-C0` dual-optimizer runner | 验证完整 DMD 工程骨架 |
| P0 | 固定使用 `641 step-6000` 初始化 student | 避免 teacher / student 起点摇摆 |
| P1 | 接入 `G_real` frozen forward | 验证 real teacher 无梯度 |
| P1 | 接入 `G_fake` optimizer | 实现 DMD2 式 fake model 更新 |
| P1 | 20 step smoke + state 保存 | 确认可 resume、可扩展 |

详细技术记录：

```text
doc/flashvsr_stage3_dmd_plan_20260511.md
doc/flashvsr_stage3_memory_alignment_plan_20260514.md
```
