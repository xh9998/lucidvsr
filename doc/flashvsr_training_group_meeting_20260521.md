# FlashVSR / LucidVSR 组会汇报（2026-05-21）

## 1. 本次汇报主线

5 月 18 日之后，主要工作从“把 Stage3 DMD 跑起来”转向“确认 Stage3 是否真的按预期训练”。这几天重点做了三件事：

| 方向 | 做了什么 | 当前结论 |
|---|---|---|
| Stage3 D4.4 稳定化 | 重新审阅 DMD2 / OSEDiff / FlashVSR 逻辑，整理 D4.4 训练实现 | D4.4 是当前最接近论文语义的主线版本 |
| 梯度与 loss 验证 | 验证 student / fake / real 的更新归属，并量化各 loss 对参数的梯度影响 | 梯度归属基本正确，但 Flow / LPIPS 明显强于 DMD |
| 对照实验与评测 | 启动 `flow_weight=0.1` 对照，并补测 D4.4 100/200/300 step | 用固定 10 个 synthetic 视频观察训练早期视觉变化 |

一句话总结：Stage3 现在不是链路跑不通的问题，而是需要把 DMD、Flow、LPIPS 三类目标的训练贡献调到合理范围。

## 2. Stage3 D4.4 当前版本

D4.4 是这几天主要维护的 Stage3 版本。它相对前面的 D3.2 / D4.2，重点是更正规地管理 student 和 trainable fake model。

### 2.1 D4.4 早期稳定性问题与修复

D4.4 最初按 48 卡恢复训练时并不稳定，clean2 resume 在还没到下一次 validation / step-200 checkpoint 前再次退出。复查 `run.log` 后，问题不是数据读取或 W&B，而是 fake 分支的大规模同步：

| 现象 | 定位 |
|---|---|
| 训练停在 fake backward 附近 | 报错集中在 `fake_accelerator.backward(fake_loss)` |
| NCCL watchdog timeout | DeepSpeed/ZeRO 在 fake 侧 `ALLREDUCE` 超时 |
| `NumelIn=287838720` | 与 `G_fake.lq_proj_in` 参数规模基本吻合 |
| 数据读取异常 | 未见 `SlowDown`、DataLoader worker 或 conductor/notary 报错 |

当时的关键判断是：D4.4 为了贴近 DMD2，让 `G_fake` 每个 runner step 都更新；但 `G_fake` 包含很大的 `lq_proj_in`，每步都把这部分参数纳入 ZeRO 同步，通信压力过大，导致 48 卡长训容易在 fake backward / all-reduce 处断掉。

最终没有采用“冻结 fake lq_proj_in”的方案，因为这样会削弱 `G_fake` 作为 fake distribution critic 的条件适应能力。当前采用的是折中修复：

| 修复 | 目的 |
|---|---|
| `G_fake` LoRA 每个 runner step 更新 | 保留 dfake=5 的 fake critic 高频更新语义 |
| `G_fake.lq_proj_in` 每 5 个 runner step 更新一次 | 降低最大通信参数组的同步频率 |
| fake 侧 DeepSpeed bucket 下调 | 减少单次 collective 压力 |
| 48 卡改为 40 卡 fresh | 避开 a9 坏卡和 48 卡不稳定组合 |

修复后启动了新的 40 卡 D44版本2。这个版本不是简单“少用卡”，而是同时改了 fake 侧同步策略，解决了前面 D4.4 反复断在 fake branch 的主要工程问题。

| 模块 | 当前设定 |
|---|---|
| student 初始化 | Stage2 v6.4.1 `step-6000` |
| `G_real / G_fake` 初始化 | Stage1 USMGT `step-3000` |
| 训练目标 | Flow + MSE + `2 * LPIPS` + DMD + Fake FM |
| 更新方式 | `G_fake` 每个 runner step 更新，student 每 5 个 runner step 更新一次 |
| 工程实现 | student 和 `G_fake` 分别用 Accelerator / DeepSpeed 管理 |

当前 40 卡主实验：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5
```

## 3. 关键验证结果

### 3.1 训练归属验证

我们做了专门的 16 卡 ownership / gradcheck，确认 D4.4 没有明显的“串梯度”问题。

| 验证点 | 结果 |
|---|---|
| fake-only turn 是否更新 student | 不更新，`student_delta=0` |
| `G_real` 是否被冻结 | 冻结，`real_delta=0` |
| 首帧 pixel / LPIPS 权重 | `x4` 生效 |
| generator turn 中 fake 是否更新 | 会更新，这是当前 dfake=5 设计，不是 bug |

这里需要注意命名：日志里的 `generator_turn` 更准确应理解为 `student_update_turn`，表示这一轮 student 也更新，并不表示 fake 停止更新。

### 3.2 DMD 尖峰处理

D4.4 日志里发现过一次 DMD student loss 极端尖峰。复查后确认原因不是 NaN，而是 DMD loss 用平方项，少数位置的较大梯度会把 loss 拉高。

已加处理：

| 改动 | 作用 |
|---|---|
| `stage3_dmd_loss_max=3.0` | 限制单个 batch 的 DMD loss 异常冲击 |
| 保留 `stage3_dmd_grad_norm_max=5.0` | 继续作为硬异常保护 |
| 日志新增 `dmd_loss_clamp` | 方便判断是否触发 clamp |

这个处理保留 DMD 方向，只限制极端 batch 的单步影响。

需要注意：这个是代码和 config 层面的补丁，只有后续重启或新开 D4.4 训练才会真正生效；已经在内存里运行的 40 卡进程不会自动吃到这个改动。

## 4. Loss / 梯度量级复查

Gemini 提醒的核心问题不是 loss 数值本身，而是不同 loss 对 trainable 参数的梯度贡献可能不平衡。我们做了独立验证，没有修改正式训练代码。

参数梯度量级结果：

| Loss | sampled params L2 | LoRA L2 | LQ projector L2 |
|---|---:|---:|---:|
| Flow | 1.7389 | 1.3424 | 1.1053 |
| MSE | 0.0602 | 0.0382 | 0.0466 |
| LPIPS | 1.5213 | 0.6436 | 1.3785 |
| DMD | 0.0267 | 0.0117 | 0.0240 |

当前判断：

| 现象 | 解释 |
|---|---|
| W&B loss 看起来健康，但视觉可能 100 step 好、200/300 step 变糊 | loss 标量下降不一定代表目标方向正确 |
| DMD loss 数值不算很大，但参数梯度很弱 | DMD 对 trainable 参数的实际影响可能被 Flow / LPIPS 压住 |
| LPIPS 权重 `2.0` 暂时不直接改 | 论文和 OSEDiff 默认也使用 `lambda=2`，先做对照，不直接偏离论文 |

结论：当前最值得先试的是降低 Flow 权重，而不是马上削弱 LPIPS。

## 5. Flow 降权对照实验

为了验证 Flow 是否过强，启动了一个 16 卡短训对照：

| 项 | 设定 |
|---|---|
| 实验 | D4.4 `flow_weight=0.1` |
| 其它权重 | MSE=1，LPIPS=2，DMD=1，Fake FM=1 |
| 目的 | 看降低 Flow 后，早期 checkpoint 是否更少变糊 |
| 机器 | `bfs6vaz4d6` + `i6hf4scd4y`，共 16 卡 |

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_flow0p1_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260521_111700
```

首条 loss 已确认权重生效：

```text
loss = 0.1 * flow + 1.0 * mse + 2.0 * lpips
```

这个实验的目标不是替代 40 卡主线，而是作为最小干预 ablation，判断“Flow 是否过强”。

## 6. 外部评测补充

为了避免只看训练 loss，补做了固定 synthetic 测试集的 one-step streaming 推理。

| 评测 | checkpoint | 输出位置 |
|---|---|---|
| D4.4 40 卡主线 | step-100 / 200 / 300 | `/Users/lixiaohui/Desktop/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521` |
| Flow0.1 对照 | step-50 / 100 | `/Users/lixiaohui/Desktop/stage3_flow0p1_vs_v7d32_10synthetic_20260521` |

D4.4 结果也已备份：

```text
s3://lxh/data/test/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521
```

当前观察重点：

| 问题 | 目的 |
|---|---|
| 100 step 是否好于 200/300 step | 判断训练后期变糊是不是持续现象 |
| Flow0.1 是否缓解变糊 | 判断 Flow loss 是否主导了优化方向 |
| D4.4 是否优于 D3.2 | 判断 D4.4 的 DMD / teacher 对齐修改是否有效 |

## 7. 推理时间口径更新

这次也重新拆了单视频推理时间，避免把模型加载、视频 I/O、保存文件和真实模型计算混在一起比较。PPT 和汇报里建议统一使用“模型部署完成后的模型运算时间 / 输出帧数”。

| 方法 | 模型运算合计 | 输出帧 | 模型时间 |
|---|---:|---:|---:|
| FlashVSR official | 11.01s | 85 | 0.13s/frame |
| LucidVR Stage3 D3.2 | 11.12s | 85 | 0.13s/frame |
| SeedVR2-3B | 55.44s | 89 | 0.62s/frame |
| SeedVR-3B | 231.17s | 89 | 2.60s/frame |

关键结论：去掉冷启动、模型加载、pipeline 初始化、视频读写和保存后，LucidVR Stage3 的核心推理时间已经和 FlashVSR official 基本一致。之前看到 FlashVSR / baseline 时间差异较大，主要是统计口径里混入了冷启动和外部 subprocess / I/O 开销。

## 8. D3.2 回看与 dfake 判断

D3.2 是这次对照里一个重要 baseline。它不是当前最接近论文语义的版本，但 2000 step 结果视觉上有改善，因此需要单独解释。

对 D3.2 `run.log` 做分段统计后，后期改善主要不是来自 `dmd_student` 下降：

| 区间 | loss | student | fake_loss | dmd_student |
|---|---:|---:|---:|---:|
| 1301-1500 | 1.0209 | 0.6026 | 0.0504 | 0.3679 |
| 1501-1700 | 1.0829 | 0.6417 | 0.0570 | 0.3843 |
| 1701-1900 | 1.0008 | 0.5822 | 0.0452 | 0.3735 |
| 1901-2084 | 1.0172 | 0.5878 | 0.0488 | 0.3806 |

结论：

- D3.2 在 1700 step 以后变好，主要来自 `student` 聚合支路和 `fake_loss` 下降；
- `dmd_student` 没有明显下降，反而基本持平或略高；
- 因为 D3.2 当时没有单独记录 `flow / mse / lpips`，所以无法进一步拆出 student 内部是谁下降。

这说明 D3.2 不能简单被丢掉：它是一个强 baseline。但从论文复现角度，D4.4 的 dfake=5 仍然有意义，因为它让 `G_fake` 更频繁跟随当前 student fake distribution。后续应以同等 student update 数的固定测试集视频对比，判断 dfake 主线是否真正优于 D3.2-like 同频训练。

## 9. 当前结论

| 问题 | 当前判断 |
|---|---|
| D4.4 代码是否明显错 | 暂时没有发现核心梯度归属错误 |
| DMD 是否真的接入 | 是，DMD 分支有非零 loss 和非零参数梯度 |
| 为什么 loss 健康但视频可能变差 | Flow / LPIPS 对参数梯度更强，DMD 相对弱，优化方向可能仍偏向重建/回归而不是 DMD 目标 |
| 是否应该马上大改论文权重 | 不建议直接大改，先通过 flow0.1、DMD 增权等短训对照判断 |
| Stage3 当前最大风险 | 不是跑不起来，而是 loss 权重和训练阶段动态还没完全稳定 |

## 10. 下一步计划

| 优先级 | 计划 | 目的 |
|---|---|---|
| P0 | 看 D4.4 step-100/200/300 与 flow0.1 对照视频 | 判断早期变糊主要由谁造成 |
| P0 | 继续补 flow0.1 更公平 step 对齐评测 | 16 卡和 40 卡样本数不同，需要看等样本量结果 |
| P1 | 如果 flow0.1 有效，启动更大卡数复现实验 | 确认不是小卡数偶然现象 |
| P1 | 如果 DMD 仍弱，做 `DMD weight` 增权短训 | 增强分布蒸馏信号 |
| P2 | 保留 LPIPS=2 主线，同时准备 LPIPS 降权 ablation | 只有在残影/模糊持续时再偏离论文 |

资源诉求：继续保留当前 40 卡主线和 16 卡短训对照资源。Stage3 已进入“训练目标调平衡”的阶段，需要靠多组短训和固定测试集评估来确定最终参数。
