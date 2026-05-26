# FlashVSR / LucidVSR 组会汇报（2026-05-11）

## 1. 本周主线

| 主线 | 目标 | 当前结论 |
|---|---|---|
| Stage2 跳变定位 | 找到 89 帧推理中周期性视觉跳变的来源 | dense full attention 不跳，sparse/chunk causal 路径都会跳 |
| Stage2 首帧问题 | 解释偶发首帧模糊、首帧不稳定 | 旧 v6 的 target 构造可能语义错位，已启动 v6.4 验证 |
| Stage3 方案 | 明确后续 distillation 怎么做 | 计划采用 DMD one-step distillation，并加入 pixel / LPIPS 约束 |

本周核心结论：Stage2 训练链路已经跑通，但 sparse-causal 推理在 chunk 边界上仍不够稳定；当前优先级是先把 Stage2 的对齐问题查清楚，再进入 Stage3。

## 2. Stage2 跳变 Probe

固定测试条件：

| 项 | 设置 |
|---|---|
| checkpoint | Stage2 v6 `step-10000` |
| 输入 | 89 帧 synthetic 退化测试集 |
| 推理尺寸 | `1280x768` |
| LQ 输入 | bicubic x4 后进入模型 |
| colorfix | 默认 AdaIN，另做 no-colorfix 对照 |
| 对比视频 | 重点看 `takano04` 逐帧结果 |

Probe 设计：

| Probe | 关键设置 | 想排查什么 | 结果 |
|---|---|---|---|
| A | v6.2 full-DiT mask，无 KV cache，`topk_ratio=2` | KV cache 是否是唯一原因 | 跳变 |
| B | A + `topk_ratio=4` | top-k 太小是否是主因 | 跳变 |
| C | `dense_full` | 去掉 sparse/chunk 后是否还跳 | 不跳变 |
| D | A + `topk_ratio=8` | 继续扩大 top-k 是否有效 | 跳变 |
| E | 1-step 推理 | 50-step diffusion 是否放大边界 | 只能辅助，画质不代表最终 |
| F | no colorfix / AdaIN offset / wavelet | colorfix 是否导致跳变 | 主结论不变 |
| G | dump LQ projector chunk stats | projector 数值是否爆掉 | 数值稳定 |
| H | 官方式 spatial local mask + chunk-grouped top-k | 诊断 v6.2 的 sparse block 选择是否未对齐官方推理 | 仍跳变 |
| v6.1 | streaming + KV cache | 接近作者推理路径 | 跳变 |
| v6.2 | full sequence + mask，无 KV cache | 更接近训练 mask | 跳变 |
| FlashVSR official | 官方 baseline | 官方是否完全不跳 | 也有轻微边界变化，但明显更小 |

Probe 结论：

| 观察 | 说明 |
|---|---|
| A/B/D/H/v6.1/v6.2 都跳，C 不跳 | 问题集中在 sparse/chunk causal 路径 |
| `topk_ratio=2/4/8` 都跳 | 不是简单扩大 top-k 就能解决 |
| no-cache 的 v6.2 也跳 | 不是 KV cache 单独导致 |
| G 的 projector stats 稳定 | LQ projector 没有明显数值爆炸 |
| FlashVSR official 也有轻微边界 | chunk 化本身有边界风险，但官方模型抑制得更好 |

逐帧观察到的跳变段：

| 段 | 现象 |
|---|---|
| `1-21` | 开头 3 个 chunk 对应区域，整体相对一致 |
| `22-29` | 第一个明显边界 |
| `30-37` | 下一段 chunk 区域 |
| `38-46` 及后续 | 基本按约 8 帧一个 chunk 继续跳 |

详细记录：

```text
doc/flashvsr_stage2_v6_jump_probe_20260509.md
```

## 3. 学习率对照 v6.3

| 实验 | 学习率 | 其他设置 | 目的 | 结果 |
|---|---:|---|---|---|
| v6 | `1e-5` | 89f, video-only, block sparse, worker2 | Stage2 主线 | 有 chunk 跳变 |
| v6.3 | `3e-6` | 与 v6 一致，只降低学习率 | 验证是否学习率过大 | 仍有跳变 |

结论：降低学习率没有根治跳变，问题更像是 sparse/chunk 结构与训练-推理对齐问题，而不是单纯优化过冲。

## 4. 首帧问题与 v6.4

旧 v6 的可疑点：

| 部分 | 旧 v6 |
|---|---|
| LQ projector | `89 -> 22` |
| noise latent | `22` |
| GT target | `89 -> Wan VAE -> 23 -> drop z0 -> 22` |
| 风险 | drop 掉 `z0` 后，原 `z1` 被放到首位，但 Wan decoder / DiT 仍可能把首位当作特殊首帧 latent |

v6.4 新假设：

| 部分 | v6.4 |
|---|---|
| LQ projector | `89 -> 22` |
| noise latent | `22` |
| GT target | `take first 85 frames -> Wan VAE -> 22` |
| 目的 | 保留首 latent 的语义，避免把非首帧 latent 放到首位 |

v6.4 与旧 v6 对比：

| 项 | v6 / v6.1 | v6.4 |
|---|---|---|
| target 构造 | `89 -> 23 -> drop z0 -> 22` | `85 -> 22` |
| 第一个 target latent | 原 `z1` 被放到首位 | 真实首 latent |
| 首帧 loss | 语义可能错位 | 保留 Wan 首帧特殊权重 |
| 当前目的 | 跑通 Stage2 sparse-causal | 验证首帧对齐新假设 |

v6.4 训练状态：

| 项 | 内容 |
|---|---|
| 训练代码 | `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py` |
| 启动脚本 | `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh` |
| 配置 | `stage2_release_48gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml` |
| 初始化 | Stage1 v5.3.5 `step-10000` |
| 节点 | 5 节点 40GPU，绕开 D-state 的 `wfnwbym4v6` |
| 训练目录 | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260511_40gpu_v64_lr1e5_r2` |
| 当前状态 | 已出 loss，已触发 `validation/step-10` |

已观察 loss：

| step | loss |
|---:|---:|
| 1 | 0.088556 |
| 2 | 0.064814 |
| 3 | 0.099703 |
| 4 | 0.435564 |
| 5 | 0.105875 |
| 8 | 0.068873 |
| 9 | 0.093882 |

## 5. Stage2 训练 / 推理机制

训练和推理的关系：

| 阶段 | 做法 | 目的 |
|---|---|---|
| 训练 | 一次输入完整 latent 序列，用 chunk-level sparse causal mask 控制可见范围 | 学会不看未来 chunk |
| 推理 | 按 chunk 流式推进，用 KV cache 保存历史 K/V | 避免每个新 chunk 重算历史 |

89 帧到 chunk 的关系：

| 概念 | 对应 |
|---|---|
| 输入帧 | `89` frames |
| Stage2 projector 输出 | `22` latent-time |
| chunk 大小 | `2` latent-time |
| 开头 full attention 区域 | 前 `6` latent-time，即前 `3` chunks |
| 后续区域 | 每个 chunk 约对应原始 `8` 帧 |

block sparse attention：

| 变量 | 含义 |
|---|---|
| block | `(2, 8, 8)`，即 2 个 latent-time、8x8 空间 token |
| `topk_ratio` | 训练和 H 统一为官方式 chunk-grouped top-k：按 temporal chunk 聚合所有 spatial query blocks 后选 block pairs |
| temporal local window (`local_num`) | 训练用；限制最多看最近多少个历史 chunk |
| spatial local mask (`local_range`) | 只用于 H / 官方高分辨率推理诊断，不进入当前 Stage2 训练 |
| chunk causal mask | 训练用；控制当前 chunk 不能看未来 chunk |

当前 kernel 对齐状态：

| 项 | 状态 |
|---|---|
| 官方 CUDA block-sparse kernel | 已使用 `/Block-Sparse-Attention` 的 `block_sparse_attn_func` |
| 当前代码行为 | `wan_video_dit_stage2_v6.py` 缺少 kernel 会直接报错，不会静默 fallback |
| 仍需核对 | 训练用 chunk causal / temporal `local_num` 是否与作者训练口径一致；top-k 已改为 chunk-grouped；`local_range/kv_ratio` 只作为推理侧核对 |

v6.1 与 FlashVSR 官方推理对齐状态：

| 项 | FlashVSR 官方逻辑 | 当前 v6.1 状态 | 后续处理 |
|---|---|---|---|
| 输入帧包装 | 目标 `8n-3`，tail repeat 4 帧，送入模型变 `8n+1` | 部分脚本依赖输入已准备好 | 需要在 v6.1 wrapper 中显式实现 |
| colorfix | 用生成帧和 `LQ_video[:, :, :frames.shape[2]]` 对齐修正 | 已有 colorfix | 固定为官方对齐方式 |
| KV cache | chunk 流式推进，历史 K/V 给后续 chunk 用 | 已有 streaming/cache 路径 | 继续核对 cache 生命周期 |
| 推理步数 | 官方 release 更接近 one-step / 少步数部署 | 当前 Stage2 仍按多步 diffusion 测 | Stage3 后再对齐 one-step |

## 6. 汇报时如何讲 A-H / v6.1 / v6.2 / FlashVSR

统一结果目录：

```text
/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFGH_clean_20260511
```

展示顺序：

| 顺序 | 看什么 | 路径 | 想说明什么 |
|---:|---|---|---|
| 1 | `takano04` 逐帧 | `frames_takano04/` | 最容易看到 chunk 边界 |
| 2 | C dense full | `frames_takano04/C_dense_full` | dense 不跳，是上界 |
| 3 | A/B/D/H | `frames_takano04/` | sparse/top-k 以及 H 的推理侧 spatial local mask 诊断后仍跳 |
| 4 | v6.1 / v6.2 | `frames_takano04/` | cache 和 no-cache 都跳 |
| 5 | FlashVSR official | `frames_takano04/flashvsr_official` | 官方也有轻微边界，但小很多 |
| 6 | G diagnostics | `diagnostics/` | projector 输出稳定 |
| 7 | v6.1 early-to-late ckpts | `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_inferv61_20260506` | 早期到后期都有边界问题，不只是训太久 |
| 8 | v6.2 early-to-late ckpts | `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_inferv62_20260506` | no-cache/full-mask 路径同样有边界问题 |

汇报结论：

| 结论 | 支撑 |
|---|---|
| 跳变不是普通画质问题 | C dense full 不跳 |
| 跳变不是 KV cache 单独导致 | v6.2 no-cache 也跳 |
| 跳变不是 top-k 太小单独导致 | `topk_ratio=2/4/8` 都跳 |
| projector 不是直接数值爆点 | G 的 chunk stats 稳定 |
| 官方路径仍值得继续对齐 | FlashVSR official 边界更弱 |

## 7. Stage3 计划

Stage3 目标：

| 目标 | 说明 |
|---|---|
| one-step distillation | 从 Stage2 sparse-causal model 蒸馏成更快的 one-step streaming VSR |
| 修正边界 | 用 pixel / LPIPS 直接约束连续帧外观 |
| 加强首帧 | 显式 pixel loss 可以补足 latent flow loss 对首帧不稳定的问题 |

论文对应 loss：

| loss | 作用 |
|---|---|
| `L_DMD` | one-step distribution matching |
| `L_FM` | 保持 diffusion / flow matching 约束 |
| pixel MSE | 约束像素级细节和连续性 |
| `2 * LPIPS` | 约束感知质量 |

模型角色：

| 角色 | 来源 | 用途 |
|---|---|---|
| `G_one` | Stage2 sparse-causal model | 最终 student |
| `G_real` | Stage1 full-attention teacher | real distribution direction |
| `G_fake` | Stage1 teacher copy | fake latent distribution |

风险判断：

| 风险 | 处理 |
|---|---|
| 如果 Stage2 teacher 跳变太强，Stage3 可能蒸馏进跳变 | 先看 v6.4 是否改善首帧和边界 |
| 如果 dense full 明显更稳 | 保留 C dense full 作为上界和 teacher 参考 |

Stage3 计划文档：

```text
doc/flashvsr_stage3_dmd_plan_20260511.md
```

## 8. 下一步

| 优先级 | 动作 | 目的 |
|---|---|---|
| P0 | 继续观察 v6.4 validation 和后续 checkpoint | 判断首帧对齐改法是否有效 |
| P0 | 用固定 89f 测试集补测 v6.4 | 与 v6 / v6.3 / dense probe 对比 |
| P1 | 核对训练侧 chunk causal / `local_num` / top-k block pair | 判断跳变是否来自训练 sparse 实现差异 |
| P1 | 单独核对推理侧 `local_range` / `kv_ratio` | 只用于 inference 对齐，不混入训练 |
| P1 | 准备 Stage3 DMD smoke | 验证 DMD + pixel + LPIPS 链路 |
| P2 | 保留 C dense full 作为上界 | 避免只在错误 sparse path 上盲训 |
