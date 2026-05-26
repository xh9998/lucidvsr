# FlashVSR 组会汇报（2026-05-08）

## 1. 本周主线

本周工作从 Stage1 teacher 训练推进到 Stage2 sparse-causal adaptation。整体目标是：先得到质量尽可能稳定的 SR teacher，再把它改造成适合长视频流式推理的快速 VSR 模型。

当前主线：

| 方向 | 目标 | 当前状态 |
|---|---|---|
| Stage1 teacher | 训练 full-attention SR teacher，保证训练 / validation / inference 对齐 | `v5.3.5` 和 `v5.3.6` 已作为主 teacher |
| Stage2 v6 | 把 full-attention DiT 改成 sparse-causal DiT | 48GPU 正式训练已启动，已扫到 `step-8000` |
| 固定测试链路 | 用统一 synthetic / real 测试集追踪 ckpt 质量 | 89f Stage2 每 500 step 已完成一次测试扫描 |

## 2. Stage1：当前 teacher 版本

Stage1 的重点是先把 SR teacher 训练对，不提前引入 streaming / causal 逻辑。前期反复定位后，最终确定 Stage1 必须使用 `nonstreaming_aligned` projector，让 LQ latent 和 GT latent 在时间维严格对齐。

```text
Stage1 规则:
  17 raw frames -> 5 latent-time
  89 raw frames -> 23 latent-time
  LQ projector 和 GT / DiT latent-time 完全对齐
```

### 2.1 当前两个主 teacher

| 版本 | 帧数 | 训练规模 | 数据 | projector | 作用 |
|---|---:|---:|---|---|---|
| `v5.3.6` | 17f | 16GPU | Takano + Yubari + image pseudo-video | random init, nonstreaming aligned | 17f teacher，短视频测试和对照 |
| `v5.3.5` | 89f | 48GPU | Takano + Yubari + image pseudo-video | random init, nonstreaming aligned | 89f teacher，Stage2 初始化来源 |

实验目录：

```text
v5.3.6:
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200

v5.3.5:
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300
```

### 2.2 为什么 Stage1 不能提前 streaming

前期发现，如果 Stage1 projector 用 streaming 逻辑，会出现 LQ latent 比 GT latent 少一段的问题。以 89 帧为例：

```text
GT / WAN VAE:
  89 -> 23 latent-time

旧 streaming projector:
  首帧只做 warm-up，不输出
  89 -> 22 latent-time

结果:
  LQ 22 和 GT 23 不对齐
```

所以 Stage1 当前策略是：先训练完整对齐的 full-attention teacher，把 streaming 留到 Stage2。

## 3. Stage2 v6：改了什么

Stage2 不是重新训练 WAN，而是把 Stage1 teacher 继续适配成 sparse-causal DiT。主要改动集中在三块：数据、LQ projector、DiT self-attention。

| 模块 | Stage1 | Stage2 v6 | 改动目的 |
|---|---|---|---|
| 数据 | video + image joint | video only | 对齐论文 Stage2，只做视频 causal adaptation |
| LQ projector | nonstreaming aligned | streaming causal | 让 projector 具备流式输入能力 |
| DiT self-attention | full attention | chunk causal + block sparse | 降低长视频 attention 成本 |
| loss target | 完整 latent-time | 丢掉首帧 latent 后训练 | 对齐 Stage2 projector 的 89->22 |
| inference | 整段推理 | streaming + KV cache | 部署时避免重复计算 |

保持不变：

```text
WAN base DiT 权重不变
VAE 不变
cross-attention 不变
FFN 不变
训练参数主要是 DiT LoRA + LR Proj-In
```

## 4. 89 帧为什么变成 22

作者 Stage2 的关键设定是：首帧不作为要预测的 latent 输出，而是作为 causal projector 和后续流式推理的上下文。

```text
输入 LQ:
  f0, f1, f2, ..., f88

projector warm-up:
  [f0, f0, f0, f0] -> 只建立 cache，不输出 latent

后续每 4 帧输出 1 个 latent-time:
  [f1,  f2,  f3,  f4 ] -> z1
  [f5,  f6,  f7,  f8 ] -> z2
  ...
  [f85, f86, f87, f88] -> z22

Stage2 LQ projector:
  89 raw frames -> 22 latent-time
```

GT 侧仍然按 WAN VAE 正常编码：

```text
WAN VAE:
  89 raw frames -> 23 latent-time

Stage2 loss:
  drop first latent
  23 -> 22
```

最后训练对齐关系是：

```text
LQ projector: 89 -> 22
GT target:    89 -> 23 -> drop first -> 22
DiT input:    22
loss:         22
```

这个规则是 Stage2 和 Stage1 最大的不同。

## 5. chunk、mask、block sparse 和 top-k

### 5.1 chunk 是什么

Stage2 的 causal 不是 token 级下三角，而是 chunk 级因果。当前对齐作者口径：

```text
1 chunk = 2 latent-time
1 latent-time 约对应原始视频 4 帧
所以 1 chunk 约对应原始视频 8 帧

89f Stage2:
  22 latent-time -> 11 chunks
```

### 5.2 为什么前 6 个 latent-time full attention

官方逻辑里，开头 6 个 latent-time 等于前 3 个 chunks。这 3 个 chunks 之间使用 full attention。

原因：

| 设计 | 作用 |
|---|---|
| 开头 3 个 chunks full attention | 给模型足够的起始上下文，避免刚开始就只能看很短历史 |
| 后续 chunk causal | 保证模型不能看未来，适配流式推理 |
| local window | 控制最多回看多少历史，避免每个 chunk 都看完整过去 |

文字示意：

```text
chunk index:
  C0 C1 C2 C3 C4 C5 C6 ...

attention visibility:
  C0 -> C0 C1 C2
  C1 -> C0 C1 C2
  C2 -> C0 C1 C2
  C3 -> C0 C1 C2 C3
  C4 -> C1 C2 C3 C4       local window 开始滑动
  C5 -> C2 C3 C4 C5
```

对应的 mask 图：

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/flashvsr_stage2_chunk_causal_mask.svg
```

### 5.3 block sparse 是什么

chunk mask 决定“哪些时间块可以互相看”。在允许看的 chunk pair 内，再做 `(2,8,8)` 的 block sparse attention。

```text
block size:
  time  = 2 latent-time
  h     = 8 latent cells
  w     = 8 latent cells

tokens per block:
  2 * 8 * 8 = 128
```

如果不加 sparse attention，attention 成本仍接近 full attention，只是加了 causal mask。加了 block sparse 后，模型先用 block-level 特征估计哪些区域最相关，再只在选中的 block pair 上做真实 token attention。

### 5.4 top-k 是什么

top-k 是 block sparse 里的“候选块选择”。

流程：

```text
Q/K token block
  -> 每个 block 内 average pooling
  -> 得到 block-level Q/K
  -> 算粗 attention map
  -> 在 causal mask 允许范围内选 top-k block pairs
  -> 只对这些 block pairs 做 128 x 128 token attention
```

当前 v6 设置：

```text
topk_ratio = 2.0
```

这个值按官方推理默认配置先固定进入训练。后续如果作者给出训练专用 top-k，可以替换。

## 6. 训练用 mask，推理用 cache

这部分是本周和作者对齐后最重要的结论。

| 阶段 | 实现方式 | 目的 |
|---|---|---|
| Stage2 训练 | 完整 22 latent-time 一次送入 DiT，用 chunk-level causal mask 限制可见范围 | 学会 causal 约束，不看未来 |
| Stage2 推理 | 按 chunk 流式送入，保存每层 K/V cache | 避免每个新 chunk 重算历史 |

训练不需要真的逐 chunk 跑 cache，因为训练要的是“因果约束下的学习信号”。推理才需要 cache，因为推理要的是速度。

### 6.1 推理 cache 怎么对应流式

v6.1 推理采用官方风格的 `6 + 2 + 2 + ...` chunk 处理：

```text
Stage2 latent stream:
  22 latent-time

推理分段:
  first call: 6 latent-time = C0 C1 C2
  next calls: 2 latent-time = C3, then C4, then C5 ...
```

每一层 DiT block 都维护自己的 cache：

```text
for each DiT self-attention layer:
  cache_k[layer] 保存过去可见 block 的 K
  cache_v[layer] 保存过去可见 block 的 V

当前 chunk:
  Q 来自当前 chunk
  K/V 来自 cache + 当前 chunk
```

这样推理时不需要重新处理过去所有 chunks。

### 6.2 作者是单步，我们现在是 50 步

当前 v6.1 inference 仍然是 diffusion 多步推理：

```text
num_inference_steps = 50
```

cache 的使用范围是“每一个 denoising step 内部的 chunk 流式推理”。也就是说，每个 diffusion timestep 会重新清空 projector cache 和 DiT K/V cache，然后按 `6+2+...` 跑完整个视频。

作者最终系统还有 Stage3 one-step distillation。Stage3 之后，模型只需要 1 个 denoising step，cache 的收益会更直接。

当前关系：

```text
Stage2 v6:
  50 denoising steps
  each step uses streaming/cache over chunks

Stage3 target:
  1 denoising step
  streaming/cache cost becomes much lower
```

### 6.3 官方流式推理的帧数规则

Stage2 的核心设定是让 DiT 预测首帧之后的 latent stream。以 89 帧为例，projector 输出 22 个 latent-time，GT 侧也通过 `drop first latent` 对齐到 22 个 latent-time。

官方 FlashVSR 推理还有一个容易混淆的输入长度规则：它不是把“任意 N 帧”直接送进模型，而是先把真实视频整理成适合 streaming chunk 的长度。

可以近似理解成：

```text
真实想处理 N 帧
先在末尾 repeat 最后一帧 4 次
得到 N + 4 帧
再取 <= N + 4 的最大 8n + 1 作为模型输入 F

模型实际输入帧数 = F
有效输出帧数 = F - 4
```

所以官方最自然的真实目标长度是 `8n-3`。因为：

```text
真实目标: 8n - 3
末尾补 4: 8n + 1
模型输入: 8n + 1
有效输出: 8n - 3
```

典型例子：

```text
真实想超分:
  85 frames

官方输入准备:
  85 real frames + 4 repeated tail frames = 89 model input frames
  89 = 8 * 11 + 1

pipeline 内部:
  process_total_num = (89 - 1) / 8 - 2 = 9

最终有效输出:
  85 frames
```

这里的 `85` 才是用户真正关心的真实视频长度；`89` 是为了让流式窗口闭合而构造出来的模型内部输入长度。末尾多出来的 4 帧不是新内容，而是最后一帧的 padding。

更多例子：

| 原始视频帧数 `N` | `N + 4` | 模型输入 `F = max 8n+1 <= N+4` | 有效输出 `F-4` |
|---:|---:|---:|---:|
| 17 | 21 | 17 | 13 |
| 21 | 25 | 25 | 21 |
| 33 | 37 | 33 | 29 |
| 37 | 41 | 41 | 37 |
| 81 | 85 | 81 | 77 |
| 85 | 89 | 89 | 85 |
| 89 | 93 | 89 | 85 |
| 93 | 97 | 97 | 93 |

这仍然是超分模型，因为对用户有效的输入输出仍然是一一对应的真实帧。repeat 出来的 4 帧只是内部 padding，类似图像模型先 pad 到 64 倍数，最后再 crop 回有效区域。最终对外应该表现为：

```text
用户给 N 帧真实 LQ
wrapper 内部自动整理成 8n+1 输入
模型流式推理
最终保存 N' 帧有效 SR
```

其中 `N' = F - 4`。如果要严格输入多少真实帧就输出多少真实帧，需要 wrapper 进一步做分段 / 拼接 / 边界处理；官方当前脚本更偏向固定处理满足 `8n-3` 的目标段。

示意图：

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/flashvsr_official_streaming_frame_rule.svg
```

### 6.4 当前 v6.1 推理和官方逻辑是否完全一样

当前 v6.1 推理已经对齐了核心 streaming/cache 思路：

```text
每个 denoising timestep:
  clear projector cache
  clear DiT K/V cache
  按 6 + 2 + 2 + ... chunk 流式处理
  当前 timestep 内 cache 只保存当前 timestep 的历史 K/V
```

这和训练的对齐关系是：

```text
训练:
  随机采一个 timestep t
  整段 latent 一次进入 DiT
  用 chunk causal mask 模拟流式可见范围

推理:
  对每个 timestep t
  真实按 chunk 流式跑
  cache 里保存的也是同一个 timestep t 的历史 chunk
```

所以 cache 不会跨 denoising timestep 混用。每个 timestep 的 cache 都是独立的一轮流式推理。

但当前 v6.1 外部测试还没有完整复刻官方输入准备阶段的 tail repeat 逻辑。当前脚本假设输入已经是合适的 `8n+1` 长度，例如 89 帧，然后直接进入 streaming inference。因此，如果 SR 输出帧数和 LQ 输入帧数不同，测试脚本会按较短长度做 color fix 和保存，可能看到类似：

```text
sr=85, lq=89
```

这不是 Stage2 训练目标错了，而是当前 v6.1 inference wrapper 还缺少 official 的“原始视频先取 `8n-3`，末尾 repeat 4，送入 `8n+1`”这层完整包装。当前测试主要用于观察 checkpoint 趋势；如果要做最终 demo，需要把 official 的 tail-repeat 输入准备也接进 v6.1 inference。

## 7. Stage2 v6 实验设置

当前正式训练实验：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2
```

初始化来源：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

| setting | 当前值 | 目的 |
|---|---|---|
| `num_frames` | 89 | 对齐长视频 Stage2 |
| `dataset` | video only | Stage2 不再混图像 |
| `Takano:Yubari` | 1:1 | 保持两个视频源都参与 |
| `batch_size` | 1 | 89f + block sparse 显存较高 |
| `dataset_num_workers` | 2 | 提升 GPU 利用率 |
| `attention` | block sparse chunk causal | 对齐 Stage2 结构 |
| `topk_ratio` | 2.0 | 官方默认 sparse selection |
| `trainable` | LoRA + LR Proj-In | 适配 sparse-causal，不全量训 WAN |

启动结果：

| 检查项 | 结果 |
|---|---|
| 2GPU smoke | 正常出 loss |
| 48GPU 正式训练 | 正常出 loss |
| GPU 利用率 | 多数时间高位，48 卡可到 `99-100%` |
| 显存 | 较高但可运行 |

## 8. v6 测试怎么做

当前测试脚本：

```text
wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_step500_incremental_on_my_20260508.sh
```

推理入口：

```text
wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py
```

测试设置：

| 项目 | 当前值 | 说明 |
|---|---|---|
| 输入 | 89f LQ，四分之一尺寸 | 与 VSR 任务一致 |
| 预处理 | bicubic x4 | 对齐模型输入尺寸 |
| inference | streaming/cache | 对齐 Stage2 推理 |
| attention | `block_sparse_chunk_causal` | 与训练 attention 一致 |
| top-k | `topk_ratio=2.0` | 与训练一致 |
| KV cache | `kv_ratio=3.0` | 推理缓存配置 |
| steps | 50 | 当前还不是 Stage3 one-step |
| color fix | `adain` | 减少推理色偏 |

这次测试做了增量逻辑：已经测过的 step 不重复测，只补新 checkpoint。

已完成扫描：

```text
step-500, 1000, 1500, 2000,
step-2500, 3000, 3500, 4000,
step-4500, 5000, 5500, 6000,
step-6500, 7000, 7500, 8000
```

每个 step 输出 10 个视频。

本地结果目录：

```text
/Users/lixiaohui/Desktop/stage2_v6_worker2_scan89_v61_20260506_8way
```

## 9. 当前结论

1. Stage1 teacher 的正确版本已经收敛，当前主要使用 `v5.3.5` 作为 89f Stage2 初始化。
2. Stage2 的训练思想已经理清：训练用完整序列 + chunk causal mask，推理用 streaming + KV cache。
3. v6 已经把 video-only、89->22、chunk mask、block sparse、top-k、LoRA + projector training 串成正式训练链路。
4. 当前最需要看的不是“代码能不能跑”，而是 step 扫描结果中哪一段质量最好，以及 Stage2 是否真正带来可接受的 sparse-causal 适配。

## 10. 下一步

| 下一步 | 目的 |
|---|---|
| 人工查看 `step-500` 到 `step-8000` 的固定测试结果 | 判断 Stage2 最佳 checkpoint 区间 |
| 对比 Stage1 teacher / FlashVSR official / SeedVR | 明确质量、稳定性和速度关系 |
| 如果后期质量下降，结合 loss 与测试结果回看 | 判断是否需要调学习率、top-k 或 local window |
| Stage2 稳定后进入 Stage3 | 做 one-step distillation，把 50 步推理压到 1 步 |
