# FlashVSR Stage 2 问题整理：结构改进 / Causal / Streaming

日期：2026-04-27

## 背景

当前已经完成 Stage 1 相关训练实验，下一步准备实现论文中的 Stage 2：

> Block-Sparse Causal Attention Adaptation

为了避免把多个概念混在一起，先把 Stage 2 / streaming inference 相关内容拆成三层：

- 结构改进：把 Wan DiT 的 self-attention 从 dense full attention 换成 block-sparse attention；
- Causal：给 self-attention 加时间因果约束；
- Streaming：把输入和 DiT forward 改成分段形式，并引入 KV cache。

这三件事相关，但不完全等价。

## 1. Wan / Stage 1 当前结构理解

当前 Wan DiT 可以简化理解为：

```text
VAE latent
-> patch_embedding
-> tokens

重复 30 层 DiTBlock:
  1. self-attention
  2. cross-attention / text attention
  3. FFN / MLP

-> head
-> unpatchify
-> predicted noise / velocity
```

每个 `DiTBlock` 里大致是：

```text
x
-> norm + timestep modulation
-> self_attn(x)
-> residual add

-> norm
-> cross_attn(x, text_context)
-> residual add

-> norm + timestep modulation
-> FFN(x)
-> residual add
```

Stage 2 理论上主要改的是：

```text
self_attn(x)
```

不应该改：

```text
VAE
patch_embedding
cross-attention
FFN
head
```

`LR Proj-In` 需要改成 causal variant，但它属于条件注入支路，不是 DiT self-attention 本体。

## 2. 三个概念的拆分

### 2.1 结构改进：Dense Self-Attention -> Block-Sparse Self-Attention

Stage 1 / v5 里 self-attention 是 full attention：

```text
所有 token 看所有 token
```

Stage 2 目标是把它换成 block-sparse attention：

```text
把 token 切成 3D block
只在选中的 block pair 上做 attention
```

论文和官方推理代码中的 block size 是：

```text
(2, 8, 8)
```

也就是：

```text
2 个 latent-time
8 个 height token
8 个 width token
```

每个 block 展平成：

```text
2 * 8 * 8 = 128 tokens
```

这一层可以理解为“换 self-attention 的计算方式”。

### 2.2 Causal：给 Attention 加时间因果 Mask

Causal 本质上是 mask 规则：

```text
当前 latent-time 只能看当前和过去
不能看未来
```

注意：不能简单使用普通 sequence causal mask。

原因是 Wan token flatten 后顺序类似：

```text
(time, h, w)
```

如果直接用普通 `is_causal=True`，会让同一个 latent-time 内后面的空间 token 看不到前面的空间 token，这不符合视频 causal 的语义。

正确的 causal 应该是：

```text
同一 latent-time 内所有空间 token 可以互相看；
未来 latent-time 被屏蔽。
```

如果是 dense attention，需要 time-aware causal mask。

如果是 block-sparse attention，需要 block-level causal mask。

所以 causal 可以理解为：

```text
self-attention 上增加时间因果约束
```

### 2.3 Streaming：分段 Forward + KV Cache

Streaming 不是单纯的 causal，也不是单纯的 block-sparse。

Streaming 是推理/训练组织方式：

```text
把长视频分成 chunk
每次只处理当前 chunk
通过 KV cache 访问过去 chunk
```

FlashVSR 官方推理代码中包含以下 streaming 相关逻辑：

1. `LR Proj-In` 分段处理：

```text
4 帧 4 帧送入 causal LR projection
```

2. DiT 分段处理：

```text
首段 latent-time f = 6
后续 latent-time f = 2
```

3. DiT self-attention 使用 KV cache：

```text
当前 chunk 的 K/V 与历史 cache 拼接
当前 Q 通过 block-sparse attention 看当前和过去
```

4. 输出再拼回完整 latent 序列。

因此 streaming 可以理解为：

```text
projector chunking + DiT chunking + KV cache + overlap/buffer
```

## 3. 当前疑问：Stage 2 训练到底包含哪些东西

论文 Stage 2 的文字描述包括：

- causal masking；
- block-sparse attention；
- causal LR Proj-In；
- video-only flow matching training。

论文后面又讨论了 streaming inference 和 KV cache eviction。

因此目前存在一个实现上的疑问：

```text
Stage 2 训练时是否已经使用 streaming chunk + KV cache？
还是 Stage 2 只训练 block-sparse causal attention，
KV cache 主要是 Stage 3 / inference 阶段处理？
```

官方开源推理代码里，block-sparse self-attention 和 streaming cache 是耦合出现的。但这不一定等价于训练时也完全这么做。

## 4. 关于 89 帧和 `(2,8,8)` block 的问题

Wan VAE 的 temporal compression 规则是：

```text
latent_time = 1 + (num_frames - 1) / 4
```

所以：

```text
17 frames -> 5 latent-time
89 frames -> 23 latent-time
```

但是 block-sparse 的 temporal block size 是 2：

```text
(2,8,8)
```

如果整段 `23 latent-time` 直接进入 block partition，就不能被 2 整除。

官方推理代码中的处理方式不是全局 padding，而是 streaming chunk：

```text
first chunk: f = 6
later chunks: f = 2
```

这样每次进入 `(2,8,8)` block partition 的 temporal 维都是偶数。

当前需要确认：

- Stage 2 训练时，89 帧是否也按这种 chunk/cache 方式 forward？
- 还是训练时使用了另一套整段 sparse causal attention 实现？
- 如果训练是整段 23 latent-time，如何处理 `(2,8,8)` 的整除问题？

## 5. 需要向作者确认的问题

### Q1. Stage 2 训练是否使用 streaming chunk？

Stage 2 训练 forward 是否和官方推理一样，把 latent sequence 切成：

```text
first f=6
then f=2, f=2, ...
```

并在 self-attention 中使用 KV cache？

还是 Stage 2 训练是整段 latent-time 一次性 forward，只是换成 causal block-sparse mask？

### Q2. Stage 2 中 KV cache 是否参与训练？

KV cache 是不是只用于 inference / Stage 3？

还是 Stage 2 block-sparse causal attention 训练时也使用 KV cache？

如果训练时使用 KV cache：

- cache 是否参与梯度？
- cache 是否 detach？
- 每个 training clip 的 cache 如何初始化？

### Q3. 89 帧训练如何处理 `(2,8,8)` temporal block size？

89 帧经过 VAE 后是 23 latent-time。

如果 Stage 2 训练整段 23 latent-time：

- block partition 的 temporal size 2 如何整除？
- 是否 pad 到 24？
- pad token 是否参与 attention / loss？
- 是否裁回 23？

如果 Stage 2 训练使用 streaming chunk：

- 训练 chunk 组织是否和推理完全一致？
- first chunk 是否固定 `f=6`？
- later chunk 是否固定 `f=2`？

### Q4. 官方推理里的 overlap/buffer 在训练中如何处理？

官方推理看起来不是简单：

```text
0:6
6:8
8:10
```

而是存在 overlap / buffer，例如首段后续从靠近 `4:6` 的位置继续。

需要确认：

- Stage 2 训练是否复刻这个 overlap？
- overlap 部分的预测是否计算 loss？
- 如果同一个 latent-time 被多个 chunk 预测，loss 如何去重？
- 最终 training target 如何和 chunk 输出对齐？

### Q5. Block-sparse mask 是 causal mask 还是 top-k sparse mask 的组合？

论文里包含：

- causal mask；
- block-level coarse attention；
- top-k block selection；
- local attention window。

需要确认 Stage 2 训练中实际启用哪些：

- 是否一开始就启用 top-k block selection？
- `topk_ratio` / `kv_ratio` / `local_range` 的训练设置是什么？
- local window 是否和 inference 一致？

### Q6. `LR Proj-In` 的 causal variant 训练方式

官方推理中 `LR Proj-In` 是 4 帧 4 帧 streaming 处理，并带 causal cache。

需要确认 Stage 2 训练时：

- `LR Proj-In` 是否也按 4 帧 chunk 处理？
- cache 是否参与训练？
- Stage 1 训练得到的 `LR Proj-In` 如何迁移到 causal variant？
- 是否从 FlashVSR 官方 `LQ_proj_in.ckpt` 初始化？

## 6. 当前建议实现路线

为了避免多个变量一起变化，建议实现拆成：

### v6.0：Dense Time-Causal Baseline

目的：只验证 causal mask 语义和训练链路。

特点：

- 不用 flash-attn；
- 不用 block-sparse；
- 不适合正式大训练；
- 只作为小分辨率 smoke / 数值基线。

### v6.1：Block-Sparse Causal，暂不做 Streaming

目的：验证 block-sparse kernel 和 causal block mask。

需要根据作者回答决定：

- 是否允许整段 latent-time padding；
- 或者是否必须直接进入 chunk/cache 方案。

### v6.2：Official-Style Streaming Chunk + KV Cache

目的：对齐官方推理结构。

内容：

- `LR Proj-In` 4 帧 chunk；
- DiT 首段 `f=6`；
- 后续 `f=2`；
- 每层 self-attention 使用 KV cache；
- 明确 overlap 输出和 loss 对齐规则。

### v6.3：Top-k Block Selection + Local Window + Cache Eviction

目的：进一步对齐论文和官方推理性能优化。

内容：

- block-level average pooling；
- coarse attention map；
- top-k block pair；
- local attention window；
- KV cache eviction。

## 7. 当前判断

如果目标是严格复现论文阶段：

```text
Stage 2 = causal block-sparse self-attention + causal LR Proj-In
Stage 3 = streaming/KV cache adaptation
```

那训练实现应该先搞清楚 Stage 2 是否真的不需要 KV cache。

如果目标是最快训出能服务官方 FlashVSR streaming 推理的模型：

```text
直接做 official-style chunk + KV cache 更稳
```

因为官方推理代码已经把 block-sparse attention 和 streaming cache 绑定在一起使用。

当前最需要作者确认的是：

```text
Stage 2 训练 forward 到底是整段 causal block-sparse，
还是和 inference 一样的 streaming chunk + cache？
```

