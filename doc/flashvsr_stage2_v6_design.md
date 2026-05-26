# FlashVSR Stage2 / v6 唯一执行版设计

日期：2026-04-29

本文是当前 `Stage2 / v6` 的唯一主文档，只保留当前认为正确的理解、v6 具体执行方案、以及仍需向作者确认的问题。早期关于 dense causal、token 级 causal、`kv_ratio/local_range` 进入训练等推导不再作为执行依据。

## 1. 三阶段当前理解

### 1.1 Stage1：非流式 SR teacher

Stage1 的目标是先训练一个全注意力 SR teacher，不做 streaming causal 约束。

当前 `v5.3.5 / v5.3.6` 的正确规则是：

- DiT / GT 侧按 WAN VAE 标准时间压缩：
  - `latent_time = 1 + (raw_frames - 1) / 4`
  - `17f -> 5`
  - `89f -> 23`
  - `5f image pseudo-video -> 2`
- LQ projector 使用 `lq_proj_temporal_mode=nonstreaming_aligned`：
  - 首帧 `f0` 复制 3 份；
  - 拼到原始视频前面；
  - 整段过 3D conv projector；
  - 不丢掉 warm-up 输出。
- 因此 Stage1 LQ projector 与 GT / DiT latent-time 对齐：
  - `17f LQ -> 5`
  - `89f LQ -> 23`
  - `5f image pseudo-video LQ -> 2`

当前正在跑的 Stage1 修正版实验：

- `17f / 16GPU / v5.3.6`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200`
- `89f / 48GPU / v5.3.5`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_175800`

这两个实验是 Stage1 teacher 线，不是 Stage2。

### 1.2 Stage2：Sparse-Causal DiT adaptation

Stage2 从 Stage1 teacher 继续，把 full-attention DiT 适配为 sparse-causal DiT。

Stage2 的核心变化：

- 数据切为 video only。
- LR Proj-In 改为 causal streaming variant。
- DiT self-attention 改为 chunk-level causal + `(2,8,8)` block-sparse attention。
- 训练仍使用 flow matching loss。
- `topk` 进入训练，并且训练全程固定；block pair selection 使用官方式 chunk-grouped top-k，不使用 spatial local mask。
- `kv_ratio`、`local_range`、KV cache 属于推理技巧，不进入 Stage2 第一版训练。

### 1.3 Stage3：one-step distillation

Stage3 是后续 one-step sparse-causal student 蒸馏，不是当前 v6 第一阶段实现目标。

当前只记录边界：

- 从 Stage2 sparse-causal DiT 继续。
- 做 one-step student。
- 损失包含 DMD / FM / pixel MSE / LPIPS。
- TC decoder 和 KV cache 更偏推理加速与部署。

## 2. Stage2 时间维和 LR projector

### 2.1 Stage2 为什么是 `89f -> 22`

Stage2 是 streaming adaptation，不再让 DiT 直接预测首帧 latent。

以 89 帧为例：

```text
输入 LQ:
  f0, f1, f2, ..., f88

warm-up/cache:
  [f0, f0, f0, f0]
  只建立 causal conv cache，不输出训练 latent

后续每 4 帧一组:
  [f1, f2, f3, f4]      -> 第 1 个 LQ latent-time
  [f5, f6, f7, f8]      -> 第 2 个 LQ latent-time
  ...
  [f85, f86, f87, f88]  -> 第 22 个 LQ latent-time

最终:
  89 raw frames -> 22 LQ latent-time
```

这里“前四帧给后四帧 cache”的含义是：

- 每个 4 帧 group 经过 causal projector 后都会更新 cache；
- 下一组 4 帧会带着上一组留下的 cache 继续计算；
- 这个行为应对齐 FlashVSR 官方 `Causal_LQ4x_Proj.stream_forward()`。

### 2.2 GT / noise / loss 对齐

WAN VAE 对 GT 正常编码会得到：

```text
89f GT -> 23 latent-time
```

v6 Stage2 第一版采用：

- GT 正常过 WAN VAE 得到 `23 latent-time`。
- 丢掉首帧对应的第 0 个 GT latent。
- 只保留后面 `22 latent-time` 作为训练 target。
- noise / timestep / model output / flow matching loss 全部只在这 22 个 latent-time 上计算。

因此 Stage2 训练的是：

```text
LQ projector: 89f -> 22
GT target:    23 -> drop first -> 22
DiT input:    22 latent-time
loss:         22 latent-time
```

这个是当前最关键的对齐规则。

## 3. Stage2 causal 与 block sparse

### 3.1 chunk 是 causal 单位

Stage2 causal 不是 token 级，而是 chunk 级。FlashVSR 官方代码里的
`generate_causal_block_mask()` 也是先在 `seqlen=f//2` 的 chunk 轴上构造
mask，再展开到 block-sparse kernel 使用的 block 矩阵。

定义：

```text
1 chunk = 2 latent-time
```

对于 `89f -> 22 latent-time`：

```text
22 latent-time -> 11 chunks
```

开头：

```text
first 6 latent-time = C0 + C1 + C2 = 3 chunks
```

这 3 个 chunk 之间是 full attention。

官方 mask 的基础形式是：

```python
causal_mask = (j <= i) & (j >= i - local_num + 1)
causal_mask[0, 1] = True
causal_mask[:2, 2] = True
```

含义：

- `i` 是 query chunk；
- `j` 是 key/value chunk；
- `(j <= i)` 表示 chunk 之间 causal；
- `(j >= i - local_num + 1)` 表示最多看最近 `local_num` 个历史 chunk；
- 当前 v6 第一版令 `stage2_local_num=-1`，表示复用官方代码里的随机 `local_num` 采样；
- 官方随机逻辑会在 `seqlen-3 / seqlen-4 / seqlen-2 / seqlen` 中采样；
- `causal_mask[0, 1] = True` 和 `causal_mask[:2, 2] = True` 把前三个 chunk 补成 full attention start window。

后续：

- 每个 `2 latent-time` 是一个 chunk。
- chunk 内 full attention。
- chunk 之间 causal，只能看当前和过去 chunk。
- 不再额外加入 tail drop。末尾是否看不到最早 chunk 由官方 `local_num` 滑窗自然决定。

mask 图：

- `doc/flashvsr_stage2_chunk_causal_mask.svg`

### 3.2 `(2,8,8)` 是 block-sparse 单位

在 chunk mask 允许的 chunk pair 内，再做 block-sparse。

block 定义：

```text
time block   = 2 latent-time
height block = 8 latent spatial cells
width block  = 8 latent spatial cells

tokens per block = 2 * 8 * 8 = 128
```

层级关系：

```text
raw video
  -> LQ projector / WAN VAE latent-time
  -> chunk-level causal mask
  -> allowed chunk pair 内切 (2,8,8) block
  -> block-level Q/K average pooling
  -> coarse block attention map
  -> fixed top-k block pair selection
  -> selected 128 x 128 token attention
```

### 3.3 top-k 进入训练

Stage2 训练需要 top-k block sparse attention。

当前执行规则：

- `topk` 训练全程固定。
- 不随机变化。
- 第一版先对齐官方推理默认 `topk_ratio=2.0`。
- 如果作者给出训练专用 top-k，再替换。

### 3.4 不进入 Stage2 第一版训练的参数

以下只作为推理技巧，不进入 v6 第一版训练：

- `kv_ratio=3.0`
  - 推理 KV cache 保留历史长度相关。
- `local_range=9`
  - 推理 local attention / locality constraint 相关。
- KV cache
  - 推理 streaming continuity 机制。
  - 训练第一版使用完整序列 + chunk-level causal mask，不显式跑推理 cache 循环。

## 4. v6 模型参数与初始化

### 4.1 替换的是 self-attention 算法，不是新主干

v6 只替换 DiT block 里的 self-attention 计算方式：

- `q/k/v/o` 线性层 shape 不变；
- cross-attention 不改；
- FFN 不改；
- norm 不改；
- embedding/head 不改；
- block-sparse attention 本身不新增可学习参数。

因此：

- WAN 预训练权重可以正常导入。
- Stage1 LoRA 可以正常导入。
- Stage1 projector 同名权重可以作为 causal projector 初始化。

但注意：

- full attention 变成 sparse-causal attention 后，连接方式变了；
- 即使参数 shape 能导入，也需要继续训练适配；
- 不能认为换 attention 后完全不用训练。

### 4.2 v6 第一版训练哪些参数

建议第一版 v6 不全量打开 WAN 主干，继续用 LoRA 适配。

训练：

- DiT LoRA；
- `LR Proj-In`。

冻结：

- base WAN DiT 原始权重；
- VAE；
- text/prompt 相关部分。

不训练：

- block-sparse attention 算子本身；
- `kv_ratio/local_range`；
- image branch。

原因：

- Stage2 是结构适配，不是重新训练 WAN。
- 原始 `q/k/v/o` 权重能复用。
- LoRA 用来适配 full attention 到 sparse-causal attention 的分布变化。
- projector 行为从 Stage1 nonstreaming aligned 切到 Stage2 causal streaming，必须参与训练。

## 5. v6 具体执行方案

### 5.1 新文件

不要在当前 v5.3.5 / v5.3.6 训练文件上继续叠 Stage2。

已新增：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py`
- `diffsynth/models/wan_video_dit_stage2_v6.py`
- `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6.py`
- `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_batch.py`
- `wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_blocksparse.yaml`
- `wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-BlockSparse.sh`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse.sh`

### 5.1.1 当前代码对应关系

- `FlashVSRStage2VideoOnlyDataset`
  - 只读视频源。
  - 支持 `internal_url` 单源 smoke。
  - 支持 `yubari_video_tar_url + takano_video_tar_url` 双源训练。
- `WanVideoUnit_NoiseInitializerStage2`
  - 直接生成 Stage2 需要的 latent 长度。
  - 对 89 帧生成 22 个 latent-time 的 noise。
- `WanVideoUnit_InputVideoEmbedderStage2`
  - GT 正常过 VAE。
  - 然后执行 `input_latents = input_latents[:, :, 1:]`。
  - 对 89 帧即 `23 -> 22`。
- `FlashVSRStage2ValidationCallback`
  - 固定抽 video-only validation 样本。
  - 保存 `hr.mp4 / lq.mp4 / sr.mp4 / meta.json`。
  - `meta.json` 记录输入帧数和输出帧数，方便继续核对 Stage2 streaming 长度规则。
- `FlashVSRLQProjIn`
  - Stage2 pipeline 强制使用 `temporal_mode="streaming"`。
  - 对 89 帧输出 22 个 LQ latent-time。
- `block_sparse_chunk_causal_attention`
  - 使用 `(2,8,8)` 分块。
  - 先按官方 `generate_causal_block_mask()` 语义构建 chunk-level causal mask：
    - causal + `local_num` 历史窗口；
    - 前三个 chunk 通过硬编码补成 full attention；
    - 当前第一版 `stage2_local_num=-1`，即按官方随机 local window 逻辑采样。
- 再在合法 chunk pair 里做官方式 chunk-grouped fixed top-k block selection。
  - 最后调用 `block_sparse_attn_func`。
- `enable_stage2_causal_attention`
  - 只 patch 当前 `WanModel` 实例的 `self_attn.forward`。
  - 不改变 state dict。
  - 因此 Stage1 LoRA / projector checkpoint 可导入。

### 5.2 实现顺序

1. 复制当前 v5.3.5/v5.3.6 的稳定训练骨架。
2. 删除 image branch，只保留 video 数据。
3. 实现 Stage2 causal streaming `LR Proj-In`：
   - `f0` 复制 3 次 warm-up；
   - warm-up 不输出；
   - 后面每 4 帧输出 1 个 latent；
   - `89f -> 22`。
4. GT / noise / loss 统一裁到后 22 latent-time。
5. DiT self-attention 替换为官方 `(2,8,8)` block sparse 路径。
6. 实现 chunk-level causal mask。
7. 在允许的 chunk pair 内做官方式 chunk-grouped fixed top-k block sparse。
8. 2 卡 smoke：
   - 先确认 shape；
   - 再确认 loss；
   - 再确认显存。
9. 正式多机训练。

### 5.3 不建议的路线

不建议先做这些：

- 不建议再做 token 级 dense causal baseline。
- 不建议把 `kv_ratio/local_range` 混入训练。
- 不建议为了 `(2,8,8)` 强行在数据侧 padding。
- 不建议把 Stage2 直接塞进 v5.3.5/v5.3.6 文件。

## 6. 仍需确认的问题

这些不是写 v6 骨架的阻塞项，但会影响最终是否完全作者对齐。

1. Stage2 loss 是否确实只覆盖裁掉首帧后的 `22 latent-time`。

当前执行假设：

```text
GT 23 -> drop first -> train 22
```

需要作者确认是否完全一致。

2. `local_num` 的训练采样是否完全使用官方随机分布。

当前执行假设：

- `stage2_local_num=-1`；
- 按官方代码在 `seqlen-3 / seqlen-4 / seqlen-2 / seqlen` 中随机采样；
- 不额外做 tail drop。

需要确认训练是否也使用这套随机 local window。

3. top-k 数值。

当前执行假设：

- `topk_ratio=2.0`
- 训练全程固定。

需要确认训练是否使用同样数值。

4. 被选中的 `(2,8,8)` block pair 内是否还需要额外 token mask。

当前理解：

- causal 已经在 chunk 级完成；
- block 的 time size 正好等于 chunk time size；
- 因此 block 内不再需要 token-level 下三角。

需要作者确认。

## 7. 当前结论

v6 第一版代码已经写出，但还需要远端 smoke 后才能作为正式训练入口。

第一版目标不是直接完成最终推理系统，而是先把 Stage2 训练链路打通：

```text
video only
89f -> 22 latent-time
causal streaming LR Proj-In
GT/noise/loss 对齐到 22
chunk-level causal mask
(2,8,8) block sparse
fixed top-k
LoRA + projector training
```

如果这版 smoke 能稳定出 loss，再继续处理长视频推理的 KV cache / local range / TC decoder。

## 8. 2026-05-06：89f / worker2 / 48GPU 正式启动记录

本次目标是从当前较稳定的一阶段 89f `v5.3.5` checkpoint 进入 Stage2 训练，并优先保证 GPU 利用率。

### 8.1 继承的一阶段 checkpoint

使用的一阶段权重：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

该 checkpoint 同时作为：

- `lq_proj_in` 初始化；
- DiT LoRA 初始化。

启动前已确认六个节点本地相同路径都存在该文件。非主节点缺文件时，从：

```text
s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

拉取到本地对应路径。

### 8.2 新增 worker2 配置

新增 2GPU smoke：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_89f_videoonly_blocksparse_worker2.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-89f-VideoOnly-BlockSparse-Worker2.sh
```

新增 48GPU 正式训练：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh
```

关键配置：

- `dataset_mode=stage2_video_only`
- `num_frames=89`
- `height=768`
- `width=1280`
- `batch_size=1`
- `dataset_num_workers=2`
- `dataloader_prefetch_factor=1`
- `dataloader_persistent_workers=true`
- `dataloader_in_order=false`
- `stage2_attention_mode=block_sparse_chunk_causal`
- `stage2_topk_ratio=2.0`
- `stage2_local_num=-1`
- `yubari_video_prob=0.5`
- `takano_video_prob=0.5`
- `validation_num_samples=0`
- `use_gradient_checkpointing=true`
- `use_gradient_checkpointing_offload=false`

### 8.3 2GPU smoke 结果

smoke 机器：`b8gkuie2ns`，只使用 GPU `0,1`，其余 GPU `2-7` 保持占卡。

smoke 输出目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_smoke_2gpu_v6_lora_89f_videoonly_blocksparse_worker2_20260506_smoke_b8_worker2
```

结果：

- `lq_proj_in` 加载成功：`keys=8, missing=0, unexpected=0`
- LoRA 加载成功：`keys=480, missing=825, unexpected=0`
- 第一条 loss：`step=1 loss=0.038978`
- smoke 时 GPU `0,1` 利用率达到 `100%`

结论：89f / video-only / worker2 / block-sparse chunk causal / stage1 checkpoint 热启动路径可正常出 loss。

### 8.4 48GPU 正式训练

六节点分配：

```text
rank0 b8gkuie2ns
rank1 wfnwbym4v6
rank2 kh5idf7f98
rank3 hj65iqg9rh
rank4 zhki5rrddw
rank5 xwk6qjuej5
```

启动参数：

```text
MASTER_ADDR=240.12.149.199
MASTER_PORT=29606
RUN_TS_OVERRIDE=20260506_014800_stage2_v6_48gpu_worker2
```

正式实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2
```

启动结果：

- wandb 已开启；
- 第一条 loss：`step=1 loss=0.115593`
- b8 单机显存约 `90GB/GPU`
- 六个节点 48 张 GPU 利用率均约 `99-100%`

当前结论：worker2 版本的数据加载没有阻塞 48 卡 Stage2 训练，GPU 利用率达到目标。

## 9. 2026-05-10：v6.3 低学习率对照实验

### 9.1 为什么做 v6.3

Stage2 `v6` 在 `lr=1e-5` 下已经能训练，但 sparse/chunk 推理结果存在明显 chunk 边界跳变。由于 dense full-attention 对照 `probeC` 不跳，而 sparse/chunk 相关 probe 普遍跳，当前问题更像是 sparse/chunk 训练和推理路径的稳定性问题。

论文 `2510.12747v1.pdf` 的 Training Details 写的是三阶段统一使用 AdamW，`learning rate=1e-5`，没有明确要求 Stage2 降低学习率。因此 `v6.3` 不是作者复现配置，而是一个稳定性对照：保持 Stage2 结构不变，只把学习率从 `1e-5` 降到 `3e-6`，观察 loss 和边界跳变是否更稳定。

### 9.2 v6.3 与 v6.1 的关系

`v6.3` 复用 `v6.1` 训练入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_1_lora.py
```

原因是 `v6.1` 已经补上 Stage2 validation，validation 使用 streaming / KV-cache 推理路径，更接近作者推理方式。`v6.3` 只新增一套独立 config 和 sh：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_3_lora_89f_videoonly_bs1_lr3e6_blocksparse_worker2_val.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-3-Lora-89f-VideoOnly-bs1-lr3e6-BlockSparse-Worker2-Val.sh
```

### 9.3 v6.3 关键配置

```text
dataset_mode: stage2_video_only
num_frames: 89
height: 768
width: 1280
batch_size: 1
learning_rate: 3e-6
dataset_num_workers: 2
dataloader_prefetch_factor: 1
dataloader_persistent_workers: true
dataloader_in_order: false
validation_num_samples: 3
validation_num_inference_steps: 50
stage2_attention_mode: block_sparse_chunk_causal
stage2_topk_ratio: 2.0
stage2_local_num: -1
```

继承的一阶段权重：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

该 checkpoint 同时初始化：

- `lq_proj_in`
- DiT LoRA

### 9.4 v6.3 启动记录

48GPU 机器：

```text
rank0 b8gkuie2ns
rank1 wfnwbym4v6
rank2 kh5idf7f98
rank3 hj65iqg9rh
rank4 zhki5rrddw
rank5 xwk6qjuej5
```

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_3_lora_89f_videoonly_bs1_lr3e6_blocksparse_worker2_val_20260510_180600
```

启动前问题：

- 本机 `sync` tmux 中 `wfn/kh/hj/zh/xwk` 五个同步窗口已断；
- 从机缺少新 `v6.3` sh，第一次启动时 master 卡在等 rank；
- 恢复同步后确认所有从机 `V63_SH_PRESENT`，再重新启动。

启动结果：

- wandb 已开启；
- `lq_proj_in` 导入成功：`keys=8, missing=0, unexpected=0`；
- DiT LoRA 导入成功：`keys=480`；
- `Stage2 v6 attention mode: block_sparse_chunk_causal`；
- `Stage2 v6 topk_ratio: 2.0`；
- 前 9 步 loss 在 `0.061436` 到 `0.120919` 区间；
- b8 观察显存约 `143GB/GPU`，多数 GPU 利用率达到 `100%`。

### 9.5 当前 v6.1 inference 与作者官方推理仍可能不同的地方

`v6.1` 已经实现同类高层逻辑：

- LR projector 走 streaming / causal path；
- DiT 推理按 chunk 进入；
- 前序 chunk 的 K/V 会进入 cache；
- 后续 chunk 通过 cache 看到历史上下文；
- 使用 block-sparse chunk causal attention。

仍需要谨慎的地方：

- 当前实现是我们复刻的 streaming / KV-cache / sparse path，不是直接调用官方闭源或编译后的完整训练 kernel；
- 官方推理里的 block pair 选择、local window、top-k 细节可能和当前 PyTorch/自定义路径仍有微差；
- probe 结果显示只要走 sparse/chunk 路径就容易出现边界跳变，而 dense full-attention 不跳，因此后续需要重点核对官方 sparse kernel 与训练 sparse mask 是否完全一致。

### 9.6 v6.4.1：top-k 选择改为官方对齐的 chunk-grouped 版本

`v6.4.1` 是在 `v6.4` 首帧目标对齐逻辑基础上的 sparse attention 修正版。

本轮只修正 top-k 组织方式，不改变 Stage2 的其它训练设定：

- 仍从 Stage1 `v5.3.5` 89f checkpoint 初始化 `lq_proj_in` 与 DiT LoRA。
- 仍使用 `num_frames=89`、video-only 数据、Takano/Yubari 各 0.5。
- 仍使用 `stage2_attention_mode=block_sparse_chunk_causal`。
- 仍使用 `stage2_topk_ratio=2.0`。
- 训练中不加入 spatial local mask；`local_range` 仍视为高分辨率推理技巧，不进训练主线。

#### 9.6.1 旧 top-k 问题

旧 sparse path 更偏“全局 block/query 级 top-k”：

- 先得到所有 block pair 的 coarse score；
- 在较全局的 block 组织上选 top-k；
- 虽然也有 chunk causal allowed mask，但 top-k 的统计单元没有严格以 chunk 为组。

这个做法能跑，但和作者口述的流程不完全一致。作者流程更接近：

1. 先决定 chunk 之间哪些 pair 合法；
2. 只在合法 chunk pair 内看 block pair；
3. 每个 query chunk 内单独选 top-k block pair；
4. 对选中的 block pair 做真实 attention。

#### 9.6.2 新 top-k 逻辑

当前 `_select_topk_blocks(...)` 的核心流程：

1. 将 Q/K 按 `(2,8,8)` 划成 block。
2. 对每个 block 做 average pooling，得到 block-level Q/K。
3. 计算 block-level coarse score。
4. 先应用 chunk causal / temporal allowed mask。
5. 将 score reshape 为：

```text
(batch, head, query_chunk, spatial_block_pairs_in_allowed_chunks)
```

6. 在每个 `(batch, head, query_chunk)` 内根据 `stage2_topk_ratio` 选阈值。
7. 只保留该 chunk 内 top-k 的 block pair。

这样训练 sparse attention 的选择单元就和作者口径一致：以 chunk 为因果单位，以 `(2,8,8)` block 为稀疏 attention 单位。

#### 9.6.3 v6.4.1 启动记录

训练代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
```

关键 attention 文件：

```text
diffsynth/models/wan_video_dit_stage2_v6.py
diffsynth/models/wan_video_dit_stage2_v6_clean.py
diffsynth/models/wan_video_dit_stage2_v6_1.py
```

配置：

```text
wanvideo/model_training/flashvsr/configs/history/stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml
```

启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-1-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh
```

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100
```

启动节点：

```text
rank0 b8gkuie2ns
rank1 kh5idf7f98
rank2 hj65iqg9rh
rank3 zhki5rrddw
rank4 xwk6qjuej5
```

启动确认：

- `lq_proj_in` 导入成功：`keys=8, missing=0, unexpected=0`
- DiT LoRA 导入成功：`keys=480`
- `Stage2 v6 attention mode: block_sparse_chunk_causal`
- `Stage2 v6 topk_ratio: 2.0`
- 已出 loss：`step=1 loss=0.100667`，`step=2 loss=0.129647`，`step=9 loss=0.098759`
- 已触发 `step-10` validation，`sample_000` 已生成 `sr.mp4`
