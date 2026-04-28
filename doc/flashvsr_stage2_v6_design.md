# FlashVSR Stage 2 / v6 设计记录

日期：2026-04-27

## 目标

`v6` 对应 FlashVSR 论文中的 Stage 2：把 Stage 1 的 full-attention DiT 改造成 sparse-causal DiT，用于后续流式推理。

这一版先不继续混图像数据，训练数据切回 video only。这样可以把问题集中在三件事：

- `LR Proj-In` 是否已经是 causal 形式；
- DiT self-attention 是否按时间 causal；
- 后续 block-sparse / streaming cache 是否能和 FlashVSR 官方推理代码对齐。

## 论文里的 Stage 2 要点

论文 Stage 2 的核心描述是：

- 从 Stage 1 full-attention DiT 继续训练；
- 加 causal mask，每个 latent 只能看当前和过去位置；
- 引入 block-sparse attention；
- `LR Proj-In` 转成 causal variant；
- 继续用 flow matching loss；
- Stage 2 只用 video data，不再做 image-video joint training。

Block-sparse 细节：

- Q/K/V 按 3D block 切分；
- block size 是 `(2, 8, 8)`；
- 每个 block 展平后长度是 `2 * 8 * 8 = 128`；
- Q/K 被 reshape 成 `(B, block_num, 128, C)`；
- 对每个 block 内 token 做 average pooling，得到 block-level feature；
- 用 block-level feature 计算粗粒度 block-to-block attention map；
- 选 top-k block pair；
- 只在选中的 block pair 上做完整 `128 x 128` attention。

论文声称这样可以把 attention cost 降到 dense baseline 的 10%-20%，同时保持效果。

## 与 FlashVSR 官方推理代码的对齐

官方代码位置：

- `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/diffsynth/models/wan_video_dit.py`
- `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/diffsynth/pipelines/flashvsr_full.py`
- `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/examples/WanVSR/utils/utils.py`

官方实现里的关键结构：

- `WindowPartition3D.partition/reverse`
  - 把 `(B,F,H,W,C)` 按 `(2,8,8)` 切成 block；
  - 每个 block 是 128 个 token。
- `generate_draft_block_mask`
  - 先对 block 内 token 做平均池化；
  - 再根据 Q/K block feature 算 coarse attention；
  - 再 top-k 选 block pair。
- `block_sparse_attn_func`
  - 官方已经通过自定义 kernel 编译完成；
  - 当前 FlashVSR 官方推理能跑，说明环境里的 kernel 可用。
- `Causal_LQ4x_Proj`
  - pixel shuffle 做 8 倍空间压缩对齐；
  - 两层 `CausalConv3d`，每层 2 倍 temporal compression；
  - 总 temporal compression 是 4 倍；
  - 支持 cache，用于 streaming inference。

需要注意的一点：

- 不能简单用 PyTorch/FlashAttention 的 `is_causal=True`。
- WAN token 顺序是按 `(latent_time, h, w)` flatten 的。
- 如果直接 sequence-causal，会错误地让同一帧内后面的空间 token 看不到前面的空间 token。
- 正确 causal mask 应该是 time-aware：同一个 latent time 内的所有空间 token 互相可见，只屏蔽未来 latent time。

## v6 实现阶段

### v6.0：video only + causal LR Proj-In + dense time-causal attention

目的：先验证训练链路是对的。

实现内容：

- 新增 `diffsynth/models/wan_video_dit_stage2_v6.py`；
- 新增 `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py`；
- 新增 smoke config：
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_densecausal.yaml`
- 新增 smoke 启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-DenseCausal.sh`

第一版使用 `dense_time_causal`：

- self-attention 仍复用原来的 `q/k/v/o` 参数；
- 只替换 self-attention forward；
- 构造 time-aware dense mask；
- 同一 latent frame 内空间 token 全部可见；
- 未来 latent frame 被屏蔽。

这版是 correctness baseline，不是最终高分辨率训练方案。原因是 dense mask 在 1280x768 / 89f 时会非常大。

### v6.1：official-style streaming block causal

目的：接入官方 `block_sparse_attn_func`，并采用官方推理里的 chunk/cache 语义，而不是对整段奇数 latent-time 做 padding。

实现思路：

- 仍使用官方 `(2,8,8)` block layout；
- 第一个 chunk 使用 `f=6`；
- 后续 chunk 使用 `f=2`；
- 每层 self-attention 保存 `pre_cache_k/v`；
- 当前 chunk 的 K/V 与历史 cache 拼接后再进入 block-sparse attention；
- block-level mask 允许当前 chunk 访问 cache 中的过去 temporal block；
- 先不做 top-k pruning，用这个版本确认 block-sparse kernel 和训练链路能跑。

这里更正一个前期判断：

- 原始帧数满足 `4n+1` 时，VAE 后 latent-time 是奇数，比如 `17 -> 5`、`89 -> 23`；
- 官方 block window 的 temporal size 是 2；
- 但 FlashVSR 官方推理没有把 `5/23` 这种整段奇数 latent-time 一次性送进 block partition；
- 官方是通过 streaming chunk 避开这个问题：
  - 首段进入 DiT self-attention 的 latent-time 是 `6`；
  - 后续段进入 DiT self-attention 的 latent-time 是 `2`；
  - 因此每次进入 `(2,8,8)` block partition 的 temporal size 都能被 2 整除。

所以 v6 的 block-sparse 正式路线不应默认采用 “pad 1 个 latent-time 再裁回” 的方案。这个 padding 方案只能作为临时 fallback，不作为对齐官方的实现。

### v6.2：FlashVSR top-k block sparse

目的：对齐论文和官方推理里的 sparse selection。

需要补齐：

- block average pooling；
- coarse block attention map；
- local window mask；
- top-k block pair selection；
- 对齐官方 `topk_ratio / kv_ratio / local_range` 的参数语义。

### v6.3：top-k block selection / local attention window

目的：把训练结构和推理结构进一步对齐。

需要补齐：

- 官方 `generate_draft_block_mask` 的 top-k selection；
- local attention window；
- KV cache eviction；
- `LR Proj-In` cache；
- long-video streaming inference 的 overlap/buffer 组织方式。

## 当前代码边界

当前已经写入的是 v6.0 的训练骨架和 v6.1 的 kernel 接口：

- `dense_time_causal`
  - 可用于 smoke 和数值检查；
  - 不适合正式大分辨率长帧训练。
- `block_streaming_causal`
  - 已预留官方 `block_sparse_attn_func` 路径；
  - 已改为 cache/chunk 语义，不再对整段 odd latent-time 做 padding；
  - 训练侧还需要补全官方 overlap/buffer 的 loss 对齐方式。

当前还没有完成：

- top-k block selection；
- 与官方 `generate_draft_block_mask` 的逐项数值对齐；
- local attention window；
- 官方首段/后续段 overlap 组织在训练 loss 里的精确定义；
- v6 正式 16 卡启动脚本。

## 当前需要确认的问题

1. Stage2 训练是否要完全复刻官方推理的 overlap/buffer 输出组织。

官方推理不是简单 `0:6, 6:8, 8:10...`，而是首段 `0:6`，后续从 `4:6` 开始接 buffer/cache。训练时如果完全照抄，需要明确 duplicated/overlap latent 对 loss 的处理方式。

2. Stage2 smoke 用多少帧更合适。

`17f -> 5 latent-time` 不适合直接测试官方 block-sparse streaming chunk，因为首段要求 `f=6`。`dense_time_causal` 可以继续用 17f smoke，但 `block_streaming_causal` smoke 应该使用更长帧数。

3. v6.1 是否先做 all-past block causal，再做 top-k。

官方最终推理使用 `generate_draft_block_mask` 做 top-k block selection。为了定位问题，建议先做 all-past block causal，通过后再接 top-k；否则 kernel、mask、top-k 三个变量会同时变化。
