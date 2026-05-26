# FlashVSR Stage 3 训练计划：DMD One-Step Distillation

日期：2026-05-11

本文档基于本地论文 `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/2510.12747v1.pdf` 中 Stage 3 相关描述，以及当前 LucidVSR 已完成的 Stage 1 / Stage 2 代码状态，整理第三阶段训练需要实现的内容、依赖和风险。

## 0. 旧六节点母机 artifacts 兜底路径

2026-05-14 更新：

- 旧六节点母机已释放，节点包括：
  - `b8gkuie2ns`
  - `wfnwbym4v6`
  - `kh5idf7f98`
  - `hj65iqg9rh`
  - `zhki5rrddw`
  - `xwk6qjuej5`
- Stage 1 稳定母本 `v5.3.5` 与 Stage 2 `v6.4.1` 等历史 ckpt / validation / log 如果新机器需要恢复或测试，优先从旧主机的云端 artifacts 找：

```text
s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts
```

这个路径内容等价于 `b8gkuie2ns` 释放前本地：

```text
/mnt/task_wrapper/user_output/artifacts
```

注意：不要再默认尝试 ssh 旧六节点母机；新训练/测试需要文件时，从上述 S3 artifacts 按需复制到当前目标机器。

## 1. 论文里 Stage 3 到底做了什么

论文把 FlashVSR 训练分为三阶段：

- Stage 1：全注意力 image-video joint SR teacher。
- Stage 2：把 Stage 1 继续适配成 sparse-causal DiT，用于 streaming inference。
- Stage 3：把 Stage 2 的 sparse-causal DiT 蒸馏成 one-step streaming VSR model。

Stage 3 的核心名称是：

```text
Distribution-Matching One-Step Distillation
```

论文公式写得很明确，Stage 3 的总 loss 是：

```text
L =
  L_DMD(z_pred, G_one, G_real, G_fake)
  + L_FM(z_pred, G_fake)
  + ||x_pred - x_gt||_2^2
  + lambda * L_LPIPS(x_pred, x_gt)
```

其中：

- `G_one`：最终要训练出来的 one-step sparse-causal student。
- `G_real`：Stage 1 full-attention DiT teacher。
- `G_fake`：Stage 1 full-attention DiT 的 copy，用来学习 fake latent distribution。
- `z_pred`：`G_one` 一步生成出来的 latent。
- `x_pred`：`z_pred` 经过 decoder 后得到的 HR video/frame。
- `x_gt`：GT HR video/frame。
- `lambda = 2`。

结论：

- 是的，Stage 3 使用 DMD。
- 是的，Stage 3 使用 pixel-space reconstruction loss。
- 是的，Stage 3 使用 LPIPS loss，权重论文写为 `lambda=2`。
- 论文还保留了 flow matching loss，作用在 `z_pred` 和 `G_fake` 上。

## 2. Stage 3 和 Stage 2 的关系

Stage 3 不是重新从 Wan 或 Stage 1 开训，而是从 Stage 2 sparse-causal model 继续训练。

当前可作为 Stage 3 起点的代码与实验是：

- Stage 1 稳定母本：`v5.3.5`
  - 89 帧
  - non-streaming aligned LQ projector
  - 一个真实 video branch + 一个 image pseudo-video branch
  - 图像 branch 为 5 帧假视频
  - 用于给 Stage 2 提供初始化
- Stage 2 当前主线：`v6 / v6.1 / v6.3`
  - video-only
  - sparse-causal attention
  - causal LR projector
  - 训练使用 block/chunk causal mask
  - inference 使用 streaming KV-cache 路径

Stage 3 应该继承 Stage 2 的这些设定：

- 数据只用 video，不再用 image branch。
- LQ projector 使用 causal streaming variant。
- DiT self-attention 使用 Stage 2 sparse-causal 结构。
- 推理目标是 one-step，而不是 50-step。

## 3. 训练数据和输入形式

论文写明：

```text
Stage 2 continues training with videos only; Stage 3 adopts the same setting.
```

因此 Stage 3 数据应和 Stage 2 一致：

- Takano video + Yubari video。
- 不使用 image。
- 89 帧训练优先。
- 当前建议保持 Takano:Yubari = 1:1。
- 退化逻辑先保持当前 Stage 2 使用的 Aliyun full degradation，以便控制变量。

Stage 3 的输入形式：

- 输入 LR frames。
- 输入 Gaussian noise。
- 不再输入历史 predicted clips。
- 所有 latent 在统一 timestep 下训练。
- 使用 block-sparse causal attention mask 保持 streaming 因果关系。

这点和 Teacher Forcing / Self Forcing 不同：

- 不用 GT previous clip。
- 不用 predicted previous clip 作为额外输入。
- 训练和推理都只依赖 LR + noise + causal cache/mask。

## 4. DMD 部分需要怎么实现

论文只给了高层公式，没有展开 DMD 的完整代码级细节。因此实现需要借鉴 DMD / one-step diffusion 代码。

最低可行实现需要三个模型角色：

### 4.1 `G_one`

这是要训练的 one-step student。

初始化：

- 从当前 Stage 2 checkpoint 初始化。
- 加载 LoRA。
- 加载 LQ projector。
- 保持 Stage 2 sparse-causal attention 结构。

训练目标：

- 输入 LR + Gaussian noise。
- 只跑一步，输出 `z_pred`。
- 反向传播来自：
  - DMD loss
  - FM loss
  - pixel MSE
  - LPIPS

### 4.2 `G_real`

这是 real distribution teacher。

论文说：

```text
The Stage 1 full-attention DiT serves as the teacher G_real
```

因此它不是 Stage 2 sparse-causal 模型，而是 Stage 1 full-attention teacher。

实现上需要：

- 加载 Stage 1 稳定母本，例如 `v5.3.5` 的最终 checkpoint。
- 使用 full-attention / Stage 1 路径。
- 作为 frozen teacher，不更新参数。
- 给 DMD loss 提供 real score / real distribution direction。

### 4.3 `G_fake`

论文说：

```text
its copy G_fake learns the distribution of fake latents, following the DMD pipeline
```

含义：

- `G_fake` 是 `G_real` 的 copy。
- 但它需要学习 fake latent distribution。
- 它和 `G_one` 不是同一个模型。
- 它在 DMD 里通常承担 fake score model / fake denoiser 的角色。

实现上有两个可能路线：

- 复刻 DMD 原始做法：`G_fake` 有自己的 optimizer，按 DMD 论文方式更新。
- 如果先做 smoke，可先 frozen `G_fake` 或简化更新，只验证 pipeline，但这不是最终正确版本。

正式复刻需要别人写好的 DMD 代码或官方 DMD loss 代码来对齐。

## 5. Pixel 和 LPIPS loss 怎么接

Stage 3 不只在 latent 上蒸馏，还要把 `z_pred` decode 到 pixel space。

流程：

```text
z_pred = G_one(LR, noise)
x_pred = WanVAE.decode(z_pred)
loss_pixel = MSE(x_pred, x_gt)
loss_lpips = LPIPS(x_pred, x_gt)
```

论文写：

```text
loss_recon = ||x_pred - x_gt||_2^2 + 2 * LPIPS(x_pred, x_gt)
```

注意点：

- 直接 decode 全部 latent 显存会爆。
- 论文写了“Due to memory constraints, two latents are randomly selected per iteration for decoding, with previous ones detached from gradients.”
- 因此训练时不能每次 decode 全部 89 帧对应 latent。
- 当前可实现为：每个 iteration 随机选 2 个 latent-time / frame group 做 decode 和 pixel/LPIPS。
- 未选中的 latent 只参与 DMD/FM，不参与 pixel decode。

## 6. LPIPS 依赖

需要引入 LPIPS 实现，常见选择：

```python
import lpips
loss_fn = lpips.LPIPS(net="vgg")
```

或者复用已有项目里的 LPIPS 封装。

注意：

- LPIPS 输入通常要求 RGB，范围可能是 `[-1, 1]`。
- 需要确认当前视频 tensor 范围，是 `[0, 1]`、`[-1, 1]` 还是 degradation pipeline 输出格式。
- LPIPS 应只在 rank local 上计算，再参与分布式 loss。
- LPIPS 网络本身 frozen，不参与训练。

## 7. 如果提供别人写好的 DMD + LPIPS 代码，能否复刻

可以复刻，但需要确认代码给出了以下内容：

- DMD loss 的完整 forward 逻辑。
- `G_real / G_fake / G_one` 三者的更新顺序。
- `G_fake` 的 optimizer 是否单独更新。
- timestep / noise sampling 规则。
- fake latent 的 detach / no-grad 位置。
- DMD loss 权重、FM loss 权重、pixel loss 权重、LPIPS 权重。
- 是否使用 EMA。
- 是否需要 discriminator-like alternating update。
- 是否需要 mixed precision 下的稳定化处理。

如果代码只给 LPIPS，不给 DMD，Stage 3 只能先做“one-step + pixel/LPIPS/FM”版本，但那不等价于论文 Stage 3。

如果代码给的是通用 DMD，例如图像扩散模型 DMD，也可以迁移，但需要改成：

- video latent shape。
- Stage 2 sparse-causal DiT forward。
- LR projector conditioning。
- Wan scheduler / flow matching 目标。
- streaming causal mask。

## 8. 建议的实现路线

### Step 0：先不要一次性上 48 卡

先做 1-2 卡 smoke：

- 只取 1 个 batch。
- 只跑 1 个 timestep。
- 只 decode 1-2 个 latent。
- 确认 forward / backward / optimizer step / ckpt 保存全部能走通。

### Step 1：实现 Stage 3 数据和模型包装

新增代码，不覆盖 Stage 2：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_lora.py
```

新增 config：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_lora_89f_videoonly_dmd.yaml
```

新增 sh：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-Lora-89f-VideoOnly-DMD.sh
```

### Step 2：先接 one-step student，不接 DMD

目标：

- `G_one` 从 Stage 2 ckpt 加载。
- 输入 LR + noise。
- 一步输出 `z_pred`。
- decode 随机 2 个 latent。
- 只训：

```text
MSE + 2 * LPIPS
```

用途：

- 验证 decode、LPIPS、显存、数据和分布式都正常。
- 这个版本不是最终论文复刻，只是工程 smoke。

### Step 3：接 FM loss

加入：

```text
L_FM(z_pred, G_fake)
```

这里需要明确 `G_fake` 的初始化和目标：

- 先用 Stage 1 teacher copy。
- 再按 DMD 代码确定是否更新。

用途：

- 验证 latent-space training 是否稳定。
- 检查 loss scale。

### Step 4：接 DMD loss

正式接入：

```text
L_DMD(z_pred, G_one, G_real, G_fake)
```

此时需要 DMD 代码。

建议先完全照搬别人可工作的 DMD 代码，不要自己猜公式。

要做的适配：

- 把 image latent 改成 video latent。
- 把 text/image condition 改成 LR projector condition。
- 把 dense DiT forward 改成 Stage 2 sparse-causal DiT forward。
- 把 teacher `G_real` 固定为 Stage 1 full-attention。
- 把 fake model `G_fake` 的更新策略照 DMD 原实现保留。

### Step 5：小规模稳定性 sweep

先扫：

- `lr=1e-6`
- `lr=3e-6`
- `lr=1e-5`
- `lpips_weight=2`
- `pixel_weight=1`
- `dmd_weight=1`
- `fm_weight=1`

实际权重要看 DMD 代码默认值，不能只按论文公式盲开。

### Step 6：48 卡正式训练

只有 smoke 满足下面条件才上：

- 2 卡能出 loss。
- 显存不持续爬升。
- `G_one / G_fake` optimizer 状态保存正常。
- validation 能跑 one-step inference。
- checkpoint 能 reload。
- 输出视频没有明显 chunk jump 新增问题。

## 9. Validation / inference 要怎么做

Stage 3 是 one-step 模型，因此 validation 不能再用 Stage 2 的 50-step inference。

正确 validation：

```text
LR video
-> causal LR projector
-> Gaussian noise
-> G_one one-step sparse-causal DiT
-> Wan decoder or TC decoder
-> SR video
```

当前建议先用 Wan decoder，不先做 TC Decoder。

原因：

- TC Decoder 是论文 3.4 的额外加速模块。
- 它有单独训练目标。
- 先把 Stage 3 one-step DiT 做通，再考虑 TC Decoder。

## 10. 当前最关键的不确定项

以下问题需要 DMD 代码或作者确认：

- `G_fake` 是否每 step 更新，还是交替更新。
- `G_fake` 的初始化是否严格来自 `G_real`。
- `G_real` 使用 Stage 1 full-attention teacher 时，是否需要 50-step teacher sample，还是只需要 teacher score。
- DMD loss 的 timestep 分布。
- `L_FM(z_pred, G_fake)` 的具体 target 怎么构造。
- DMD / FM / pixel / LPIPS 的实际 loss 权重。
- LPIPS decode 的两个 latent 是按 latent-time 选，还是按 frame 选。
- “previous ones detached from gradients” 在实现中具体是 detach cache，还是 detach earlier decoded latent。

## 11. 对当前 LucidVSR 的建议

最稳妥的 Stage 3 路线：

1. 保留现有 Stage 2 `v6.3` 低学习率训练。
2. 新开 Stage 3 `v7` 分支，不动 Stage 2 代码。
3. 先实现 one-step + pixel/LPIPS smoke。
4. 拿到 DMD 代码后再接真正 DMD。
5. `G_real` 固定加载 Stage 1 `v5.3.5`。
6. `G_one` 固定加载 Stage 2 最好 ckpt。
7. `G_fake` 按 DMD 代码初始化为 `G_real` copy。
8. validation 一律用 one-step，不再用 50-step。

## 12. 最小可交付代码清单

建议新增：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_lora.py
wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7.py
wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_batch.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_lora_89f_videoonly_dmd.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-Lora-89f-VideoOnly-DMD.sh
```

如果 DMD 代码给到，再新增：

```text
wanvideo/model_training/flashvsr/losses/dmd_loss.py
wanvideo/model_training/flashvsr/losses/lpips_loss.py
```

## 13. 结论

Stage 3 不是简单把 Stage 2 继续训短步数，而是一个新的 one-step distillation 训练：

- 要保留 Stage 2 的 sparse-causal / streaming 结构。
- 要从 Stage 2 初始化 `G_one`。
- 要用 Stage 1 full-attention 作为 `G_real` teacher。
- 要引入 `G_fake` 学 fake latent distribution。
- 要使用 DMD + FM + MSE + LPIPS。
- 如果提供可工作的 DMD 和 LPIPS 代码，可以复刻；其中 LPIPS 很直接，DMD 需要严格照原实现迁移，不能只按论文公式手写猜测。

## 14. 2026-05-13：论文与代码证据库

本节集中记录 Stage3 相关论文和代码事实。后续任何 `v7` 改动都必须先对照本节，避免只验证“代码能跑”而漏掉语义不一致。

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/2510.12747v1.pdf
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/2405.14867v2.pdf
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/OSEDiff/2406.08177v3.pdf
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/main/sd_guidance.py
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/main/train_sd.py
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/OSEDiff/train_osediff.py
```

### 14.1 FlashVSR 论文对 Stage3 的硬约束

FlashVSR Stage3 不是普通的 one-step pixel regression。论文公式写的是：

```text
L =
  L_DMD(z_pred, G_one, G_real, G_fake)
  + L_FM(z_pred, G_fake)
  + ||x_pred - x_gt||_2^2
  + 2 * L_LPIPS(x_pred, x_gt)
```

必须满足的语义：

- `G_one`：从 Stage2 sparse-causal model 初始化，是最终训练目标；
- `G_real`：Stage1 full-attention teacher，frozen；
- `G_fake`：Stage1 full-attention teacher 的 copy，用来学习 fake latent distribution；
- `DMD/FM` 主体作用在 latent space；
- `pixel/LPIPS` 必须先 decode：`z_pred -> Wan decoder -> x_pred`，再和 `x_gt` 对齐；
- Stage3 只用 video data，不再用 image branch；
- Stage3 推理目标是一阶段 one-step streaming VSR，不是 50-step validation。

关于 pixel/LPIPS 的显存约束，论文明确提到：

```text
Due to memory constraints, two latents are randomly selected per iteration for decoding,
with previous ones detached from gradients.
```

因此正确实现不能固定 decode 前缀，也不能全片 decode。正确方向是：

- 每个 iteration 随机选 `2` 个 latent-time window；
- 只有被选中的 latent window 走 Wan decoder 算 MSE/LPIPS；
- 未选中的 latent 仍然走 DMD/FM/flow，但不参与 pixel/LPIPS decode；
- Wan VAE 的第 0 个 latent 有首帧特殊语义，随机 window 对 GT 帧的映射必须显式处理；
- “previous ones detached from gradients” 不能忽略，需要在代码里明确是 detach earlier latent context，还是 detach decoder cache/previous latent。

当前未完全确认的问题：

- 随机 2 latent 是否必须连续；
- 如果随机 window 不包含 latent 0，Wan decoder 是否需要从更早 latent 作为 detached context 开始 decode；
- 第 0 latent 对应首帧不压缩，后续 latent 对应 4 帧压缩段，pixel loss 的 frame 对齐规则必须单独写清楚。

### 14.2 DMD2 对 v7 最有用的部分

DMD2 论文重新解释了 DMD 的核心：训练 one-step generator 时，不强制逐点拟合 teacher 的采样轨迹，而是通过 real score 和 fake score 的差来做 distribution matching。形式上，DMD loss 的梯度来自：

```text
s_real(F(G(z), t), t) - s_fake(F(G(z), t), t)
```

其中：

- `G` 是要训练的 one-step generator；
- `s_real` 由 frozen teacher diffusion model 提供；
- `s_fake` 由动态训练的 fake diffusion critic 提供；
- `F(..., t)` 表示把 generator 输出重新加噪到某个 timestep。

这和 FlashVSR Stage3 的三个模型角色是对应的：

```text
G_one  -> DMD2 里的 one-step generator G
G_real -> real score teacher
G_fake -> fake score model / fake denoiser
```

DMD2 对我们最重要的工程结论是：`G_fake` 不能只是形式上存在。论文明确说，去掉 regression loss 后训练不稳定，主要原因是 `G_fake` 没有及时跟上 `G_one` 输出分布的变化。DMD2 的解决方案是 two time-scale update rule：

```text
每更新 1 次 generator，更新多次 fake score model。
```

论文中 ImageNet 推荐 `5` 次 fake update；SD v1.5 阶段甚至用了 `10` 次 fake update。对 LucidVSR v7 来说，这意味着：

- `G_fake` 应该有独立 optimizer；
- `G_fake` 更新频率应该可配置，例如 `fake_update_ratio=5`；
- 不能把 `G_fake` 冻住后声称实现了 DMD；
- smoke 阶段可以先简化，但正式 v7 必须恢复 `G_fake` 的交替更新。

DMD2 还提出了 GAN loss，但 FlashVSR Stage3 公式里没有 GAN 项。因此 LucidVSR v7 第一版不应主动加入 GAN，否则会偏离 FlashVSR 论文主线。GAN 可以作为后续感知质量增强实验，而不是 v7 的第一目标。

### 14.3 DMD2 代码事实

本地 DMD2 代码里可以确认几个关键实现事实：

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/main/sd_guidance.py
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/main/train_sd.py
```

`sd_guidance.py` 的 DMD 逻辑：

- `compute_distribution_matching_loss(...)` 中：
  - 对 generator 输出 latent 重新加噪；
  - 分别用 `real_unet` 和 `fake_unet` 预测噪声；
  - 由 `pred_real_image` 和 `pred_fake_image` 构造 DMD gradient；
  - loss 写成 `0.5 * mse(original_latents, (original_latents - grad).detach())`，相当于手写 distribution matching gradient。
- `compute_loss_fake(...)` 中：
  - `latents = latents.detach()`；
  - `fake_unet` 对 fake latent 重新加噪后的样本做噪声预测；
  - fake loss 是 `MSE(fake_noise_pred, noise)`；
  - 这说明 `G_fake` 不是 frozen teacher，而是要追随当前 generator 产生的 fake distribution。

`train_sd.py` 的更新逻辑：

- 存在两个 optimizer：
  - `optimizer_generator`
  - `optimizer_guidance`
- 存在 `dfake_gen_update_ratio`；
- generator 并不是每步都更新，代码里有：

```text
COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
```

这和当前 LucidVSR 通用 runner 冲突：`launch_training_task` 只有一个 optimizer，不能直接实现 DMD2 的双 optimizer / alternating update。

因此工程结论：

- `G_fake` 必须是单独 trainable model；
- 必须有单独 optimizer 和 scheduler；
- 必须有独立保存 / resume；
- 不能用 `model.trainable_modules()` 的单 optimizer runner 假装实现 `G_fake`；
- 如果只把 fake loss 加到 `G_one` 上，会是语义错误。

### 14.4 DMD2 对 loss 设计的提醒

DMD2 反对依赖大规模 teacher-generated regression pair，因为这种 regression 会把 student 绑死在 teacher 的采样路径上，削弱 distribution matching 的意义。

但 FlashVSR Stage3 不是纯 DMD2，它显式保留了：

```text
pixel MSE + 2 * LPIPS
```

所以 v7 的判断是：

- 不需要预先生成大量 teacher trajectory pair；
- 不需要让 student 逐点回归 50-step teacher 输出；
- pixel/LPIPS 是和 GT 对齐的 reconstruction supervision，不是 teacher trajectory regression；
- DMD 部分仍然应该在 latent distribution 上做，而不是把所有 supervision 都变成 pixel loss。

DMD2 还提醒了稳定性问题：如果 v7 出现整体亮度漂移、颜色周期性波动或全局统计不稳定，第一怀疑对象应该是 `G_fake` 更新频率不足，而不是先改数据或退化。

### 14.5 OSEDiff 对 v7 最有用的部分

OSEDiff 是 Real-ISR 的 one-step diffusion 训练，不是视频 VSR，也不是 FlashVSR Stage3 的直接等价实现。但它对 v7 有三个很有用的工程启发。

第一，OSEDiff 把 one-step restoration 写成：

```text
LQ -> latent -> one-step diffusion denoise -> decoded HQ
```

这和我们 v7 的目标一致：不要再用 50-step inference 做 validation，而是直接让 `G_one` 一步输出 `z_pred`，再 decode 得到 `x_pred`。

第二，OSEDiff 的 data loss 是：

```text
MSE(x_pred, x_gt) + lambda1 * LPIPS(x_pred, x_gt)
```

它和 FlashVSR Stage3 的 pixel-space reconstruction 项一致。OSEDiff 论文里 `lambda1=2`，FlashVSR Stage3 也写了 `lambda=2`。因此 v7 的 pixel loss 第一版可以直接使用：

```text
pixel_weight = 1
lpips_weight = 2
```

第三，OSEDiff 强调 VSD / DMD 类 regularization 最好在 latent space 做，而不是在 image space 反复 encode/decode。这和 FlashVSR Stage3 的设计也一致：DMD/FM 主体应该作用在 latent 上，pixel/LPIPS 只作为抽样 decode 的 reconstruction 约束。

### 14.6 OSEDiff 代码事实

本地 OSEDiff 代码里确认：

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/OSEDiff/train_osediff.py
```

关键实现：

- `net_lpips = lpips.LPIPS(net='vgg').cuda()`；
- `net_lpips.requires_grad_(False)`；
- 训练里同时计算：
  - `loss_l2`
  - `loss_lpips`
  - `loss = loss_l2 + loss_lpips`

对 LucidVSR 的工程结论：

- LPIPS 应作为 frozen loss-only module；
- 不应该把 LPIPS 注册进 Deepspeed 管理的训练模型，否则会污染 state 保存；
- 当前 `v7-A/v7-B` 用 `_get_stage3_lpips(...)` 作为 loss-only cache 是正确修复方向。

### 14.7 OSEDiff 不能直接照搬的地方

OSEDiff 的对象是单图 Real-ISR，使用 Stable Diffusion 的 VAE encoder / UNet / decoder 结构；LucidVSR 是 Wan video latent + causal sparse DiT + LQ projector。因此不能直接照搬以下部分：

- 不能照搬 OSEDiff 的 text prompt / DAPE 逻辑；
- 不能照搬 SD 的 `z_L -> z_H` 公式到 Wan video latent；
- 不能照搬图像 batch 的 LPIPS 维度处理，必须适配 `B, T, C, H, W` 或逐帧计算；
- 不能训练 Wan VAE decoder，当前应保持 Wan decoder frozen；
- 不能把 OSEDiff 的 VAE encoder LoRA 设定引入 v7，除非后续专门做 decoder/encoder 实验。

OSEDiff 对 v7 的价值是 loss 和 latent regularization 思想，不是模型结构本身。

### 14.8 对 v7 第一版的更新建议

结合 FlashVSR、DMD2、OSEDiff 三篇论文，v7 第一版应该按下面顺序实现：

```text
Phase A: one-step + pixel/LPIPS smoke
Phase B: 加入 G_fake 的 FM / fake-score 更新
Phase C: 加入真正 DMD loss
Phase D: 再考虑 GAN 或其他 perceptual regularizer
```

Phase A 的目的只是打通：

- Stage2 checkpoint 初始化 `G_one`；
- 一步输出 `z_pred`；
- 随机 decode 1-2 个 latent group；
- 计算 `MSE + 2 * LPIPS`；
- 能 backward、保存 checkpoint、validation one-step 输出。

Phase B / C 才是真正的 FlashVSR Stage3：

- `G_real` 使用 Stage1 full-attention teacher；
- `G_fake` 从 `G_real` copy 初始化；
- `G_fake` 使用独立 optimizer；
- `fake_update_ratio` 至少先设为 `5`；
- DMD/FM 在 latent space 上做；
- pixel/LPIPS 只对随机抽中的 decoded latent/frame group 做。

### 14.9 当前仍需要代码确认的问题

读完论文后，仍需要通过 DMD2 / OSEDiff 代码确认下面实现细节：

- DMD2 代码里 real score 和 fake score 是如何从 denoiser output 换算出来的；
- fake update 和 generator update 的具体顺序；
- DMD loss 是否需要对 `G_one` 输出做手写 gradient，还是能写成普通 autograd loss；
- timestep 采样范围和权重 `w(t)`；
- fake data 是否 detach，以及在哪里 detach；
- LPIPS 输入范围，是否需要从 `[0, 1]` 转成 `[-1, 1]`；
- 多卡下 `G_fake` optimizer state 如何保存和恢复；
- 对 Wan video latent，`G_real/G_fake/G_one` 的 latent 长度必须完全一致，尤其是 89 帧输入对应 85 帧有效输出的 Stage2 语义。

因此下一步应该开始读代码，而不是继续只按公式推导。

### 14.10 Stage3 代码改动强制自查表

后续每次改 `v7`，必须在本节维护一条审计记录。每条记录至少包含：

```text
版本：
代码入口：
论文要求：
代码实际做法：
是否对齐：
如果不对齐，为什么暂时允许：
验证方式：
不能声称已经完成的内容：
```

最低自查项：

| 检查项 | 论文/代码要求 | 当前必须检查什么 |
|---|---|---|
| one-step | Stage3 是 one-step student | validation/inference 不能默认 50-step |
| DMD | 需要 `G_one/G_real/G_fake` | 是否真的有三模型，还是只有 `G_one` |
| G_fake | DMD2 代码显示必须独立 optimizer | 是否有 dual optimizer / fake update ratio |
| pixel loss | 必须 `z_pred -> decoder -> x_pred` | 是否真的 decode 后再算 MSE/LPIPS |
| random 2 latents | 论文要求每步随机选 2 个 latent decode | 是否仍然固定前缀或全片 decode |
| previous detach | 论文要求 previous ones detached | 是否实现 detached context/cache |
| 首帧 | Wan VAE 第 0 latent 特殊 | 第 0 latent 的 GT frame mapping 和 loss weight 是否明确 |
| LPIPS | OSEDiff 代码 frozen loss-only | LPIPS 是否避免注册进 Deepspeed module tree |
| 保存 | DMD2 有多 optimizer state | checkpoint/resume 是否覆盖 `G_fake` optimizer |
| runner | 当前 runner 单 optimizer | 是否误用 `launch_training_task` 跑双模型 |

## 15. 2026-05-13：v7-A 代码落地范围

本次新增 `v7-A`，只作为 Stage3 的第一个可验收闭环，不声称已经完成完整 DMD。

新增文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`

`v7-A` 当前实现：

- 复用 `v6.4` 的 video-only 数据读取；
- 复用 Stage2 sparse-causal DiT / causal LQ projector / checkpoint 导入路径；
- 新增 `Stage3AOneStepReconLoss`：
  - 随机采样 flow-matching timestep；
  - 用 `G_one` 一步预测 `noise_pred`；
  - 通过 `scheduler.step(..., to_final=True)` 得到 `z_pred`；
  - 用 Wan decoder 把 `z_pred` decode 到 pixel space；
  - 计算 `loss_flow + MSE + 2 * LPIPS`；
  - `LPIPS` 按逐帧 `BT,C,H,W` 方式计算。

重要审计：

| 项目 | 当前 `v7-A` 做法 | 和论文是否一致 | 备注 |
|---|---|---|---|
| one-step student | 是，一步 `scheduler.step(..., to_final=True)` | 对齐 | 这是 Phase A 的核心 |
| pixel/LPIPS 必须 decode | 是，`z_pred -> pipe.vae.decode -> x_pred` | 对齐 | 这点已验证 |
| 随机 2 latent decode | 否，当前固定取前缀 `z_pred[:, :, :stage3_recon_num_latents]` | 不对齐 | 这是当前最大语义缺口 |
| 未选 latent 只走 DMD/FM | 当前未实现 DMD/FM，未选 latent 只走 flow | 部分对齐 | Phase A 只 smoke，不是完整 Stage3 |
| previous detach | 未实现 | 不对齐 | 需要后续确定 decoder context/cache 的 detach 方式 |
| G_real | 未实现 | 不对齐 | Phase A 刻意不做 |
| G_fake | 未实现 | 不对齐 | Phase A 刻意不做 |
| DMD loss | 未实现 | 不对齐 | Phase A 刻意不做 |
| LPIPS loss-only module | 是，后续改为 `_get_stage3_lpips(...)` cache | 对齐 | 避免 Deepspeed state 保存错误 |
| 首帧 pixel weight | MSE 首帧乘 4，LPIPS 默认不乘 | 部分对齐 | 依据作者“pixel loss 首帧乘 4”说明 |

因此，`v7-A` 只能叫：

```text
one-step + fixed-prefix pixel/LPIPS smoke
```

不能叫完整 FlashVSR Stage3。

当前刻意没有实现的内容：

- 没有 `G_real`；
- 没有 `G_fake`；
- 没有 DMD loss；
- 没有 fake score model 的独立 optimizer；
- 没有 GAN loss。

验收方式：

1. 先在 2 卡 smoke 运行 `FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`。
2. 必须确认：
   - 能从 Stage2 checkpoint 导入 LoRA 与 `lq_proj_in`；
   - 能输出第一条 loss；
   - loss 分量中至少包含 flow / mse / lpips；
   - validation 能用 one-step 路径保存视频；
   - checkpoint 能保存并重新加载。
3. `v7-A` 通过后再进入 `v7-B`：新增 `G_fake` 和独立 fake-score 更新。

## 16. 2026-05-13：v7-A smoke 验收结果

`v7-A` 的第一轮 smoke 已经完成到一次 forward / backward / trainable checkpoint 保存。

已验证内容：

- Stage2 `v6.4.1` checkpoint 能作为 `v7-A` student 初始化来源；
- `lq_proj_in` 和 LoRA 都能正确导入；
- one-step denoise 路径能得到 `z_pred`；
- Wan decoder 能在 temporal prefix 上 decode 出 pixel video；
- `MSE + LPIPS` 都实际参与 loss；
- 训练导出的 `step-1.safetensors` 包含 LoRA 与 `lq_proj_in`。

关键 smoke 数值：

```text
loss=0.422148
flow=0.184295
mse=0.004079
lpips=0.116887
recon_latents=2
decoded_frames=5
```

本次 smoke 暴露出两个正式训练前必须处理的问题：

1. 不能直接 decode 完整 89 帧视频计算 pixel/LPIPS loss。
   - 1280x768、89 帧下 Wan decoder 显存过大；
   - 当前修法是 `flow loss` 监督整段 latent，pixel/LPIPS 只监督 temporal prefix；
   - 后续可扩展为随机 temporal chunk，但不能回到整段 decode。

2. Deepspeed training state 保存会因为后注册的 frozen LPIPS/VGG 参数报错。
   - trainable safetensors 已保存成功；
   - 失败点是 `accelerator.save_state()`；
   - 根因是 forward 内把 LPIPS 注册到 `pipe` 上，导致 Deepspeed state 保存时看到一个不在初始化 named-param 映射里的 frozen module。

因此当前结论是：

- `v7-A` 的核心训练逻辑有效；
- `v7-A` 初版还不是可长训版本；
- 下一步应先修 Deepspeed state 保存，再进入 `v7-B` 的 `G_fake` / DMD 实现。

## 17. 2026-05-13：pixel loss 首帧权重

Stage3 解码到 pixel space 后，pixel/MSE loss 需要显式处理 Wan VAE 的首帧特殊性。

当前 `v7-A` 采用：

```text
loss_mse = mean_t( weight[t] * mse_frame[t] )
weight[0] = 4.0
weight[t>0] = 1.0
```

注意这里不是按 `sum(weight)` 重新归一化，而是仍按原始帧数平均。这样第 0 帧的 pixel 梯度就是普通逐帧平均下的 4 倍，符合“首帧 pixel loss 乘 4”的设计。

当前配置：

```yaml
stage3_first_frame_pixel_weight: 4.0
stage3_first_frame_lpips_weight: 4.0
```

更新：2026-05-14 后，LPIPS 首帧也跟随 pixel 首帧乘 4。原因是 LPIPS 仍然是 pixel-space reconstruction loss 的 perceptual 部分，如果只放大 MSE 而不放大 perceptual loss，首帧监督仍不完全对齐 Wan 首 latent 的特殊语义。

## 18. 2026-05-13：v7-A state 保存与 resume 已修复

修复点：

- LPIPS 不再注册为 `pipe` 的子模块；
- 新增 `_get_stage3_lpips(pipe, net)` 作为 loss-only cache；
- 如果历史 `_stage3_lpips` 已经进入 `_modules`，会从 `_modules` 里移除；
- smoke 启动脚本支持 `EXTRA_ARGS` 透传，后续可以稳定追加 `--resume_training_state_dir` 等调试/恢复参数。

验证一：完整 20 step smoke。

- 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_smoke_statefix`
- 结果：
  - `step-1 / step-2 / step-5 / step-10 / step-20.safetensors` 均保存成功；
  - `training_state/step-1 / step-2 / step-5 / step-10 / step-20` 均保存成功；
  - 无 `Traceback / RuntimeError / ValueError`。

验证二：从 `training_state/step-10` resume。

- 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_true_resume_from_step10`
- 关键日志：

```text
[resume] training state loaded step=10 epoch_id=0
```

- 结果：
  - 从 step 10 继续训练到 step 20；
  - 成功保存新的 `step-20.safetensors`；
  - 成功保存新的 `training_state/step-20`。

当前结论：

- `v7-A` 已经通过基础长训链路验收；
- 现在可作为 Stage3 的第一个稳定分支继续扩展；
- 下一步可以进入 `v7-B`：加入 `G_fake` / DMD 相关 loss，而不是继续修 one-step reconstruction 的基础工程问题。

## 19. 2026-05-13：v7-B 第一版边界

`v7-B` 已经从 `v7-A` 复制为独立分支，但第一版只做安全脚手架，不直接开启 DMD。

新增入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-B-Lora-89f-VideoOnly-FakeFM.sh
```

当前 `v7-B` 参数：

```yaml
stage3_fake_checkpoint: null
stage3_fake_fm_weight: 0.0
stage3_fake_update_ratio: 5
```

这个版本有意把 `stage3_fake_fm_weight` 锁在 `0.0`。如果设成非 0，代码会直接报错。

原因是当前通用训练入口 `launch_training_task` 只有一个 optimizer：

```text
optimizer = AdamW(model.trainable_modules(), ...)
```

而 DMD2 的 `G_fake` 不是普通 loss 分量，它需要第二个可训练模型和第二个 optimizer：

```text
student / generator optimizer
G_fake / guidance optimizer
fake_update_ratio
generator turn / guidance turn
```

因此不能把 `G_fake` 直接塞进现有单 optimizer runner，否则会得到一个“能跑但语义错误”的训练。

`v7-B` 当前审计：

| 项目 | 当前 `v7-B` 做法 | 和论文是否一致 | 备注 |
|---|---|---|---|
| 独立入口 | 是，复制自 `v7-A` | 工程上对齐 | 保证不污染 `v7-A` |
| one-step validation | 是，继承 `v7-A` one-step 路径 | 对齐 | 不是 50-step |
| G_fake 参数 | 已接入 `stage3_fake_checkpoint / stage3_fake_fm_weight / stage3_fake_update_ratio` | 只是接口 | 尚未训练 `G_fake` |
| G_fake 更新 | 没有，`stage3_fake_fm_weight` 非 0 会报错 | 不对齐 | 防止误跑错误 DMD |
| dual optimizer | 没有 | 不对齐 | 必须新写 runner |
| 随机 2 latent decode | 没有，仍固定前缀 decode | 不对齐 | 下一步先修这个 |
| previous detach | 没有 | 不对齐 | 随机 latent decode 时一起设计 |
| DMD loss | 没有 | 不对齐 | 需要 DMD2 迁移 |

因此，`v7-B` 目前不是完整 DMD，只是：

```text
v7-A 的安全复制分支 + G_fake 参数脚手架 + 防误用 guard
```

本轮 smoke 只验证：

- `v7-B` 独立入口不污染 `v7-A`；
- Stage2 checkpoint 能导入；
- one-step reconstruction loss 仍然可训练；
- checkpoint 和 training state 能正常保存；
- fake 参数已经接入配置和日志，但 fake 更新尚未启用。

smoke 目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_20260513_v7b_smoke
```

验证结果：

```text
step-1 / step-2 / step-5 / step-10 / step-20.safetensors saved
training_state/step-1 / step-2 / step-5 / step-10 / step-20 saved
step=20 loss=0.669951
```

下一步真正进入 `v7-B` DMD，需要单独写 dual-optimizer runner，而不是继续复用当前 `launch_training_task`。

## 20. 2026-05-13：下一步先修 random latent decode，而不是急着接 DMD

当前已经确认：

- `v7-A/v7-B` 的基础 one-step + decoder + MSE/LPIPS 链路能跑；
- `v7-A/v7-B` 都没有实现论文要求的随机 2 latent decode；
- 如果继续在固定前缀 decode 上接 DMD，会把一个不对齐的 pixel loss 设计固化进主线。

因此下一步顺序调整为：

1. 只在 `v7-B` 修改，不碰 `v7-A`。
2. 新增 random latent decode window：
   - 每个 iteration 随机选择一个连续 latent window；
   - window 长度默认 `stage3_recon_num_latents=2`；
   - 只对这个 window decode 后算 MSE/LPIPS；
   - 其他 latent 保留完整 flow loss。
3. 明确 GT frame 对齐：
   - latent 0 对应首帧特殊解码；
   - latent `i>0` 对应后续 4 帧组；
   - 不能简单用 `x_gt[:, :, :target_frames]`。
4. 明确 previous detach：
   - 如果 Wan decoder 必须从更早 latent/context 开始才能正确 decode window，则更早 latent 只作为 detached context；
   - pixel/LPIPS loss 只回传到被选中的 window。
5. 加 debug metadata：
   - `selected_latent_start`
   - `selected_latent_end`
   - `decoded_frame_start`
   - `decoded_frame_end`
   - `decoded_frames`
   - `detached_context_latents`
6. 2 卡 smoke 验收：
   - 至少看到不同 step 抽到不同 latent window；
   - decode frame range 和 GT range 能打印出来；
   - `loss_flow / loss_mse / loss_lpips` 都正常；
   - checkpoint 和 training state 仍能保存。

只有 random latent decode 通过后，才进入真正 `G_fake / DMD` 的 dual-optimizer runner。

## 21. 2026-05-13：`v7-B` random latent decode 第一版实现

本节记录 `v7-B` 相对 `v7-A` 的关键差异，作为后续自查基线。

代码入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
```

新增函数：

```text
_sample_stage3_recon_window(latent_t, recon_num, device)
_latent_window_to_frame_range(start, end)
```

当前规则：

1. 先完整计算 one-step latent：

```text
z_pred = scheduler.step(noise_pred, timestep, noisy_latents, to_final=True)
```

2. 每个 iteration 随机抽一个 latent-time window：

```text
recon_start, recon_end = random window of length stage3_recon_num_latents
```

默认 `stage3_recon_num_latents=2`。

3. Wan latent-time 到 frame range 的映射：

```text
latent 0      -> frame [0, 1)
latent i > 0  -> frame [1 + 4*(i-1), 1 + 4*i)
```

因此：

```text
frame_start = 0                    if recon_start == 0
frame_start = 1 + 4*(start - 1)    otherwise
frame_end   = 1 + 4*(end - 1)
```

举例：

| latent window | decoded frame window | 帧数 |
|---|---:|---:|
| `[0, 2)` | `[0, 5)` | 5 |
| `[1, 3)` | `[1, 9)` | 8 |
| `[5, 7)` | `[17, 25)` | 8 |

4. previous detach 的当前实现：

```text
z_decode = concat(
  z_pred[:, :, :recon_start].detach(),
  z_pred[:, :, recon_start:recon_end]
)
```

也就是说：

- 被选中的 window 保持梯度；
- window 之前的 latent 只作为 Wan decoder 的 causal/context 输入；
- window 之前的 latent 不吃 pixel/LPIPS 梯度；
- window 之后的 latent 不进 decoder，只保留 flow loss。

5. pixel/LPIPS GT 对齐：

```text
x_pred = decode(z_decode)[:, :, frame_start:frame_end]
x_gt   = input_video[:, :, frame_start:frame_end]
```

6. 首帧 4 倍权重：

- 只有当 `frame_start == 0` 时，pixel 首帧权重才使用 `stage3_first_frame_pixel_weight=4.0`；
- 如果随机 window 不包含全局首帧，则首帧权重自动降为 `1.0`；
- LPIPS 同理，当前默认 `stage3_first_frame_lpips_weight=4.0`。

7. debug metadata：

`pipe._stage3_last_losses` 里新增：

```text
selected_latent_start
selected_latent_end
decoded_frame_start
decoded_frame_end
decoded_frames
detached_context_latents
```

如果打开：

```bash
FLASHVSR_STAGE3_DEBUG_LOSS=1
```

日志会打印：

```text
latent_window=[s,e) frame_window=[fs,fe) detached_context_latents=s
```

当前仍需 smoke 验证：

- 随机 window 是否真的跨 step 变化；
- 高 `recon_end` 时 prefix decode 是否会导致显存不可接受；
- detached prefix 是否足以对齐 Wan decoder 的 causal context；
- checkpoint / training_state 保存是否仍正常。

需要人工重点复核的实现决定：

| 决定 | 当前实现 | 为什么这样做 | 风险 |
|---|---|---|---|
| random 2 latent | 连续 window，长度 `stage3_recon_num_latents=2` | 论文说选 2 个 latent，但没在本地代码中找到非连续采样证据；连续 window 更容易和 Wan decoder 时间上下文对齐 | 如果作者实际随机两个非连续 latent，需要改 |
| previous detach | decode `[:recon_end]` prefix，`[:recon_start]` detach | Wan decoder 是 causal decoder，直接只 decode `[recon_start:recon_end]` 可能缺少历史 context | 大 start 时仍要 forward 较长 prefix，可能慢/显存高 |
| frame mapping | latent 0 -> 1 frame，latent i>0 -> 4 frames | 对齐 Wan VAE 首帧不压缩语义 | 如果 Stage2/Stage3 使用 89->85 的流式裁帧语义，frame range 还要再对齐输出有效帧 |
| 首帧权重 | 只有 `frame_start==0` 时 pixel 首帧乘 4 | 避免随机到中间 window 时错误把局部第一帧乘 4 | 如果作者对每个 decode chunk 的局部首帧也特殊加权，需要改 |
| LPIPS 权重 | 首帧跟随 pixel 乘 4 | LPIPS 属于 pixel-space reconstruction loss 的 perceptual 部分 | 2026-05-14 已改代码和 smoke yaml 默认值 |

当前仍不是完整 Stage3：

- 仍未实现 `G_real/G_fake`；
- 仍未实现 DMD loss；
- 仍未实现 dual-optimizer runner；
- 仍未确认 FlashVSR 论文里 “previous ones detached” 是否完全等价于当前 `z_pred prefix detach`。

## 22. 2026-05-13：`v7-B` random latent decode 第二版修正

第一版 random latent decode 直接把 `[0, recon_end)` 作为 Wan decoder 输入，并把 `[0, recon_start)` detach。这个实现语义上最接近“previous ones detached”，但 2 卡 smoke 立刻暴露显存问题：

```text
OutOfMemoryError in pipe.vae.decode(...)
rank0/rank1 tried to allocate ~18.98 GiB
```

根因：

- 如果随机 window 抽到后部 latent，例如 `[20, 22)`；
- 第一版会 decode `[0, 22)` 的完整 prefix；
- 这几乎等价于 decode 整段 89 帧视频；
- Wan decoder 的 temporal cache 和 spatial activation 在 1280x768 下显存不可接受。

因此第二版改成 memory-bounded local context decode：

```text
如果 recon_start == 0:
  decode z_pred[:, :, 0:recon_end]
  pixel/LPIPS 对齐 global frame [0, frame_end)

如果 recon_start > 0:
  decode concat(
    z_pred[:, :, recon_start-1:recon_start].detach(),
    z_pred[:, :, recon_start:recon_end]
  )
  local decoder 第 0 个 latent 只作为 detached context
  pixel/LPIPS 只裁 local frame [1, 1 + selected_frame_count)
  GT 仍裁 global frame [frame_start, frame_end)
```

代码位置：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
_build_stage3_decode_window(...)
Stage3BOneStepReconLoss(...)
```

这个实现的意义：

- 被选中的两个 latent 才吃 pixel/LPIPS 梯度；
- 前一个 latent 只提供 Wan decoder causal context，并且 detach；
- 未选中的其他 latent 仍然通过 full latent flow loss 训练；
- 每次 decode 的 latent 数量从最多 23 个降到最多 3 个；
- `pipe.vae.decode(..., tiled=True)` 进一步降低空间显存。

这个实现和论文的关系：

| 项目 | 论文描述 | 第二版实现 | 状态 |
|---|---|---|---|
| 随机选 2 latent | 每轮随机选 2 个 latent decode | 随机连续 window，默认长度 2 | 基本对齐，仍需确认是否必须连续 |
| previous detached | previous ones detached | 只保留 1 个 immediate previous latent 作为 detached context | 工程近似，不是完整 prefix exact decode |
| 未选 latent | 不走 pixel decode | 只走 full latent flow loss | 对齐 |
| pixel/LPIPS | decode 后与 GT 对齐 | 通过 global frame range 对齐 GT | 对齐 |
| 首帧权重 | 首帧 pixel loss 乘 4 | 只有全局 frame 0 被抽中时乘 4 | 对齐当前理解 |

需要继续验证：

- 2 卡 smoke 是否通过；
- `FLASHVSR_STAGE3_DEBUG_LOSS=1` 下不同 step 是否抽到不同 `latent_window`；
- 中间 window 的 `local_frame_window` 和 `frame_window` 是否匹配；
- 使用 1 个 previous latent 是否足以代表 Wan decoder causal context；
- 如果后续作者确认必须完整 previous prefix，那么需要找更低显存的 decoder-cache 分段实现，而不是回到全 prefix decode。

### 22.1 第二版 smoke 后继续修正：VAE decoder checkpoint

局部上下文版本把单次 decoder 输入从最多 23 个 latent 降到最多 3 个 latent，但 2 卡 smoke 仍然在 Wan decoder 反传阶段 OOM：

```text
OutOfMemoryError in tiled_decode(...)
tried to allocate ~1.93 GiB
```

这次 OOM 不再是完整 prefix decode 的问题，而是 pixel/LPIPS loss 需要从 `x_pred` 反传到 `z_pred`，即使 Wan VAE 参数 frozen，PyTorch 仍要保存 decoder activation 以计算 `d loss / d z_pred`。

修正：

```text
_stage3_decode_with_checkpoint(pipe, z_decode)
```

当前做法：

- VAE decoder 仍然 frozen；
- `z_decode.requires_grad=True` 时，用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` 包住 `pipe.vae.decode(...)`；
- 这样 backward 时重算 VAE decoder forward，少保存 activation；
- 代价是 Stage3 pixel/LPIPS 分支更慢，但这是为了符合论文的 decode pixel loss 所必须付出的显存折中。

这个修正不改变 loss 语义，只改变 activation 保存策略。

### 22.2 VAE decode 输出设备

`pipe.vae.decode(...)` 的外层封装会把 hidden states 放到 CPU list 中逐个处理，decode 返回后输出可能回到 CPU。Stage3 pixel loss 的 GT tensor 已经在当前 rank 的 GPU 上，因此 `v7-B` 在 decode 后显式执行：

```text
x_pred = _stage3_decode_with_checkpoint(...).to(device=pipe.device)
```

这只是设备对齐，不改变数值语义。

## 23. 2026-05-14：Stage3 当前共识与下一轮验收标准

本节记录 2026-05-14 重新复核后的 Stage3 共识。更新：和作者进一步确认后，`previous ones detached` 应按完整 prefix decode 理解，而不是只保留一个 local previous latent。当前 `v7-B` 只保留 full-prefix detach 语义，不再提供 local-context fallback，避免后续实验误跑成非论文版本。

### 23.1 三阶段当前定位

| 阶段 | 当前作用 | 当前稳定状态 |
|---|---|---|
| Stage1 | 训练全注意力 SR teacher，使用 video + image joint；得到高质量但非流式的 SR 母本 | 稳定母本为 `v5.3.5` 89f non-stream LQ projector 对齐版本 |
| Stage2 | 从 Stage1 继续，把 DiT 改成 sparse-causal / block-sparse 形式；LR projector 改为 streaming causal；只用 video data | 当前主线为 `v6.4.1`，解决首帧对齐假设，仍在观察 chunk 跳变 |
| Stage3 | 把 Stage2 sparse-causal student 蒸馏成 one-step streaming 模型；加入 DMD、FM、pixel MSE、LPIPS | `v7-A` 已完成 one-step + decoder reconstruction 基础闭环；`v7-B` 正在修 random latent decode；完整 DMD 仍未实现 |

### 23.2 FlashVSR 论文原文约束

论文 Stage3 相关原文关键句是：

```text
Due to memory constraints, two latents are randomly selected per iteration for decoding,
with previous ones detached from gradients.
```

这句话只明确了三点：

- 每个 iteration 随机选 `2` 个 latent 做 decode；
- pixel / LPIPS 只在 decode 后的 `x_pred` 和 `x_gt` 上算；
- previous ones 需要 detach，不吃 pixel / LPIPS 梯度。

作者确认后的当前理解是：如果随机选到后部两个 latent，应当 forward decode `[0:recon_end]`，但 `[0:recon_start)` detach，不让 previous latents 反传；pixel / LPIPS 只对被选中的 2 个 latent 对应帧算。也就是说，如果选到最后两个 latent，forward 近似 decode 整段视频，但 backward 只经过最后两个 latent。显存控制优先依赖 decoder activation checkpoint / CPU offload / prefix no-grad cache，而不是改语义；训练主线不使用自定义 spatial tile-level decode 逃避显存问题。

### 23.3 当前采用的 random latent decode 解释

当前 `v7-B` 默认采用完整 prefix decode：

```text
如果 recon_start == 0:
  decode z_pred[:, :, 0:recon_end]
  不丢弃前置输出
  如果包含全局 frame 0，则 MSE / LPIPS 首帧权重都乘 4

如果 recon_start > 0:
  decode concat(
    z_pred[:, :, 0:recon_start].detach(),
    z_pred[:, :, recon_start:recon_end]
  )
  previous prefix 只 forward，不反传
  丢弃 previous prefix 对应输出
  只对 recon_start:recon_end 对应的 GT frames 算 MSE / LPIPS
```

代码位置：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
_sample_stage3_recon_window(...)
_build_stage3_decode_window(...)
Stage3BOneStepReconLoss(...)
```

新增参数：

```yaml
stage3_decoder_cpu_offload: true
```

`stage3_decoder_cpu_offload=true` 只作用在 Stage3 Wan decoder reconstruction 分支：在 decoder checkpoint 外层使用 `torch.autograd.graph.save_on_cpu()`，让 decoder 反传所需 activation 尽量放到 CPU。DiT 主干仍保持 Stage2 的 ZeRO2 / gradient checkpoint / block-sparse attention 路径。

当前实现与论文的关系：

| 问题 | 当前决定 | 原因 | 仍需验收 |
|---|---|---|---|
| 随机 2 latent | 连续随机 window，`stage3_recon_num_latents=2` | 连续 window 更符合 Wan decoder 时间上下文，也更易对齐 GT frame range | 如果作者实现是非连续随机，需要改 |
| previous detach | decode 完整 `[0:recon_end]`，其中 `[0:recon_start)` detach | 对齐作者确认后的解释 | 需要 smoke 验证显存 |
| 是否 decode 完整 prefix | 是，`full_prefix` 为默认 | 作者确认 forward 可以 decode 整段，detach 控制 backward | 若 OOM，再考虑 offload / checkpoint / local fallback |
| 首帧权重 | 只有抽到全局 frame 0 时 MSE / LPIPS 都乘 4 | Wan 首 latent 不压缩，只在全局首帧成立；中间 window 的局部第一帧不是全局首帧 | 已把代码和 smoke yaml 默认值改为 `4.0` |

### 23.4 Stage2 / Stage3 的 85 + repeat4 规则

流式 Stage2 / Stage3 统一采用下面规则，避免尾帧 repeat 被错误当作真实监督：

```text
有效视频帧: 85
LQ 输入帧: 85 + repeat(last frame, 4) = 89
LQ projector 输出: 22 个 temporal block
GT VAE latent: 原始 89 帧自然得到 23 个 latent
训练 loss: 只取前 22 个 latent / 对应有效 85 帧
inference: 同样 85 + repeat4，输出后只保留有效 85 帧
```

这个规则当前作为 `v6.4.1 -> v7` 的主线假设。原因是尾部 repeat 的 4 帧是为了满足 streaming projector / FlashVSR 形状要求，不应该作为真实 GT 监督。后续如果尾帧 artifact 仍明显，可以专门做一个对照：是否让 GT / loss 包含 repeat tail。

关于官方形状，当前理解是 FlashVSR 更偏好以 `8n-3` 个有效输入帧为流式单位，再通过 tail repeat 补齐成满足内部处理的长度。以 85 为例：

```text
85 = 8 * 11 - 3
85 + repeat4 = 89 = 8 * 11 + 1
```

这也是当前选择 `85 -> 89 -> 85` 的原因。

### 23.5 Validation 和 random latent decode 是两个验收点

`v7-B validation` 当前是 one-step validation，用来回答：

```text
Stage3 student 一步生成的视频视觉上是否正常。
```

`random latent decode` 是训练 loss 内部的 pixel / LPIPS 分支，用来回答：

```text
训练时是否真的随机抽 2 个 latent decode，并且只让这 2 个 latent 吃 pixel / LPIPS 梯度。
```

因此二者不是同一个验收点：

- validation 通过，只说明 one-step inference 链路能跑；
- random latent decode 还必须通过 debug log / loss metadata 验证；
- 完整 Stage3 还要等 `G_real / G_fake / DMD dual optimizer` 接入。

### 23.6 v7-A / v7-B 当前进度

| 分支 | 当前状态 | 已验证 | 还缺什么 |
|---|---|---|---|
| `v7-A` | one-step + fixed reconstruction prefix smoke | forward / backward / LPIPS / checkpoint / training_state 已通过 | 不含 random latent decode，不含 DMD |
| `v7-B` | one-step + random latent decode + G_fake 参数脚手架 | 代码已实现 full-prefix detach、VAE decode checkpoint、decoder activation CPU offload、LPIPS loss-only cache | random latent decode 尚未最终验收；DMD dual optimizer 尚未实现 |
| `v7-C` | 预留给完整 DMD dual-optimizer runner | 未开始 | 需要独立 runner：student optimizer + G_fake optimizer，G_real frozen |

当前 `v7-B` 仍不是完整 DMD。`stage3_fake_fm_weight=0` 是刻意设置的 guard，因为当前 DiffSynth runner 只有一个 optimizer，不能硬塞 DMD2 的 `G_fake` alternating update。完整 DMD 应新写 `v7-C` 或独立 runner。

### 23.7 下一轮 smoke 验收标准

下一轮 `v7-B` smoke 至少要看到下面结果，才算 random latent decode 通过：

| 验收项 | 预期 |
|---|---|
| `latent_window` | 多个 step 里确实变化，不是固定前缀 |
| `recon_latents` | 恒等于 `2` |
| `decoded_frames` | window `[0,2)` 输出 5 帧；中间 window 输出 8 帧 |
| `detached_context_latents` | `full_prefix` 下应等于 `recon_start`；`recon_start == 0` 时为 0 |
| `decoder_cpu_offload` | 应为 `true` |
| `first_frame_pixel_weight` | 只有抽到全局首帧时为 4.0，否则为 1.0 |
| `first_frame_lpips_weight` | 只有抽到全局首帧时为 4.0，否则为 1.0 |
| checkpoint | 至少保存到 `step-20.safetensors` |
| training_state | 保存正常，不再因为 LPIPS module 破坏 DeepSpeed state |
| 显存 | 不 OOM；优先使用 VAE decode checkpoint / selected window decode / decoder CPU offload；不把自定义 spatial tile-level decode 作为主线 |

如果使用 2 节点做 smoke，其余 6 节点默认继续占卡，避免空卡影响利用率记录。只有明确需要并行实验时，才释放更多卡。

## 24. 2026-05-14 v7-B 对作者解释后的强制对齐项

这一节记录新的约束：后续 `v7-B` 不再保留 local-context 近似作为主线。作者 / 俊豪确认的解释是：

```text
每轮随机选 2 个连续 latent 做 pixel / LPIPS。
如果选中 [recon_start, recon_end)，Wan decoder forward 需要 decode [0, recon_end)。
其中 [0, recon_start) 作为 previous context，只参与 forward，不参与 backward。
也就是 previous prefix detach。
如果随机选到最后两个 latent，forward 等价于 decode 整段有效视频 prefix。
```

### 24.1 当前 v7-B 代码是否做到

当前代码已经按这个方向改：

| 项目 | 当前代码 | 状态 |
|---|---|---|
| 随机 2 latent | `_sample_stage3_recon_window(latent_t, recon_num=2, ...)` | 已做，连续 window |
| full prefix decode | `_stage3_decode_selected_window_full_frame(...)` 按整帧 `[0:recon_end)` 顺序推进 Wan decoder cache | 已做 |
| previous detach | `[0:recon_start)` 只在 `torch.no_grad()` 下推进 VAE decoder cache；`[recon_start:recon_end)` 才带梯度 | 已做 |
| 只监督选中 window | decoder 只返回选中的 2 latent 对应帧，GT 用全局 `frame_start:frame_end` 对齐 | 已做 |
| pixel 首帧权重 | `stage3_first_frame_pixel_weight=4.0`，仅全局 frame0 生效 | 已做 |
| LPIPS 首帧权重 | `stage3_first_frame_lpips_weight=4.0`，仅全局 frame0 生效 | 已做 |
| decoder activation CPU offload | `_stage3_decode_selected_with_checkpoint(..., cpu_offload=True)` 使用 `torch.autograd.graph.save_on_cpu()` | 已做 |
| VAE decode checkpoint | `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` | 已做 |
| LPIPS 权重缓存 | `s3://lxh/models/SR/vgg16-397923af.pth` -> `/mnt/torch_cache/hub/checkpoints/` | 已做 |

当前对应文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-B-Lora-89f-VideoOnly-FakeFM-641Data.sh
```

### 24.2 显存三件套核对

当前 `v7-B 641data smoke` 对齐 Stage2/641 的省显存路径：

| 机制 | 当前状态 | 位置 |
|---|---|---|
| DeepSpeed ZeRO2 | 开启 | `accelerate_zero2_flashvsr_2gpu_noactckpt.yaml` -> `deepspeed_zero2_flashvsr_noactckpt.json` |
| optimizer / param CPU offload | 开启 | `deepspeed_zero2_flashvsr_noactckpt.json` |
| gradient checkpointing | 开启 | `use_gradient_checkpointing: true` |
| gradient checkpointing offload | Stage3 smoke 打开 | `use_gradient_checkpointing_offload: true`，用于给 decoder / LPIPS 反传留显存余量 |
| flash attention | 继承 Stage2 DiT attention 路径 | `wan_video_dit_stage2_v6.py` 里调用 `flash_attention(...)` |
| Stage3 decoder CPU offload | 开启 | `stage3_decoder_cpu_offload: true` |

需要注意：Stage3 pixel / LPIPS 的显存大头不是 DiT LoRA，而是 Wan decoder reconstruction 分支的 activation。作者说 “Stage3 只需要 decode CPU offload，其他不用” 时，对应到当前代码就是只在 `_stage3_decode_selected_with_checkpoint(...)` 外层包 `torch.autograd.graph.save_on_cpu()`，而不是把整套 DiT 或数据退化都搬到 GPU / CPU offload。

2026-05-14 的第一次 full-prefix smoke 证明了一个关键点：如果直接把 `[0:recon_end)` 整段放进 VAE decoder checkpoint，rank0 能出 loss，但 rank1 在更靠后的 window 上会在 decoder backward 重算整段 prefix 时 OOM。正确修法不是退回 local context，而是把 paper 里的 “previous detach” 实现成：

```text
prefix [0:recon_start)      -> no-grad forward，只推进 Wan decoder cache
selected [recon_start:end)  -> 带梯度 decode，计算 pixel / LPIPS
```

这样仍然是 “decode 0 到 end”，但 backward 只需要 selected window 的 activation。当前 `v7-B` 已按这个方向改成 tile 内 prefix no-grad cache + selected grad decode。

第二个显存峰值来自 LPIPS。直接把 2 个 latent 对应的 8 帧 `768x1280` 全部一次性送进 VGG，会让 VGG activation 和 DiT/VAE decoder graph 叠在一起。当前代码改为：

```text
_stage3_lpips_video_loss(...)
  for each selected decoded frame:
      LPIPS(frame_pred, frame_gt) with checkpoint
      saved tensor CPU offload if stage3_decoder_cpu_offload=true
```

语义仍然是对选中 window 的所有帧求 LPIPS 平均；只是把 VGG 的 activation 峰值拆成逐帧 checkpoint，避免一次性压满显存。

2026-05-14 进一步决策：撤掉自定义 spatial tile-level training decode 作为主线。原因是：

- FlashVSR Stage3 的核心语义是 Wan decoder causal prefix + selected latent backward；
- 自定义 tile-level training decode 虽然能降低峰值，但会把问题转成“tile 是否影响 causal decoder / 空间融合”的额外变量；
- 如果 full-frame selected decode 仍 OOM，优先说明 Stage2 DiT 底座、VAE decoder activation、LPIPS activation 或参数/graph 释放方式仍有问题，不应先用 tile 绕开。

当前主线实现改为：

```text
_stage3_decode_selected_with_checkpoint(...)
  checkpoint(
    _stage3_decode_selected_window_full_frame(...)
  )

_stage3_decode_selected_window_full_frame(...)
  clear Wan decoder cache
  if recon_start > 0:
      no_grad decode prefix [0:recon_start)
      只推进 decoder cache，不保留 backward graph
  grad decode selected [recon_start:recon_end)
  返回 selected window 对应的完整空间帧
```

`stage3_decoder_cpu_offload=true` 只包 selected full-frame decode 的 checkpoint 保存张量。LPIPS 仍逐帧串行计算并可使用同一个 CPU offload 设置。

如果这个版本仍然 OOM，下一步不是恢复 tile，而是拆分测量：

| Probe | 目的 |
|---|---|
| Stage2 flow only | 测 DiT / block-sparse / LoRA 底座峰值 |
| Stage3 + MSE decode only | 测 Wan decoder backward 增量 |
| Stage3 + MSE + LPIPS | 测 LPIPS/VGG 增量 |
| 小分辨率 full-frame decode | 验证 full-prefix cache 语义正确性，不被显存干扰 |

如果 Stage2 flow only 本身已经远高于作者 80GB 预期，则说明 block-sparse attention / gradient checkpoint / 参数冻结 / graph 生命周期和作者实现仍未完全对齐。

### 24.3 在线退化不占 GPU 显存的改动

之前在线退化在 DataLoader worker 里可能创建 CUDA context，并在 `cuda:${LOCAL_RANK}` 上产生临时 tensor。这会导致：

- worker 开大时 GPU0 / local rank 相关显存峰值乱跳；
- 训练显存看起来不是模型本身造成，而是数据退化抢显存；
- worker 数量和显存峰值耦合，导致很难稳定调 `dataset_num_workers`。

现在改成：

```text
wanvideo/data/flashvsr/datasets/streaming_dataset.py
_degradation_cuda_device() -> "cpu"
```

含义：

- 在线退化仍然在 DataLoader 里做；
- 但退化模型和中间 tensor 不再上 GPU；
- GPU 显存峰值应主要来自模型 forward/backward；
- 代价是 CPU 数据处理压力会增加，worker 数需要重新通过 smoke 测吞吐。

### 24.4 641 数据入口核对

`v6.4.1 / 641` 的真实数据入口不是直接扫描 Takano 目录，而是：

```yaml
yubari_video_tar_url: "conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/"
takano_video_tar_url: "/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt"
yubari_video_prob: 0.5
takano_video_prob: 0.5
```

`takano_video_train_all.txt` 仍在：

```text
s3://lxh/data/mainfest/takano_video_train_all.txt
```

因此之前的数据问题不是 “641 manifest 丢了”。更准确的判断是：

- CLI `conductor s3 ls conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/` 可以工作；
- Python/fsspec/worker 内的远端对象访问可能和 CLI conductor 使用的凭证链路不同；
- 正式 v7-B 应优先复用 641 的 manifest + Yubari root，而不是临时改成本地 mp4 smoke。

为此新增了 `641data` smoke，专门验证真实数据入口：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-B-Lora-89f-VideoOnly-FakeFM-641Data.sh
```

### 24.5 下一步验收顺序

`v7-B` 必须按下面顺序验收，不再只看 “代码能跑”：

1. `641data` smoke 能读出 Yubari / Takano batch。
2. `stage3_recon_num_latents=2` 保持不降级。
3. LPIPS 开启，VGG16 从 `/mnt/torch_cache/hub/checkpoints/vgg16-397923af.pth` 读取，不在线下载。
4. debug log 里看到多个不同 `latent_window`。
5. 当 `latent_window` 靠后时，`decode_latents=[0,recon_end)`，`detached_context_latents=recon_start`。
6. `decoder_cpu_offload=True`。
7. 能保存 checkpoint 和 training_state。
8. 观察显存峰值是否接近作者所说的 80GB 级别；如果仍然 170GB 级别，需要继续查是否有 teacher / VAE / LPIPS / 数据退化额外占显存。

## 25. 2026-05-14 `v7-B` 收口结论

`v7-B` 已经完成进入 `v7-C` 前的核心验证：

| 项目 | 结论 |
|---|---|
| 数据退化 | 已改为 CPU 在线退化，不再让 DataLoader worker 创建 GPU degradation context |
| DataLoader worker | `runner.py` 的 worker init 已改为 CPU-only seed，不再调用 CUDA |
| random latent decode | 主线为 full-frame selected decode，不再用自定义 spatial tile 逃避显存 |
| previous detach | prefix 用 no-grad / cache 语义，selected window 带梯度算 pixel / LPIPS |
| 首帧权重 | `stage3_first_frame_pixel_weight=4.0`，`stage3_first_frame_lpips_weight=4.0` |
| 显存主因 | Wan decoder selected-window backward 是 120GB 级峰值主因 |
| block sparse | CUDA extension 可用；不可用时会直接报错，不会静默走 dense fallback |
| `641` 规则 | 输入正常 `89` 帧，内部 `22` latent，监督 / 输出 `85` 有效帧 |

当前速度建议：

| 设置 | 用途 |
|---|---|
| `worker=2 + GC offload off + decoder CPU offload on` | 速度优先，当前最快已完成 smoke |
| `worker=2 + GC offload on + decoder CPU offload on` | 稳定 fallback |
| `worker=8` | worker CUDA context 已修复，但 2GPU smoke 没有证明更快，暂不默认 |

因此 `v7-B` 不再继续扩大功能。下一阶段是 `v7-C`：完整 DMD / dual-optimizer runner。

### 25.1 C 阶段前新增决策

2026-05-14 用户进一步确认以下规则：

- `641` 测试结果较好，因此 Stage3 后续先完全沿用 `641` 的输入规则：
  - 正常取 `89` 帧输入；
  - 不默认把测试集预先构造成 `85 + repeat4`；
  - 内部仍按 `89 -> 22 latent -> 85 有效输出 / 监督` 执行。
- Stage2 pretrain 不再使用旧 `3k` checkpoint，后续 Stage3/C 固定使用 `641 step-6000` 结果作为 teacher/student 初始化来源，不默认替换成更高 step。
- 如果论文明确 `G_real` 来自 Stage1/full-attention teacher，则不再怀疑，`v7-C` 按 Stage1 teacher 写。
- CPU degradation 需要额外做固定参数 ablation：
  - 同一批 5 个视频；
  - 同一份 degradation params；
  - 分别在 CPU/GPU apply；
  - 导出结果到桌面肉眼检查；
  - 该 ablation 只验证设备迁移是否改变退化结果，不改变训练默认逻辑。
- Wan decoder selected-window backward 目前约带来 `+40GB` 量级增量。继续优化方向必须保持功能语义：
  - 不加 tile；
  - 不改变 selected-window / prefix-detach 语义；
  - 优先查不该留在 autograd graph 的对象；
  - 串行执行 frozen teacher / fake model / decoder / LPIPS；
  - 不让 `G_real`、LPIPS、Wan VAE decoder 进入 optimizer 或保留多余梯度。

## 26. `v7-C` 计划：完整 DMD dual-optimizer runner

### 26.1 为什么必须单独写 runner

DMD2 和 OSEDiff 的参考代码都不是单 optimizer 逻辑：

- DMD2:
  - 参考代码：`/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py`
  - `optimizer_generator` 更新 one-step generator；
  - `optimizer_guidance` 更新 guidance / fake model；
  - `dfake_gen_update_ratio` 控制 generator 是否在当前 step 更新；
  - generator turn 和 guidance turn 是两次独立 forward/backward。
- OSEDiff:
  - 参考代码：`/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/OSEDiff/train_osediff.py`
  - `optimizer` 更新 SR generator；
  - `optimizer_reg` 更新 regularization / distribution matching 分支；
  - reconstruction loss 和 distribution matching / diff loss 分开 step。

当前 DiffSynth `launch_training_task` 只有一个 optimizer，硬塞 `G_fake` 会导致：

- `G_fake` 参数和 student 参数混进一个 optimizer；
- 无法按 DMD2 的 alternating update 节奏更新；
- checkpoint / resume 也无法保存两个 optimizer 和两个 scheduler；
- 容易让 `G_real` 或 `G_fake` 被错误挂进 student graph。

所以 `v7-C` 必须新写 Stage3 专用 runner，不继续复用通用 runner。

### 26.2 模型角色

| 角色 | 初始化 | 是否训练 | 用途 |
|---|---|---|---|
| Student / `G_one` | Stage2 `v6.4.1` checkpoint | 训练 LoRA + `lq_proj_in` | one-step 输出，承接 pixel / LPIPS / DMD 梯度 |
| `G_real` | Stage1 full-attention teacher 或指定 teacher checkpoint | 冻结 | DMD 中的 real score / teacher reference |
| `G_fake` | 从 `G_real` 或指定初始化复制 | 训练 | DMD2 式 fake score / guidance model |
| Wan VAE decoder | Wan pretrained | 冻结 | decode selected latent window，算 pixel / LPIPS |
| LPIPS / VGG | pretrained VGG | 冻结 | perceptual reconstruction loss |

默认不训练 Wan VAE、不训练 LPIPS、不训练 `G_real`。

### 26.3 单 step 执行顺序

`v7-C` 的单 step 应显式串行化，避免所有大模块同时留在计算图里：

1. 读取 video batch，规则沿用 `641`：
   - 输入 `89` 帧；
   - LQ projector / Stage2 streaming path 得到 `22` latent positions；
   - GT 取有效 `85` 帧，Wan VAE target 得到 `22` latents。
2. Student forward：
   - one-step 生成 `z_pred`；
   - 保留 student graph，因为 student 要被更新。
3. Reconstruction branch：
   - 随机采样连续 `2` 个 latent window；
   - prefix `[0:recon_start)` no-grad 推 decoder cache；
   - selected `[recon_start:recon_end)` 带梯度 decode；
   - pixel MSE 和 LPIPS 只对 selected window 对应帧算；
   - 如果 window 包含全局首帧，pixel / LPIPS 首帧权重为 `4.0`。
4. DMD generator branch：
   - 对 `z_pred` 加噪；
   - `G_real` frozen/no-grad 给 real prediction；
   - `G_fake` 在 no-grad 或 detach 语义下给 fake prediction；
   - 按 DMD2 公式构造 student 的 distribution matching loss。
5. Student optimizer step：
   - 只更新 Student LoRA + `lq_proj_in`。
6. G_fake branch：
   - detach student 生成结果；
   - 用 fake / real 数据训练 `G_fake`；
   - 单独 `optimizer_fake.step()`。
7. 保存：
   - student checkpoint；
   - `G_fake` checkpoint；
   - student optimizer / scheduler；
   - fake optimizer / scheduler；
   - random states；
   - 当前 update ratio / step。

### 26.4 第一版 `v7-C` 验收顺序

不要一次把所有 loss 打开。按下面顺序验收：

| 阶段 | 开启内容 | 验收标准 |
|---|---|---|
| C0 | runner scaffold，student optimizer + fake optimizer，但 `fake_loss=0` | 两个 optimizer/scheduler 能保存和 resume |
| C1 | Student reconstruction loss | loss、random window、首帧权重、checkpoint 正常 |
| C2 | 加 `G_real` frozen forward | `G_real.requires_grad=False`，显存可控，无梯度流入 |
| C3 | 加 `G_fake` forward，但不更新 | DMD loss 数值可记录，`G_fake` 无 optimizer step |
| C4 | 打开 `G_fake` optimizer | fake 参数发生变化，student/fake 两套 grad norm 独立 |
| C5 | 打开完整 DMD loss | 能跑 20 step，保存完整 training state |

每个阶段都必须记录：

- `loss_recon_mse`
- `loss_recon_lpips`
- `loss_dmd`
- `loss_fake`
- student grad norm
- fake grad norm
- `G_real` trainable 参数数必须为 `0`
- 当前 latent window
- 当前有效输出帧数必须为 `85`

### 26.5 当前不确定项

| 问题 | 当前处理 |
|---|---|
| `G_real` 用 Stage1 还是 Stage2 teacher | 先按论文直觉使用 Stage1 full-attention teacher；如果结果不稳定，再试 Stage2 teacher |
| DMD timestep 范围 | 参考 DMD2/OSEDiff 先使用中间 timestep 区间，具体值需要独立 sweep |
| `G_fake` 初始化 | 先从 `G_real` 复制；不从 student 复制 |
| update ratio | 先参考 DMD2 的 alternating update，默认每 step 更新 fake，每 `dfake_gen_update_ratio` 控制 student/DMD 更新频率，具体以首个 smoke 日志核对 |
| 多机 DeepSpeed 双 optimizer | 需要单独验证；不能假设当前 runner 自动支持 |

### 26.6 下一步执行

下一步先写 `v7-C0`：

- 新建独立训练入口，不改 `v7-B`；
- 只搭双 optimizer / save / resume 骨架；
- 不打开真正 DMD；
- 2GPU smoke 看到两个 optimizer state 都能保存和恢复后，再进入 `C1/C2`。

## 27. 2026-05-15：`v7-C` 回看，防止偏离论文原始计划

这一节专门用于回看最初读 FlashVSR / DMD2 / OSEDiff 论文时留下的要求，核对当前 `v7-C` 代码是否发生语义偏移。结论是：`v7-C5` 已经完成 DMD student gradient 的关键链路验证，但还不能称为完整 FlashVSR Stage3。

### 27.1 当前已经对齐的部分

| 原始要求 | 当前 `v7-C` 状态 | 证据 |
|---|---|---|
| Stage3 是 one-step student | 已对齐 | `Stage3BOneStepReconLoss` 里 student 一步得到 `z_pred`，不是 50-step validation 逻辑 |
| Student 从 Stage2 初始化 | 已对齐 | C5 smoke 使用 Stage2 `v6.4.1 step-6000` 作为 student |
| `G_real` 是 frozen teacher | 已对齐 | `G_real` 使用 no-grad / frozen / eval，不进入 optimizer |
| `G_real` 需要 full-attention teacher 路径 | 已基本对齐 | `dense_full` 分支验证走 `flash_attn_2`，不是 torch dense fallback |
| pixel / LPIPS 必须 decode 后计算 | 已在 `v7-B/C` 路径接入 | `z_pred -> Wan decoder -> x_pred -> MSE / LPIPS` |
| 每次随机选 2 个 latent decode | 已按当前理解接入 | random window + selected decode，首帧权重逻辑存在 |
| DMD student loss 需要 real/fake direction | C5 已接入 | `p_real = z - real_x0`，`p_fake = z - fake_x0`，`loss = 0.5 * mse(z, (z - grad).detach())` |
| LPIPS 是 frozen loss-only module | 已对齐 | 不进入 Deepspeed 训练模块 |

当前 smoke 证据：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data_20260515_v7c5_smoke_r3_9f_256x256_nogather
```

关键日志：

```text
[stage3c_train] epoch=0 step=1 loss=1.703014 student=1.606964 fake_skeleton=0.00000100 fake_scale=0.000003 real_probe=0.304703 fake_probe=0.147766 dmd_student=0.096049 dmd_grad=1.029076
```

这说明 student loss、real/fake x0 direction、backward、optimizer step、checkpoint 保存已经打通。

### 27.2 当前发生偏移或仍未完成的部分

| 原始要求 | 当前偏移 | 必须修正 |
|---|---|---|
| `G_fake` 是 `G_real` 的 copy，并学习 fake latent distribution | 当前 C5 的 `G_fake` 仍是 frozen probe，不是真正 trainable fake score model | 下一步必须把 placeholder fake optimizer 替换为真实 `G_fake` optimizer |
| DMD2 需要 dual optimizer / alternating update | 当前 runner 有 fake optimizer 骨架，但还没有真正更新完整 `G_fake` 模型 | 需要实现 `optimizer_student` 和 `optimizer_fake` 两套真实 optimizer / scheduler / state 保存 |
| `L_FM(z_pred, G_fake)` 必须存在 | 当前 `stage3_fake_fm_weight` 仍被 guard 禁止打开 | 需要按 DMD2 `compute_loss_fake` 逻辑实现 fake FM / fake score update |
| `G_fake` 初始化应从 `G_real` copy | 当前 smoke 为了验证链路用过 Stage2 fake probe | 正式版要恢复为 Stage1 full-attention copy，除非后续明确做 ablation |
| DMD timestep / noise 权重应对齐 DMD2 或 FlashVSR 作者设定 | 当前只是用现有 scheduler 随机采样，尚未严格 sweep | 需要把 timestep range、weighting、fake update ratio 变成显式配置并记录 |
| C5 smoke 使用 `NO_GATHER_LOG` | 这是 smoke 工程规避，不是训练语义 | 正式训练前要确认日志 gather 或 metric 汇总不会卡住 |
| C5 smoke 用 9f / 256x256 | 只能证明链路，不证明 89f 正式显存与速度 | 需要 89f smoke 和至少短训练验证 |

最关键的偏移是 `G_fake`。原始论文/DMD2 代码要求 `G_fake` 是会训练的 fake score model；当前 C5 只是用 frozen fake probe 给 student 提供一个 fake direction。因此 C5 应被命名为：

```text
DMD student-gradient path validation
```

不能命名为：

```text
complete DMD2 / complete FlashVSR Stage3
```

### 27.3 为什么这个偏移重要

DMD2 代码里 `compute_loss_fake(...)` 明确让 `fake_unet` 追随当前 generator 产生的 fake distribution。如果 `G_fake` 不更新，会出现两个问题：

| 问题 | 后果 |
|---|---|
| fake score 不能跟上 student 分布变化 | DMD gradient 很快失真 |
| student 只看 frozen real/fake 差异 | 可能变成固定 teacher-direction regularizer，而不是 distribution matching |

所以后续不能只调 `stage3_dmd_weight` 或 reconstruction loss。完整 Stage3 的下一步一定是把 `G_fake` 真正训练起来。

### 27.4 下一步必须按这个顺序做

| 顺序 | 目标 | 验收 |
|---|---|---|
| C6 | `G_fake` 从 `G_real` copy 初始化，并成为真实 trainable model | `fake_trainable_params > 0`，`G_real trainable params = 0` |
| C7 | 实现 `optimizer_fake` 和 `scheduler_fake` 的保存 / resume | checkpoint 里有 fake model + fake optimizer state |
| C8 | 实现 `compute_loss_fake` / `L_FM(z_pred, G_fake)` | fake grad norm 非 0，student graph 不被 fake update 污染 |
| C9 | 实现 alternating update ratio | 日志能区分 student update step 和 fake update step |
| C10 | 89f official-size smoke | 看到 random window、pixel/LPIPS、DMD、fake loss 都正常 |

在 C6-C10 做完前，所有文档和汇报中应把当前 `v7-C5` 描述为“DMD 方向接入验证”，而不是完整第三阶段。

### 27.5 仍需保留的论文原始约束

后续不要因为工程压力丢掉下面几条：

| 约束 | 原因 |
|---|---|
| Stage3 video-only | FlashVSR Stage3 明确沿用 Stage2 video setting |
| `G_real` 使用 Stage1 full-attention teacher | 论文公式中的 real distribution teacher 不是 sparse-causal Stage2 |
| `G_fake` 从 `G_real` copy 后训练 | DMD2 的 fake score model 必须跟随 fake distribution |
| pixel / LPIPS 只 decode selected latent window | 显存约束下不应全片 pixel loss |
| selected window 的 previous context detach | 保留 Wan decoder 因果上下文，但不让 prefix 反传 |
| 首帧 pixel / LPIPS 权重为 4 | Wan VAE 首 latent 不压缩，Stage3 pixel-level loss 需要补偿 |
| validation 用 one-step | Stage3 目标是 one-step model，不再沿用 Stage1/2 50-step validation |

### 27.6 当前代码入口

当前 `v7-C` 相关代码入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py
```

当前 C4/C5 smoke 配置与脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c4_lora_89f_videoonly_dmdprobe_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C4-Lora-89f-VideoOnly-DMDProbe-641Data.sh
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C5-Lora-89f-VideoOnly-DMDLoss-641Data.sh
```

当前专门记录 C runner 的文档：

```text
doc/flashvsr_stage3_v7c_dmd_runner_plan_20260514.md
```

### 27.7 2026-05-15 C6 更新：`G_fake` 已从占位符推进到可训练模型

C6 已经完成下一步推进：

```text
C5: frozen G_fake probe + DMD student gradient
C6: trainable G_fake + fake FM loss + fake optimizer state
```

新增文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh
```

通过 smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r2_9f_256x256
```

关键结果：

```text
G_fake trainable_params=570961408
fake_loss=0.00162535
dmd_student=0.092463
dmd_grad=1.008987
fake_model_is_full_stage3=True
fake_model trainable keys=488
```

当前 C6 相对原始计划的状态：

| 原始要求 | C6 状态 |
|---|---|
| `G_fake` 不能是 frozen probe | 已修正，C6 是完整 `FlashVSRStage3BTrainingModule` |
| `G_fake` 需要独立 optimizer | 已接入 `fake_optimizer` |
| `G_fake` 需要保存 / resume 状态 | 已保存 trainable state + fake optimizer / scheduler |
| fake branch 需要学习 fake latent distribution | 已接入 fake FM loss，使用 `z_pred.detach()` 作为 fake latent |
| DMD student loss 需要 real/fake direction | 已保留 C5 逻辑 |

仍未完成：

- fake update 还不是 DMD2 严格的多次 fake substep；
- C6 只通过 9f/256 smoke，还没通过 89f 正式尺寸；
- `G_fake` 当前 standalone + 手动 all-reduce gradient，没有进入 Deepspeed ZeRO；
- 正式训练前还需要验证长训 save/resume。

因此当前正确命名是：

```text
v7-C6: trainable G_fake + fake FM + DMD student loss smoke
```

不是最终：

```text
complete Stage3 DMD2 production training
```

### 27.8 2026-05-15 `v7-D1`：作者权重 + streaming validation 的正式化分支

`v7-D1` 是在 `v7-D stable` 跑通之后单独复制出的正式分支，目标不是改动 DMD 主公式，而是把 validation、日志命名和外部扫描入口整理到可长期使用的状态。

| 项目 | `v7-D stable` | `v7-D1` |
|---|---|---|
| 训练主逻辑 | author weights + trainable `G_fake` | 保持一致 |
| `stage3_fake_fm_weight` | `1.0` | `1.0` |
| `stage3_dmd_weight` | `1.0` | `1.0` |
| 首帧 pixel / LPIPS 权重 | `4.0 / 4.0` | `4.0 / 4.0` |
| DMD spike guard | `skip@5` | 保持开启 |
| 内部 validation 数量 | `1` | `2` |
| validation 推理 | one-step | one-step streaming KV-cache |
| validation 来源 | 固定样本 | 优先从 rank0 训练 batch 缓存，避免启动前单独采样卡住 |
| W&B | offline | offline，且 run dir 内每小时同步 |
| 日志命名 | 仍有 `v7-B/v7-C` 残留 | 改为 `v7-D1` |

`v7-D1` 的 validation 明确使用：

```text
pipe.infer_from_lq_streaming(
  num_inference_steps=1,
  tiled=false,
  topk_ratio=stage2_topk_ratio,
  kv_ratio=3.0
)
```

这和最终 Stage3 推理语义一致：一阶段/二阶段的 50-step dense validation 不再作为 Stage3 的训练内 validation。代价是 validation 只在 rank0 执行，checkpoint 边界其他 rank 需要等待，因此内部 validation 只保留 2 个视频；固定 10-video 测试集仍通过外部扫描脚本执行。

新增入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d1_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d1_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D1-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-641Data-OfflineWandb.sh
wanvideo/model_inference/flashvsr/history/run_stage3_v7_d1_scan89_step500_incremental.sh
```

### 27.9 2026-05-15 `v7-D1` 暂停，当前主线回退 `v7-D stable`

`v7-D1` 的 2GPU smoke 能进入训练循环，且 validation gating 能避免 step1 过早验证；但 48GPU 下 step2 计算异常重，显存与耗时明显劣于 `v7-D stable`。因此 `v7-D1` 不作为当前正式主线继续推进。

当前正式训练回退到：

```text
train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return
```

对应文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d_stable.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_stable_authorweights_offline.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh
```

已确认：

- W&B offline 写入实验目录 `wandb/`，后台同步 tmux 为 `wandb_sync_v7d_stable`。
- 48GPU 已出 loss：
  - `loss=1.073758 flow=0.019114 mse=0.020162 lpips=0.517241`
  - `loss=0.193740 flow=0.193740`
  - `loss=0.075488 flow=0.075488`
- 当前显存与利用率更接近可长期训练状态，rank0 节点多数 GPU 在 `97-100%`。

后续如果要恢复 `v7-D1` 的 2-video streaming validation，应作为单独分支重新做性能剖析，不再直接替换当前稳定主线。

### 27.10 2026-05-15 `v7-D2` 首帧权重强制验证

为了关闭“random latent decode 抽到首帧时首帧权重是否真的生效”的问题，新增一个只用于验证的 debug override：

```text
FLASHVSR_STAGE3_FORCE_RECON_START=0
```

该变量只影响 `_sample_stage3_recon_window()` 的采样起点。默认不设置时，`v7-D2` 仍然使用随机 latent window，不改变正式训练语义。

验证入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d2_firstframe_weightcheck.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D2-FirstFrameWeightCheck.sh
```

远端验证目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d2_firstframe_weightcheck_20260515_firstframe_weightcheck_clean2
```

关键日志：

```text
stage3_v7_d2_firstframe_weightcheck force_recon_start=0
expect latent_window=[0,2) first_frame_pixel_weight=4.0 first_frame_lpips_weight=4.0
[stage3_v7_b_loss] loss=0.466705 flow=0.107180 mse=0.007805 lpips=0.175860 latent_window=[0,2) frame_window=[0,5) decode_latents=[0,2) local_frame_window=[0,5) recon_latents=2 decoded_frames=5 context_mode=full_prefix decoder_cpu_offload=True detached_context_latents=0 first_frame_pixel_weight=4.0 first_frame_lpips_weight=4.0 compute_z_pred=True need_reconstruction=True
```

结论：

- 当选中全局首 latent 时，`MSE` 与 `LPIPS` 均按首帧权重 `4.0` 进入 loss。
- `latent_window=[0,2)` 对应 `frame_window=[0,5)`，其中首 latent 对应首帧，第二个 latent 对应后续 4 帧。
- `detached_context_latents=0` 符合首帧 window 没有 prefix context 的预期。
- 该验证只关闭首帧权重问题，不关闭 `G_real/G_fake` 是否严格等价 Stage1 teacher 的问题。

额外工程注意：

- 8n 上第一次验证 OOM 的原因是占卡脚本没有遵守外层 `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7`，实际占用了 GPU0/1。
- 以后如果需要“部分 GPU 跑 smoke、其余 GPU 占卡”，不要直接假设 `gpu_stress_tc.sh` 会继承外层 `CUDA_VISIBLE_DEVICES`；应先用 `nvidia-smi --query-compute-apps=pid,used_memory` 确认实际占用。

### 27.11 2026-05-15 `v7-D3`：Stage1 teacher wrapper 对齐版

`v7-D3` 是在同事确认 `v7-D2` teacher wrapper 问题必须修之后复制的新线。它不热改正在跑的 `v7-D2`，而是把 DMD 里的 teacher/copy 语义推进到更接近论文设定：

```text
G_real/G_fake:
  Stage1 v5.3.5 checkpoint weights
  dense_full attention
  lq_proj_temporal_mode=nonstreaming_aligned

student/G_one:
  Stage3 one-step streaming

DMD comparison:
  teacher 23 positions -> explicit trim_front_to_match -> student 22 positions
```

新增入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_lora_89f_videoonly_authorweights_stage1teacher_aligned_offlinewandb.yaml
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_teacher_aligned.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-TeacherAligned.sh
```

共享底座改动：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
```

- `FlashVSRStage2Pipeline.from_pretrained()` 增加 `lq_proj_temporal_mode`，默认 `streaming`，旧线行为不变。
- D3 构造 `G_real/G_fake` 时传 `nonstreaming_aligned`。
- `flashvsr_stage2_model_fn()` 增加 `lq_latent_alignment=trim_front_to_match`，用于显式裁掉 teacher 多出来的 warm-up temporal position。
- full-attention teacher 仍使用 `dense_full` attention mode；该路径在 Stage2 patched attention 中仍调用 FlashAttention，不是手写 dense attention。

6a smoke 验证已完成：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_smoke2
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_mid10_smoke
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_tail20_smoke
```

验证结果：

| 检查项 | D3 smoke 结果 |
|---|---|
| `G_real/G_fake` projector mode | 日志确认 `lq_proj_temporal_mode=nonstreaming_aligned` |
| teacher/student latent alignment | 日志确认 `trim_front_to_match`，`88320 -> 84480` tokens，即 23 -> 22 temporal positions |
| DMD shared noisy point | real/fake 均使用同一 `timestep=368.000000`，`shared_point=True` |
| DMD zero-diff sanity | 初始 `dmd_student=0.000000`，`dmd_grad=0.000000` |
| fake optimizer path | `fake_update=1`，`fake_loss=0.06189094` |
| 首帧 window | `[0,2) -> [0,5)`，首帧权重 `4.0/4.0` |
| 中间 window | `[10,12) -> [37,45)`，首帧权重 `1.0/1.0` |
| 尾部 window | `[20,22) -> [77,85)`，不越界，首帧权重 `1.0/1.0` |

第一次 D3 smoke 曾暴露一个有效问题：`real_probe_every` 在 step0 通过 generic `probe_model.forward()` 触发，该路径没有携带 D3 alignment，导致：

```text
Stage2 requires exact LQ/DiT token match ... x=84480, lq=88320 ... alignment=exact
```

已在 `FlashVSRStage3BTrainingModule.forward()` 中补齐 teacher alignment，后续 smoke 通过。

当前结论：

- `v7-D3` 已关闭 `v7-D2` 的 teacher projector temporal-mode 主问题；
- `v7-D2` 应继续被标注为 shared-noise DMD ablation，而不是严格 Stage1 teacher/copy 复现；
- `v7-D3` 已通过 2GPU smoke、alignment、DMD zero-diff、fake update、首/中/尾 decode window 检查；
- 尚未启动 `v7-D3` 48GPU 长训；正式启动前可选择再做一次完整 Stage1 module deterministic forward 对比，但当前 projector/token-level 的关键风险已被修掉。

### 27.12 2026-05-15 D3 二次补丁：guard、temporal map、optimizer ownership

同事复核第 18 节后指出 D3 仍需补几个工程和验证细节。已补：

- `stage3_fake_checkpoint` guard：
  - D3 如果启用 `stage3_fake_fm_weight` 或 `stage3_dmd_weight`，parse 阶段必须有 `stage3_fake_checkpoint`；
  - 防止只用 yaml 启动时因 checkpoint 为 `null` 退回 scalar fake placeholder。
- temporal map 日志：
  - `[stage3_teacher_align]` 现在会打印：

```text
teacher_tokens_before=88320
teacher_positions_before=23
drop_teacher_positions=[0,1)
keep_teacher_positions=[1,23)
student_positions=[0,22)
note=teacher_position0_is_nonstreaming_aligned_warmup
```

- optimizer ownership debug：
  - 新增 `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=1`；
  - 第一轮 optimizer 前后打印 student / fake / real checksum 是否变化。
- snapshot：
  - D3 release/smoke snapshot 都包含 `train_flashvsr_stage2_v6_4_lora.py`；
  - smoke snapshot 也包含实际 attention 文件 `diffsynth/models/wan_video_dit_stage2_v6_1.py`。
- validation meta：
  - 改为 `stage3_v7_d3_one_step_direct_decode`；
  - 额外写 `validation_mode_detail=not_streaming_kvcache_validation`，避免被误读为 streaming validation。

6a patchcheck smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_patchcheck_smoke
```

关键结果：

```text
[stage3_teacher_align] ... teacher_tokens_before=88320 teacher_positions_before=23 drop_teacher_positions=[0,1) keep_teacher_positions=[1,23) student_positions=[0,22) note=teacher_position0_is_nonstreaming_aligned_warmup
[stage3d3_optimizer_ownership] student_changed=True fake_changed=True real_changed=False ... fake_update=1
[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
```

这次补丁后，D3 已比 27.11 多关闭两项风险：

- `trim_front_to_match` 不再只是 token 数对齐日志，而是明确记录了 teacher warm-up position 0 被丢弃、teacher 1..22 对齐 student 0..21；
- optimizer ownership 至少在 smoke 的参数 checksum 级别确认了 student/fake 变化、frozen real 不变。

### 27.13 2026-05-15 `v7-D3.1`：干净正式线并替换 D2 长训

按用户要求，`v7-D3` 作为带审查/验证痕迹的已验证版本保留不删，另复制干净正式线：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_1_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb.yaml
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-Clean-OfflineWandb.sh
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-1-TeacherAligned-Clean.sh
```

D3.1 保留 D3 的正式训练语义：

- `G_real/G_fake` 使用 Stage1 v5.3.5 checkpoint；
- `G_real/G_fake` wrapper 使用 `lq_proj_temporal_mode=nonstreaming_aligned`；
- student 仍是 Stage3 streaming one-step；
- DMD 入口显式 `trim_front_to_match`；
- full-attention teacher 仍走 FlashAttention kernel；
- `stage3_fake_checkpoint` guard 保留。

D3.1 移除/关闭审查项：

- 从 D3.1 train 文件移除 optimizer checksum debug 代码；
- release/smoke 脚本默认：

```text
FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0
FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0
```

6a clean smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean_20260515_v7d31_clean_smoke
```

结果：

```text
[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
stage3_teacher_align count = 0
stage3d3_optimizer_ownership count = 0
```

随后停止旧 `v7-D2` 48GPU：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb_20260515_v7d2_sharednoise_48gpu
```

6 节点均确认：

```text
PIDS_D2=0
```

启动 `v7-D3.1` 48GPU：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu
```

启动参数：

```text
MASTER_ADDR=240.12.138.137
MASTER_PORT=29531
RUN_TS_OVERRIDE=20260515_v7d31_clean_48gpu
FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0
FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0
```

当前 48GPU 状态：

```text
step=1 loss=0.854109 student=0.722192 fake_loss=0.13191710 fake_update=1 real_probe=0.093054 fake_probe=0.101467 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
```

已生成：

```text
output/training_state/step-1
output/training_state/step-2
output/validation/step-1
output/validation/step-2
```

主节点 run.log 确认：

```text
stage3_teacher_align count = 0
optimizer_ownership count = 0
```

W&B：

- 主节点 package tmux：`wandb_package_v7d31_stage1teacher_clean`
- 6a sync tmux：`wandb_sync_from_s3_v7d31_stage1teacher_clean`
- S3 包：

```text
s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu.tar.gz
```

已手动触发一次 package + sync，6a 成功同步：

```text
offline-run-20260515_011822-6nu0uhd7
offline-run-20260515_091237-60budl7y
```

仍需保留的边界：

- D3.1 修正了 D2 最大的 teacher wrapper 偏差；
- D3/D3.1 已通过 shape、temporal map、optimizer ownership smoke；
- 但还没有做完整 Stage1 teacher deterministic forward 数值等价证明。
