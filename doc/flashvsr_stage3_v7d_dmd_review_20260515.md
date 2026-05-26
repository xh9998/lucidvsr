# FlashVSR Stage3 v7-D DMD 复现审阅记录

日期：2026-05-15

本文档用于接手当前 FlashVSR / LucidVSR Stage3 工作时，对正在跑的 `v7-D stable return` 训练代码做一次 DMD 复现语义审阅。重点不是重新设计 validation。当前 validation 使用较轻的一步 full decode 是合理的工程取舍；把 validation 改成完整流式推理会明显变慢，本文不把它列为待修问题。

## 1. 本次审阅依据

### 1.1 计划与工作日志

- `doc/flashvsr_stage3_dmd_plan_20260511.md`
- `doc/FLASHVSR_WORKLOG.md`
- `doc/CODEX_HANDOFF_20260515.md`

### 1.2 论文位置

- FlashVSR paper：
  `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/2510.12747v1.pdf`
- DMD2 paper：
  `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/2405.14867v2.pdf`
- OSEDiff paper：
  `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/OSEDiff/2406.08177v3.pdf`

### 1.3 参考代码位置

DMD2：

- `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_guidance.py`
- `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py`

OSEDiff：

- `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/OSEDiff/train_osediff.py`
- `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/OSEDiff/osediff.py`

注意：早期计划文档里有一些路径写成 `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/...` 或 `/code/OSEDiff/...`。本次实际检查到的本地路径是 `/mac_code/DMD2/...` 和 `/mac_code/OSEDiff/...`。

## 2. 当前 v7-D 训练代码位置

主代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d_stable.py
```

当前 48GPU stable return 对应配置和启动脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_stable_authorweights_offline.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh
```

当前版本大体结构：

- `G_one`：从 Stage2 sparse-causal checkpoint 初始化的一步 student。
- `G_real`：加载 Stage1 full-attention teacher 权重，冻结。
- `G_fake`：从 Stage1 teacher copy 初始化，可训练，单独 fake optimizer。
- student loss：flow + pixel MSE + LPIPS。
- DMD student loss：用 `G_real` / `G_fake` 的预测差构造 latent-space gradient。
- fake loss：训练 `G_fake` 追随当前 generated fake latent distribution。

这说明 `v7-D` 已经不是早期 smoke scaffold，而是具备完整 Stage3 DMD 训练所需的大部分工程部件。

## 3. FlashVSR Stage3 目标应是什么

FlashVSR 论文 Stage3 的目标是把 Stage2 sparse-causal video DiT 蒸馏成 one-step streaming VSR model。论文公式可整理为：

```text
L =
  L_DMD(z_pred, G_one, G_real, G_fake)
  + L_FM(z_pred, G_fake)
  + ||x_pred - x_gt||^2
  + lambda * L_LPIPS(x_pred, x_gt)
```

其中：

- `G_one`：从 Stage2 sparse-causal DiT 继续训练出来的一步生成器。
- `G_real`：Stage1 full-attention DiT teacher，冻结。
- `G_fake`：`G_real` 的 copy，用来学习 `G_one` 产生的 fake latent distribution。
- `z_pred`：`G_one` 一步预测出来的 clean latent。
- `x_pred`：`z_pred` 经 Wan decoder 得到的图像/视频帧。

论文还说明因为显存限制，每次 iteration 只随机选两个 latent 做 decode 和 pixel/LPIPS，未选中的 latent 不走重建 decode。

当前 `v7-D` 在 one-step student、random selected latent decode、pixel/LPIPS、trainable `G_fake`、dual optimizer 这些工程点上基本是朝这个目标实现的。

## 4. 当前 v7-D 已经对齐的部分

### 4.1 One-step student

`Stage3BOneStepReconLoss` 中会从 input latent 和统一 timestep 出发做 one-step prediction：

```text
z_pred = scheduler.step(noise_pred, timestep, inputs["latents"], to_final=True)
```

这符合 Stage3 `G_one` 的基本形式。

### 4.2 Pixel / LPIPS 重建

当前代码随机采样 latent window，只对选中的 window 做 Wan decoder 和 pixel/LPIPS。这个设计符合 FlashVSR 论文的显存约束描述。

### 4.3 `G_real` / `G_fake` 双模型

当前代码已经构造：

- frozen `real_model`
- trainable `fake_model`
- fake optimizer / fake scheduler
- fake extra state 保存与 resume

这比早期 `v7-C` frozen probe 更接近正式 DMD 训练。

### 4.4 fake loss 没有错误地反传到 student

`_stage3c_fake_fm_loss` 内部对 `clean_latents` 做了 detach。也就是说 fake loss 主要训练 `G_fake`，不会把 fake critic 自己的 regression loss 错误地直接反传到 `G_one`。这点是对的。

## 5. 需要修正的问题 1：DMD real/fake score 必须共享同一个 noisy latent 和 timestep

### 5.1 参考实现怎么做

DMD2 的 `compute_distribution_matching_loss` 逻辑是：

1. 从 generator 得到 clean latent。
2. 采样同一个 `noise` 和同一个 `timestep`。
3. 构造同一个 `noisy_latents`。
4. 分别把同一个 `noisy_latents, timestep` 喂给 `real_unet` 和 `fake_unet`。
5. 得到 `pred_real_image` 和 `pred_fake_image`。
6. 用两者差值构造 distribution matching gradient。

关键点不是公式长什么样，而是 real score 和 fake score 必须是在同一个点上比较：

```text
s_real(noisy_latents, t) - s_fake(noisy_latents, t)
```

如果 real/fake 各自用不同的 noise 或不同的 timestep，那么这个差值就不是同一个 distribution point 上的 score difference，DMD 语义会被破坏。

OSEDiff 的 `distribution_matching_loss` 也是同一类思路：在同一个 latent/noise/timestep 点上比较 fixed model 和 trainable regularizer 的预测差。

### 5.2 当前 v7-D 怎么做

当前 `_maybe_run_stage3c_dmd_student_loss` 中分别调用：

```text
real_x0 = _stage3c_probe_predict_x0(real_model, ...)
fake_x0 = _stage3c_probe_predict_x0(fake_probe_model, ...)
```

而 `_stage3c_probe_predict_x0` 内部会自己采样 timestep 和 noise。

这意味着：

- `G_real` 看到的是一份 `t_real, noise_real, noisy_latents_real`
- `G_fake` 看到的是另一份 `t_fake, noise_fake, noisy_latents_fake`

然后代码再计算：

```text
p_real = z_detached - real_x0
p_fake = z_detached - fake_x0
dmd_grad = (p_real - p_fake) / mean(abs(p_real))
```

这个差值看起来像 DMD2，但 real/fake 并不是在同一个 noisy point 上估计 score。因此它不是严格的 DMD distribution matching gradient。

### 5.3 为什么这是实质性问题

DMD 的核心是比较：

```text
当前 generator 产生的样本，在 real distribution 下应该往哪里走
当前 generator 产生的样本，在 fake distribution 下认为已经在哪里
```

这个比较必须固定输入点。否则差值里混入了 timestep/noise 采样差异，梯度会变成：

```text
s_real(noisy_latents_a, t_a) - s_fake(noisy_latents_b, t_b)
```

这不是 DMD2/OSEDiff 的训练目标，也不再是 FlashVSR 公式里 `L_DMD(z_pred, G_one, G_real, G_fake)` 应该表达的东西。

### 5.4 建议修改

应把 probe 函数拆成两层：

1. 外层在 DMD student loss 中采样一次 `timestep/noise/noisy_latents`。
2. 内层 probe 只接收已经构造好的 `noisy_latents` 和 `timestep`，不再自己采样。

伪代码：

```python
clean_latents = student_z_pred
noise = torch.randn_like(clean_latents)
t = sample_timestep(...)
noisy_latents = scheduler.add_noise(clean_latents.detach(), noise, t)

with torch.no_grad():
    real_x0 = predict_x0(real_model, noisy_latents, t, data)
    fake_x0 = predict_x0(fake_model, noisy_latents, t, data)

p_real = clean_latents.detach() - real_x0
p_fake = clean_latents.detach() - fake_x0
dmd_grad = (p_real - p_fake) / normalizer
loss_dmd = 0.5 * mse(clean_latents, (clean_latents - dmd_grad).detach())
```

注意：

- `real_x0` 和 `fake_x0` 应 no-grad。
- `clean_latents` 在 loss 左边保留 student gradient。
- target `(clean_latents - dmd_grad)` detach。
- fake model 在 DMD student loss 中不应被这个 loss 更新；fake model 应由自己的 fake loss 更新。

这是当前最优先需要修的 DMD 语义问题。

## 6. 需要修正的问题 2：`stage3_fake_update_ratio` 语义和 DMD2 不一致

### 6.1 DMD2 参考逻辑

DMD2 的训练代码中有：

```text
COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
```

含义是：

- fake score model 基本每 step 更新。
- generator 只在某些 step 更新。
- 如果 `dfake_gen_update_ratio=5`，大致就是 fake 更新 5 次，generator 更新 1 次。

DMD2 论文强调 fake diffusion critic 必须及时追上 generator 输出分布，否则 DMD 会不稳定。

### 6.2 当前 v7-D 逻辑

当前 `_stage3c_fake_fm_loss` 里：

```text
if global_step % update_ratio != 0:
    skip fake update
```

这个语义是：

- `update_ratio=1`：fake 每 step 更新。
- `update_ratio=5`：fake 每 5 step 更新一次。

这和 DMD2 的 two time-scale update rule 是反的。

### 6.3 当前 run 的影响

当前 stable 配置里 `stage3_fake_update_ratio=1`，所以这次 7d stable return 没有因为 ratio 反向而少更新 fake。

但从代码语义看，如果后续把 ratio 调成 5，期望“fake 多训”，实际会变成“fake 少训”。这会误导后续实验。

### 6.4 建议修改

建议不要继续复用当前 `stage3_fake_update_ratio` 的含义，最好显式改名或重写：

- `fake_updates_per_student_update`
- `student_update_every_n_fake_steps`
- 或者保留旧参数但在文档和代码里明确它不是 DMD2 ratio。

如果要严格按 DMD2，需要 runner 支持 fake substeps，至少有两种实现：

方案 A：

- 每个 batch 先更新 fake 若干次。
- 再更新 student 一次。
- 成本更高，但语义最直观。

方案 B：

- 每 step 都更新 fake。
- student 每 N step 更新一次。
- 训练日志要明确哪些 step 没有 student update。

考虑 video 训练成本，短期可以先保持 ratio=1，但不能把 ratio=5 写成“DMD2 fake 多步更新”。

## 7. 需要确认的问题 3：`G_real/G_fake` 是否完全等价 Stage1 full-attention teacher

FlashVSR 论文要求：

- `G_real` 是 Stage1 full-attention DiT teacher。
- `G_fake` 是 `G_real` 的 copy。

当前 v7-D 的实现是：

- 用 `FlashVSRStage3BTrainingModule` 构造 `real_model` / `fake_model`。
- 加载 Stage1 checkpoint。
- attention mode 配成 `dense_full`。

这大方向是对的。但还需要确认一个细节：Stage1 v5.3.5 的 LQ projector / temporal mode 是否和当前 Stage3B module 实例化出来的 projector 完全一致。

如果当前模块虽然加载了 Stage1 权重，但 forward 语义走的是 Stage2/Stage3 pipeline 的 LQ embedder 或 temporal mode，那么 `G_real/G_fake` 就不是严格的 Stage1 teacher copy，而是“Stage1 权重 + Stage3 wrapper”。这会影响 DMD teacher score 的语义。

建议做一个小检查：

1. 找出 Stage1 v5.3.5 训练时的 projector temporal mode。
2. 找出当前 `real_model/fake_model` 初始化时 projector temporal mode。
3. 用同一个输入跑 Stage1 原始 module 和当前 `real_model` dense_full wrapper，比较输出差异。

如果输出一致或差异只来自无关 wrapper，则可以关闭这个风险。如果不一致，应优先让 `G_real/G_fake` 的 forward 语义回到 Stage1 teacher。

## 8. `L_FM(z_pred, G_fake)` 和当前 trainable fake diffusion critic 训练的差别

这一点容易混淆，需要单独说明。

### 8.1 FlashVSR 论文公式的字面含义

FlashVSR 论文写：

```text
L_FM(z_pred, G_fake)
```

从公式字面看，它表示：在 `z_pred` 这个由 `G_one` 生成的 latent 上，还要有一个 flow-matching 类 loss，并且这个 loss 和 `G_fake` 有关。

但论文没有展开到代码级细节，例如：

- `G_fake` 的输入是不是 `z_pred` 加噪后的 latent。
- timestep 怎么采样。
- target 是 velocity、noise、x0 还是 flow。
- 这个 loss 是更新 `G_fake`，还是也更新 `G_one`。
- 如果同时有 DMD loss，`L_FM` 在优化顺序中怎么和 `L_DMD` 交替。

所以只看 FlashVSR 公式，不能唯一推出具体代码。

### 8.2 DMD2/OSEDiff 里的 fake critic 训练是什么

DMD2 里的 fake model 不是普通 teacher，也不是 frozen reference。它是一个 trainable fake score model，任务是学习当前 generator 产生的 fake latent distribution。

DMD2 的 fake loss 大致是：

```text
latents = generator_output.detach()
noise, t = sample()
noisy_latents = add_noise(latents, noise, t)
fake_pred = fake_unet(noisy_latents, t)
loss_fake = mse(fake_pred, noise_or_target)
```

关键语义：

- `latents` detach，所以 fake loss 不更新 generator。
- fake model 单独更新。
- fake model 学的是 generator 当前输出分布上的 denoising / score。
- fake model 训练得越跟得上，DMD gradient 越稳定。

OSEDiff 也有类似分工：

- generator / one-step model 用 reconstruction + VSD/DMD 类 loss 更新。
- trainable regularizer / fake score model 用自己的 diffusion loss 更新。

### 8.3 当前 v7-D 的 fake loss 更像哪一种

当前 `_stage3c_fake_fm_loss` 做的是 trainable fake diffusion critic 训练：

- 输入是 student 产生的 `clean_latents`。
- `clean_latents` detach。
- 对它加噪、采 timestep。
- 让 `G_fake` 预测 flow/noise-like target。
- 用这个 loss 更新 `G_fake`。

这和 DMD2 的 `compute_loss_fake` 语义接近。

它不是一个直接作用在 `G_one` 上的 supervised FM loss。也就是说，当前实现中的 fake FM loss 主要职责是训练 `G_fake` 这个 critic，而不是直接让 student 拟合某个 flow target。

### 8.4 这和 `L_FM(z_pred, G_fake)` 的差别

差别可以这样理解：

#### 解释 A：把 `L_FM(z_pred, G_fake)` 理解成 fake critic 自身训练

如果 FlashVSR 作者把 `L_FM(z_pred, G_fake)` 简写为“在 `z_pred` 分布上训练 `G_fake` 的 flow matching loss”，那当前 v7-D 的 fake loss 是合理对应。

这种理解下：

- `L_DMD` 更新 `G_one`。
- `L_FM` 更新 `G_fake`。
- pixel/LPIPS 更新 `G_one`。
- `G_fake` 通过自己的 FM loss 追随 `G_one` 的 fake distribution。

这和 DMD2/OSEDiff 最接近。

#### 解释 B：把 `L_FM(z_pred, G_fake)` 理解成 student 的额外 FM regularization

也可能有人按公式字面理解为：`L_FM` 也应该对 `G_one` 产生梯度，要求 `z_pred` 在 `G_fake` 定义的 flow field 下满足某种目标。

这种理解下，当前 v7-D 就少了一个直接更新 student 的 FM 项。当前 fake loss detach 了 `clean_latents`，所以它不会通过 `L_FM` 直接推 student。

但是这个解释缺少 DMD2/OSEDiff 代码支持，而且容易把 fake critic training 和 generator training 混在一起。DMD 系方法通常不希望 fake score model 的监督 loss 直接反传进 generator；generator 应通过 real/fake score difference 的 DMD gradient 更新。

### 8.5 本次审阅建议采用的解释

建议把当前 v7-D 的 `_stage3c_fake_fm_loss` 明确命名和文档化为：

```text
G_fake fake-distribution flow-matching / denoising loss
```

而不是笼统叫 student 的 `L_FM`。

更准确的总目标可以写成：

```text
student update:
  L_student =
    L_recon_flow
    + L_pixel
    + 2 * L_LPIPS
    + w_dmd * L_DMD(z_pred; G_real, G_fake)

fake update:
  L_fake =
    w_fake * L_FM_fake(z_pred.detach(); G_fake)
```

这样既保留 FlashVSR 公式的精神，也更贴近 DMD2/OSEDiff 的代码级实现。

如果前同事认为 FlashVSR 论文里的 `L_FM(z_pred, G_fake)` 必须直接更新 student，那么需要他指出对应的作者代码或公式推导。仅从 DMD2/OSEDiff 参考代码看，当前“detach z_pred 后训练 fake critic”的做法更稳。

## 9. 当前 v7-D 是否已经达到复现目标

结论：当前 `v7-D stable return` 是一个很接近目标的工程版本，但还不能称为严格 FlashVSR Stage3 / DMD 复现。

原因：

1. DMD student loss 中 `G_real/G_fake` 没有共享同一个 noisy latent 和 timestep，这是当前最明确的语义错误。
2. fake update ratio 参数语义和 DMD2 相反，虽然当前 ratio=1 暂时没踩中，但后续实验容易误用。
3. `G_real/G_fake` 是否完全等价 Stage1 full-attention teacher 还需要做 projector / wrapper 一致性确认。
4. `L_FM(z_pred, Gfake)` 在论文中未展开，当前实现更接近 DMD2 fake critic training，而不是一个直接更新 student 的 FM 项；这需要在文档中明确，避免后续汇报时混淆。

不列为问题：

- 当前 validation 不改流式推理。轻量 one-step validation 对长训监控是合理选择，完整流式推理可以作为单独离线评估脚本，而不应拖慢训练保存点。

## 10. 建议下一步

### 10.1 不建议直接改正在跑的快照

当前 7d run 可以继续作为工程稳定性实验，但不要把它当最终 paper-aligned run。修复应开新版本，例如：

```text
v7-D2-shared-dmd-noise
```

### 10.2 优先修复 DMD shared-noise

改动范围：

- `_stage3c_probe_predict_x0`
- `_maybe_run_stage3c_dmd_student_loss`
- 如果 logging-only probe 还要保留，也应同步改成 shared timestep/noise，避免日志误导。

验收标准：

- real/fake probe 日志明确打印或记录同一个 timestep source。
- 小尺寸 smoke 能 backward。
- DMD loss 数值无 NaN。
- fake model 不被 DMD student loss 更新。
- student 能从 DMD student loss 得到梯度。

### 10.3 梳理 fake update ratio

短期：

- 当前正式 run 继续用 `stage3_fake_update_ratio=1`。
- 文档中明确这个参数不是 DMD2 的 fake多步更新。

中期：

- 增加真正的 fake substep runner。
- 或者改成 student 每 N step 更新一次，fake 每 step 更新。

### 10.4 验证 Stage1 teacher 等价性

做一个很小的 deterministic 对比：

- 同一输入。
- 同一 timestep/noise。
- Stage1 原始 teacher forward。
- 当前 `real_model` dense_full wrapper forward。
- 比较 x0/noise/flow prediction。

如果一致，这个风险关闭。如果不一致，需要让 `G_real/G_fake` 回到 Stage1 原始 forward 语义。

## 11. 给前一位同事的自查问题

请重点自查下面几件事：

1. `_stage3c_probe_predict_x0` 为什么要在 real/fake 两次调用中各自采样 timestep/noise？这是否有意为之？如果有，请给出和 DMD2/OSEDiff 不同的理论依据。
2. `stage3_fake_update_ratio=5` 在当前代码中到底表示 fake 多训还是 fake 少训？这是否和计划文档中引用 DMD2 的 two time-scale update rule 一致？
3. 当前 `G_real/G_fake` 的 forward 是否和 Stage1 v5.3.5 full-attention teacher 完全一致？尤其是 LQ projector temporal mode 和 Stage2/Stage3 wrapper 是否改变了语义。
4. 当前 `_stage3c_fake_fm_loss` 是否应该在日志和文档里改名为 `fake critic FM loss`，避免被误解为直接更新 student 的 `L_FM`？
5. 如果坚持当前实现已经等价 FlashVSR 公式，请补充作者代码、论文附录或 DMD 推导证据。

我的判断是：当前版本已经完成了大部分困难的工程接线，但 DMD shared-noise 这一处必须修，否则不能称为严格 DMD 复现。

## 12. 上一位实现者回应：W&B 网络与 v7-D 代码审阅结论

记录时间：2026-05-15

以下内容是对前面 review 的逐条回应。写法按“审阅意见 / 我的判断 / 后续动作”组织，方便后续改代码前先对齐语义。

### 12.1 W&B 网络不通不是训练代码问题

**审阅/现象：**

`v7-D stable return` 使用 W&B offline。训练节点上在线 `wandb sync` 反复失败。

**我的排查结论：**

这不是 `v7-D` 训练代码逻辑导致的，也不是 W&B key 本身无效。实际网络行为是：

- 6 个 48 卡训练节点访问 `https://api.wandb.ai` 超时；
- `wandb verify` 在训练节点失败；
- `curl -I https://api.wandb.ai` 在训练节点超时；
- `6ai5mpi47f` 可以访问 W&B，并且手动 `wandb sync` 成功；
- W&B offline 文件必须写入 artifacts 实验目录，否则只写 `/mnt/task_runtime/lucidvsr/wandb` 会随任务环境丢失。

**已经采取的工程方案：**

当前不再让 48 卡训练节点直连 W&B，而是改成 relay：

- `t5qdtykjsw`：把 `${RUN_DIR}/wandb` 和 `/mnt/task_runtime/lucidvsr/wandb/offline-run-*` mirror 后打包上传 S3；
- `6ai5mpi47f`：每小时从 S3 拉包并执行 `wandb sync`；
- 中转包：
  `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return.tar.gz`

新增脚本：

```text
wanvideo/model_training/flashvsr/scripts/package_wandb_offline_to_s3_loop.sh
wanvideo/model_training/flashvsr/scripts/sync_wandb_offline_from_s3_loop.sh
```

当前已验证：t5 上传成功，6a 下载成功，两个 offline run 均已 `wandb sync ... done`。

**后续建议：**

这套 relay 应作为当前 48 卡 Stage3 的默认 W&B 策略。除非确认训练节点能稳定直连 `api.wandb.ai`，否则不要再把在线 W&B 卡顿误判为训练代码卡住。

### 12.2 关于“DMD real/fake 必须共享同一个 noisy latent 和 timestep”

**审阅意见：**

当前 `_maybe_run_stage3c_dmd_student_loss()` 里分别调用两次 `_stage3c_probe_predict_x0()`；而 `_stage3c_probe_predict_x0()` 内部自己采样 `timestep` 和 `noise`。因此 `G_real` 和 `G_fake` 比较的不是同一个 noisy point。

**我的判断：成立，是目前最明确的 DMD 语义问题。**

我重新看了当前代码：

```text
_maybe_run_stage3c_dmd_student_loss()
  real_x0 = _stage3c_probe_predict_x0(real_model, ...)
  fake_x0 = _stage3c_probe_predict_x0(fake_probe_model, ...)

_stage3c_probe_predict_x0()
  timestep_id = torch.randint(...)
  noise = torch.randn_like(clean_latents)
  noisy_latents = scheduler.add_noise(clean_latents, noise, timestep)
```

这意味着 `real_x0` 和 `fake_x0` 的输入点不一致。DMD2 / OSEDiff 参考代码的核心是同一个 `noisy_latents` 和同一个 `timesteps` 同时送进 real/fake 模型，再比较 score / x0 方向。这个 review 判断是合理的。

**影响：**

当前 `v7-D stable return` 可以作为工程稳定性 run，但不能严格称为 paper-aligned DMD run。DMD 梯度里混入了 real/fake 各自随机噪声差异，可能引入额外方差甚至错误方向。

**建议动作：**

开新版本修，不直接热改正在跑的目录。建议命名：

```text
v7-D2-shared-dmd-noise
```

改法：

1. 新增一个只接收 `noisy_latents` / `timestep` 的 probe forward，例如：
   `stage3c_probe_predict_x0_from_noisy(...)`。
2. 在 `_maybe_run_stage3c_dmd_student_loss()` 外层只采样一次：
   `noise, timestep, noisy_latents`。
3. `G_real` 和 `G_fake` 都用同一个 `noisy_latents/timestep`。
4. `G_real/G_fake` 在 DMD student loss 中全程 no-grad。
5. 只让 student 的 `clean_latents` 接收 DMD loss 梯度。

验收必须包含：打印或保存一条 debug，证明 real/fake 使用同一个 `timestep` 和同一个 `noisy_latents` seed/source。

### 12.3 关于 `stage3_fake_update_ratio` 语义

**审阅意见：**

DMD2 中 `dfake_gen_update_ratio` 的含义更接近 fake 多更新、generator 少更新；当前代码的 `stage3_fake_update_ratio` 是 fake 每 N step 更新一次，语义相反。

**我的判断：成立，但当前 run 没实际踩雷。**

当前 stable 配置是：

```yaml
stage3_fake_update_ratio: 1
stage3_fake_fm_weight: 1.0
stage3_dmd_weight: 1.0
```

所以现在 fake 每步更新，没有因为 ratio 反向而少训。但如果后续把 `stage3_fake_update_ratio=5` 当成“fake 多训”来用，就会完全相反，变成 fake 5 step 才训一次。

**建议动作：**

短期：

- 保持当前正式 run 的 `stage3_fake_update_ratio=1`；
- 文档里明确这个参数当前含义是 `fake_update_every_n_student_steps`，不是 DMD2 的 `dfake_gen_update_ratio`。

中期：

- 如果要严格 DMD2 two-time-scale，应新增明确参数，例如：
  `stage3_fake_updates_per_student_step`；
- runner 里支持 fake substeps，或者支持 fake 每步更新、student 每 N 步更新。

### 12.4 关于 `G_real/G_fake` 是否等价 Stage1 full-attention teacher

**审阅意见：**

虽然当前 `stage3_real_attention_mode=dense_full`、`stage3_fake_attention_mode=dense_full`，但 `G_real/G_fake` 是用 Stage3B wrapper 构造的。需要确认它们的 LQ projector / temporal mode 是否真的等价 Stage1 v5.3.5 teacher。

**我的判断：这是合理风险，但还不能直接判错。**

当前代码确实是：

```text
real_model = FlashVSRStage3BTrainingModule(... stage2_attention_mode=args.stage3_real_attention_mode, ...)
fake_model = FlashVSRStage3BTrainingModule(... stage2_attention_mode=args.stage3_fake_attention_mode, ...)
```

并且配置里：

```yaml
stage3_real_attention_mode: dense_full
stage3_fake_attention_mode: dense_full
lq_proj_layer_num: 1
lq_proj_scale: 1.0
zero_init_lq_proj_in: false
```

这说明 attention 方向上是按 Stage1 dense/full teacher 的目标在做。但 wrapper 是否完全复刻 v5.3.5 的 non-streaming aligned projector、89 帧输入、aligned23 语义，还需要 deterministic 对比。

**建议动作：**

做一个小测试，不急着改正式代码：

1. 固定同一条输入视频、同一个 prompt、同一个 timestep/noise；
2. 用 Stage1 v5.3.5 原训练/推理 module forward 一次；
3. 用当前 `real_model dense_full wrapper` forward 一次；
4. 比较输出 latent / noise_pred 的 max error / mean error。

如果误差只来自 dtype 或 wrapper 无关差异，可以关闭这个风险；如果差异明显，需要优先修 `G_real/G_fake` 的 teacher wrapper 语义。

### 12.5 关于 `L_FM(z_pred, G_fake)` 的解释

**审阅意见：**

当前 `_stage3c_fake_fm_loss()` detach 了 `clean_latents`，所以 fake FM 只训练 `G_fake`，不直接更新 student。需要明确这是否符合 FlashVSR 公式。

**我的判断：当前实现更贴近 DMD2 fake critic training，不能简单说错。**

DMD2/OSEDiff 代码里 fake model 的 loss 本来就是在 generator output detach 后训练 fake score/critic。student/generator 主要通过 DMD score difference 得到梯度，而不是让 fake critic 的 diffusion loss 直接回传进 student。

所以当前：

```text
fake_clean_latents = clean_latents.detach()
```

我认为是合理的。更准确的说法是：

```text
student update:
  flow + pixel + LPIPS + DMD(real, fake)

fake update:
  fake-distribution flow matching on z_pred.detach()
```

**需要改的是命名和文档，不一定是代码。**

建议把日志和文档里的 `fake_fm` 明确写成：

```text
G_fake fake-distribution FM loss
```

避免被误解成“直接更新 student 的 FM 项”。如果有人认为 FlashVSR 公式要求这个 FM 也更新 student，需要提供作者代码或更明确公式推导。

### 12.6 关于 validation 不走完整流式推理

**审阅意见：**

本文档开头认为当前 validation 使用较轻的一步 full decode 是合理工程取舍。

**我的判断：同意。**

训练内 validation 的职责是健康监控，不应该让 rank0 每个 checkpoint 做很重的完整 streaming KV-cache 推理，导致其他 47 rank 长时间等待。完整流式推理应该通过外部测试脚本做。

当前 `v7-D stable` 先保留轻量 validation 是合理的。后续如果要 paper-aligned evaluation，需要单独用外部 inference 脚本跑，不要塞进训练保存路径。

### 12.7 我认为这份 review 中最重要的优先级

按优先级排序：

1. **必须修：DMD real/fake shared noisy latent/timestep。** 这是最明确的语义错误。
2. **必须文档澄清：`stage3_fake_update_ratio` 当前不是 DMD2 的 fake 多步 ratio。** 当前 ratio=1 不伤害，但后续容易误用。
3. **需要验证：`G_real/G_fake` wrapper 是否等价 Stage1 v5.3.5 teacher。** 这不是立即判错，但必须做 deterministic 对比。
4. **建议澄清命名：fake FM 是 fake critic training，不是直接 student FM。**
5. **W&B 不要继续在训练节点硬修在线同步。** 当前 relay 方案已经验证可用。

### 12.8 给后续改代码的建议

不要直接在当前正在跑的 `v7-D stable return` 上继续打补丁。更稳妥路线是：

```text
v7-D stable return：保留，作为工程跑通和对照实验。
v7-D2-shared-dmd-noise：新开文件/配置，专门修 DMD shared noisy point。
```

`v7-D2` 最小改动范围：

```text
_stage3c_probe_predict_x0
_maybe_run_stage3c_dmd_probe
_maybe_run_stage3c_dmd_student_loss
少量 debug / log 字段
```

不应该同时改 validation、W&B、数据集、G_real/G_fake 结构，否则很难判断 shared-noise 修复是否有效。

## 13. 对 `v7-D2 shared-noise` 新改动的复核

记录时间：2026-05-15

本节复核另一位同事提交的 `v7-D2` 新线。复核文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh
```

### 13.1 文件与基础检查

已确认三份文件存在。本机检查：

```text
/usr/bin/python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py
bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh
```

均无报错。

### 13.2 shared-noise DMD 改法是否成立

**结论：核心改法基本成立。**

`v7-D2` 在 `_stage3c_probe_predict_x0()` 中新增了：

```text
dmd_point
return_dmd_point
```

当 `dmd_point is None` 时只采样一次：

```text
timestep_id
timestep
noise
noisy_latents
```

并保存到：

```text
dmd_point = {
  "timestep": timestep.detach(),
  "timestep_id": timestep_id.detach(),
  "noise": noise.detach(),
  "noisy_latents": noisy_latents.detach(),
}
```

`_maybe_run_stage3c_dmd_student_loss()` 现在先跑：

```text
real_x0, dmd_point = _stage3c_probe_predict_x0(... return_dmd_point=True)
```

再把同一个 `dmd_point` 传给 fake：

```text
fake_x0 = _stage3c_probe_predict_x0(... dmd_point=dmd_point)
```

`_maybe_run_stage3c_dmd_probe()` 也做了同样修改。

这解决了我前面指出的最大语义问题：`G_real/G_fake` 不再各自采样不同 timestep/noise，而是在同一个 noisy latent point 上比较 x0/score 方向。这个方向和 DMD2/OSEDiff 的 shared noisy point 语义一致。

仍需 smoke 验证：

- 开 `FLASHVSR_STAGE3C_DMD_DEBUG=1` 跑极小步，确认 real/fake 日志中的 timestep 完全一致；
- 确认 `dmd_grad_norm` 不 NaN；
- 确认 `G_fake` 不会从 DMD student loss 获得梯度，只由 fake FM loss 更新。

### 13.3 fake update 跳过逻辑是否正确

**结论：基本正确。**

新参数：

```text
stage3_fake_update_every_n_steps
```

语义明确为：每 N 个 student step 更新一次 `G_fake`。旧参数：

```text
stage3_fake_update_ratio
```

保留为 legacy alias，并且配置里当前仍为 `1`。

训练循环中现在有：

```text
fake_did_update = fake_loss is not None
...
if fake_did_update:
    _average_stage3c_fake_gradients(fake_model)
...
if fake_did_update:
    fake_optimizer.step()
...
if fake_did_update:
    fake_scheduler.step()
```

所以当 fake loss 被 schedule 跳过时，fake optimizer 和 fake scheduler 不会 step。这一点和同事描述一致。

注意：当前正式配置 `stage3_fake_update_every_n_steps=1`，因此每步都更新 fake，和现有 `v7-D stable` 行为一致。这个参数不是 DMD2 的 `dfake_gen_update_ratio`，文档必须继续强调。

### 13.4 仍然存在的风险 1：D2 启动脚本只负责 t5 打包，不负责 6a 上传

`FlashVSR-Stage3-Release-48GPU-v7-D2-...sh` 中只启动了：

```text
wandb_package_v7d2_sharednoise
```

也就是训练主节点打包上传 S3。它没有自动在 `6ai5mpi47f` 上启动：

```text
sync_wandb_offline_from_s3_loop.sh
```

这不是训练语义 bug，但如果直接用这个脚本启动 48 卡，W&B offline 包会到 S3，未必会自动上传到 W&B，除非另行在 6a 挂同步 tmux。

后续启动 `v7-D2` 前需要明确做两件事：

1. t5 主节点启动 `wandb_package_v7d2_sharednoise`；
2. 6a 另起 `wandb_sync_from_s3_v7d2_sharednoise`，指向同一个 S3 tar 包。

否则会出现“训练正常、W&B 不更新”的误判。

### 13.5 仍然存在的风险 2：launch 脚本自身 PATH 可能缺 awscli

D2 launch 脚本开头目前是：

```bash
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/bin:/miniforge/bin:$PATH"
```

但我们刚刚在 W&B relay 中遇到过实际问题：新 tmux 环境里 `/usr/local/bin/conductor` 会调用 `aws`，而 `aws` 可能在：

```text
/root/.local/share/pipx/venvs/awscli/bin/aws
```

如果启动脚本运行在干净 tmux，`conductor s3 cp` 下载 manifest / VGG / checkpoint 时可能因为找不到 `aws` 或 Notary 环境失败。

D2 脚本虽然设置了 `NOTARY_CONFIG_FILE`，但 PATH 建议补成和新 relay 脚本一致：

```bash
export PATH="/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH"
```

这个是工程稳健性修复，不影响模型语义。

### 13.6 仍未关闭的风险：Stage1 teacher wrapper 等价性

D2 没有解决这个问题，也不应该在这次一起改。当前仍需后续单独验证：

```text
Stage1 v5.3.5 原始 teacher forward
vs
v7-D/D2 real_model dense_full wrapper forward
```

需要同输入、同 timestep/noise 下比较输出误差。只有这个验证通过，才能说 `G_real/G_fake` 真正是 Stage1 full-attention teacher/copy，而不是 Stage1 权重套 Stage3 wrapper 后产生了行为偏差。

### 13.7 总体判断

对同事回复的判断：

- “没动 v7-D1、没热改正在跑的 v7-D stable return”：从新增文件看基本可信。
- “DMD student loss / probe 共享同一个 dmd_point”：代码核查后成立。
- “fake update 跳过时 optimizer/scheduler 不 step”：代码核查后成立。
- “legacy ratio 避免误解”：方向正确，但仍要在后续汇报和配置里避免把它叫 DMD2 ratio。
- “可以下一步 smoke”：同意，但 smoke 前建议先补 launch PATH，并同步准备 6a W&B relay。

我的建议：

1. 不要直接上 48GPU。
2. 先远端 `py_compile` + `bash -n`。
3. 再做 2GPU smoke，打开一次 `FLASHVSR_STAGE3C_DMD_DEBUG=1` 验证 real/fake timestep 一致。
4. smoke 通过后，再考虑 `v7-D2` 48GPU。

## 14. 给后续同事：本机 `watch` tmux 恢复方法

如果本机 `watch` 会话丢失，不要手动乱开单个窗口。当前正确流程已经写入：

```text
doc/CODEX_HANDOFF_20260515.md
```

关键点：

- 使用 `bash -lc` 执行恢复脚本；
- 不要在 zsh 中用普通字符串拆词；
- 不要假设 zsh 数组是 0-based；
- 8 个窗口顺序必须是：
  `t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk 8nh48ucn8b 6ai5mpi47f`；
- 每个窗口都是先 `bolt task ssh <machine>`，进入远端后再运行 `watch -n 1 nvidia-smi`。

本次已按上述方式恢复，`tmux list-windows -t watch` 能看到 8 个窗口，且 `watch:0` / `watch:7` 均能抓到远端 `nvidia-smi` 输出。

## 15. 给后续同事：v7-D2 首帧权重验证结果

用户要求单独验证：

```text
当 Stage3 random latent decode 抽到全局首帧时，
pixel MSE 和 LPIPS 的首帧 4 倍权重是否真的进入 loss。
```

我新增了一个只用于调试的环境变量：

```text
FLASHVSR_STAGE3_FORCE_RECON_START=0
```

它只覆盖 `_sample_stage3_recon_window()` 的起点，默认不设置时不影响正式训练随机采样。

新增文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d2_firstframe_weightcheck.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D2-FirstFrameWeightCheck.sh
```

验证机器：

```text
8nh48ucn8b
```

远端实验目录：

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

- v7-D2 代码在强制抽到首帧时，确实把 `first_frame_pixel_weight=4.0` 和 `first_frame_lpips_weight=4.0` 传入了重建 loss。
- `latent_window=[0,2)` 到 `frame_window=[0,5)` 的映射符合当前 Stage3 约定。
- 该验证不能替代 teacher wrapper 等价性验证；同事继续做 `G_real/G_fake` 和 Stage1 teacher 的 forward 等价检查。

踩坑记录：

- 第一次 smoke OOM 是占卡脚本造成的，不是 v7-D2 首帧权重逻辑造成的。
- `gpu_stress_tc.sh` 在 8n 上没有按预期限制到外层 `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7`，实际占了 GPU0/1。
- 后续做部分 GPU smoke 时必须先查：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

确认空卡后再启动训练。

## 16. 给后续同事：Stage1 teacher / Stage3 wrapper projector 等价性验证结果

我已经读过第 15 节的首帧权重验证。那个验证说明：

- `FLASHVSR_STAGE3_FORCE_RECON_START=0` 下，v7-D2 确实把全局首帧的 `first_frame_pixel_weight=4.0` 和 `first_frame_lpips_weight=4.0` 传进了重建 loss；
- `latent_window=[0,2)` 到 `frame_window=[0,5)` 的映射符合当前 Stage3 random latent decode 约定；
- 但它不能关闭 `G_real/G_fake` teacher wrapper 等价性风险。

我接着在 6a 上补做了 projector 级别的等价性验证。结论是：**当前 v7-D2 的 `G_real/G_fake` 不是严格的 Stage1 v5.3.5 full-attention teacher/copy，而是 Stage1 权重套在 Stage2/Stage3 streaming wrapper 里。**

### 16.1 验证目的

第 13.6 节里留下的问题是：

```text
Stage1 v5.3.5 原始 teacher forward
vs
v7-D/D2 real_model dense_full wrapper forward
```

是否完全等价。

完整 DiT deterministic forward 很重，但当前最可疑的差异在进入 DiT 前就能检查：Stage1 v5.3.5 稳定母本使用 `nonstreaming_aligned` LQ projector，也就是 89f 对应 aligned23；而当前 Stage3B 继承 Stage2 wrapper，`FlashVSRStage2Pipeline.from_pretrained()` 中 `lq_proj_in` 的 `temporal_mode` 被硬编码为 `streaming`。

因此我先做低成本但有判别力的检查：

- 只加载 Stage1 v5.3.5 checkpoint 里的同一份 `lq_proj_in` 权重；
- 不加载完整 DiT；
- 用同一个 deterministic LQ 输入；
- 分别跑：
  - Stage1 预期语义：`nonstreaming_aligned`
  - 当前 Stage3 wrapper 语义：`streaming`
  - 参考旧模式：`nonstreaming`
- 如果 LQ conditioning token shape 已经不同，则完整 teacher forward 不可能严格等价。

### 16.2 新增脚本

新增本地脚本：

```text
wanvideo/model_training/flashvsr/tests/check_stage1_stage3_teacher_projector_equivalence.py
```

本地检查：

```bash
python3 -m py_compile wanvideo/model_training/flashvsr/tests/check_stage1_stage3_teacher_projector_equivalence.py
```

远端 6a 同步后也做了 `py_compile`，通过。

### 16.3 6a 资源状态

验证机器：

```text
6ai5mpi47f
```

卡位安排：

- GPU0/GPU1 用于验证；
- GPU2-7 继续用 `lxh_occupy_2_7` 占卡；
- 验证结束后 GPU0/GPU1 已空出来，GPU2-7 仍被占卡程序占用。

最终 6a GPU 状态：

```text
0, 0 MiB, 0 %
1, 0 MiB, 0 %
2, 166602 MiB, 100 %
3, 166602 MiB, 100 %
4, 166602 MiB, 100 %
5, 166602 MiB, 100 %
6, 166602 MiB, 100 %
7, 166602 MiB, 100 %
```

### 16.4 使用 checkpoint

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

脚本读取到：

```text
checkpoint_keys lq_proj=8 lora=480 other=0
```

### 16.5 89f projector 结果

远端日志：

```text
/mnt/task_wrapper/user_output/artifacts/debug/teacher_equiv_20260515/projector_equiv_89f_256x256.log
```

关键输出：

```text
input_shape=(1, 3, 89, 256, 256) device=cuda:0 dtype=torch.bfloat16
code_fact_stage1_pipeline_lq_proj_temporal_mode=configurable
code_fact_stage2_stage3_wrapper_lq_proj_temporal_mode=streaming_hardcoded
expected_stage1_v5_3_5_run_hint=nonstreamproj_aligned23
mode=nonstreaming_aligned output_shapes=[(1, 5888, 1536)]
mode=streaming output_shapes=[(1, 5632, 1536)]
mode=nonstreaming output_shapes=[(1, 5632, 1536)]
RESULT=FAIL_NOT_EQUIVALENT reason=shape_mismatch stage1_nonstreaming_aligned=(1, 5888, 1536) stage3_wrapper_streaming=(1, 5632, 1536)
```

解释：

- 89f、256x256 时，空间 token 数为 `16 * 16 = 256`；
- `5888 / 256 = 23`，对应 Stage1 `nonstreaming_aligned` 的 23 个 temporal latent positions；
- `5632 / 256 = 22`，对应 Stage3 wrapper 当前 `streaming` 的 22 个 temporal latent positions；
- 二者在进入 DiT 前 token 数已经不同。

### 16.6 17f projector 结果

远端日志：

```text
/mnt/task_wrapper/user_output/artifacts/debug/teacher_equiv_20260515/projector_equiv_17f_256x256.log
```

关键输出：

```text
input_shape=(1, 3, 17, 256, 256) device=cuda:0 dtype=torch.bfloat16
mode=nonstreaming_aligned output_shapes=[(1, 1280, 1536)]
mode=streaming output_shapes=[(1, 1024, 1536)]
mode=nonstreaming output_shapes=[(1, 1024, 1536)]
RESULT=FAIL_NOT_EQUIVALENT reason=shape_mismatch stage1_nonstreaming_aligned=(1, 1280, 1536) stage3_wrapper_streaming=(1, 1024, 1536)
```

解释：

- 17f、256x256 时，空间 token 数同样为 `256`；
- `1280 / 256 = 5`，对应 Stage1 `nonstreaming_aligned` 的 5 个 temporal latent positions；
- `1024 / 256 = 4`，对应当前 Stage3 wrapper `streaming` 的 4 个 temporal latent positions。

### 16.7 结论

这不是 dtype、随机种子、FlashAttention 或完整 DiT 数值容差问题，而是 LQ conditioning 在进入 DiT 前已经 shape mismatch。

所以当前 v7-D2 里：

```text
G_real/G_fake = Stage1 checkpoint weights + Stage3B/Stage2 streaming wrapper
```

而不是论文语义上更严格的：

```text
G_real/G_fake = Stage1 v5.3.5 full-attention teacher/copy
```

`stage3_real_attention_mode=dense_full` 和 `stage3_fake_attention_mode=dense_full` 只能说明 attention 方向是 full/dense；它不能修复 LQ projector temporal mode 已经变成 `streaming` 的事实。

### 16.8 对作者安排的理解

按 FlashVSR Stage3 的文字语义：

- `G_real` 应该是 Stage1 full-attention teacher；
- `G_fake` 应该是 `G_real` 的 copy，用来学习 fake latent distribution；
- 因此它们的 conditioning wrapper 应尽量保持 Stage1 teacher 的真实 forward 语义。

也就是说，哪怕 Stage1 teacher 产生 23 个 temporal positions，而 Stage3 student / one-step streaming 训练最终只比较 22 个 effective positions，也应该先让 `G_real/G_fake` 产生真正的 Stage1 `nonstreaming_aligned` 特征，然后在 DMD x0/score 对比处做显式 latent alignment / slicing。

更合理的方向是：

```text
G_real/G_fake:
  Stage1 nonstreaming_aligned teacher -> 23 latent positions

G_one/student:
  Stage3 one-step streaming -> 22 effective latent positions

DMD comparison:
  明确裁切/对齐 G_real/G_fake 的 x0/score 到和 z_pred 完全同一组 latent positions
  再计算 real_score - fake_score
```

而不是为了 shape 方便直接把 `G_real/G_fake` 的 projector 改成 streaming。

注意：这里不能随便裁 `[:, 1:]` 或 `[:, :-1]`。需要按现有 VAE latent mapping、Stage3 random latent decode、`frame_window` / `latent_window` 规则确认 22 个 effective latent positions 到底对应 Stage1 23 个 positions 中的哪 22 个。

### 16.9 建议

不要热改正在跑的 v7-D2。v7-D2 仍可作为 shared-noise DMD path 和 trainable fake critic 的长跑观察，但不能再声称已经严格复现 FlashVSR 论文里的 `G_real/G_fake = Stage1 full-attention teacher/copy`。

建议下一条线单独复制，例如 `v7-D3` 或 `v7-E`：

1. 让 `G_real/G_fake` wrapper 支持 Stage1 v5.3.5 的 `lq_proj_temporal_mode=nonstreaming_aligned`；
2. 保持 `G_one/student` 仍走 Stage3 one-step streaming；
3. 在 DMD probe / DMD student loss 里加入显式 latent alignment；
4. 先用 projector-level check 关闭 shape mismatch；
5. 再做完整 deterministic forward 或至少 probe-level x0 shape/value sanity check。

## 17. 对同事 projector 等价性发现的复核结论

我重新对照了当前代码，结论是：同事发现的问题成立，而且是必须修的主线问题，不是可以忽略的数值误差。

### 17.1 为什么这个问题成立

Stage1 v5.3.5 稳定母本的训练配置明确是：

```yaml
lq_proj_temporal_mode: nonstreaming_aligned
```

对应文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23.yaml
```

也就是说 Stage1 v5.3.5 的 teacher 语义是：

```text
89f -> 23 temporal latent positions
17f -> 5 temporal latent positions
```

但是当前 Stage2/Stage3 wrapper 里，`FlashVSRStage2Pipeline.from_pretrained()` 构造 `lq_proj_in` 时写死了：

```python
temporal_mode="streaming"
```

对应文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py
```

而 v7-D2 的 `G_real/G_fake` 又是通过 `FlashVSRStage3BTrainingModule` 构造，继承 Stage2 wrapper。因此即使 `G_real/G_fake` 加载的是 Stage1 checkpoint，它们实际 forward 的 LQ projector 语义仍然是 streaming：

```text
89f -> 22 temporal latent positions
17f -> 4 temporal latent positions
```

同事做的 projector-level check 直接证明了 shape mismatch：

```text
89f:
  Stage1 nonstreaming_aligned -> (1, 5888, 1536) = 23 * 256
  Stage3 wrapper streaming    -> (1, 5632, 1536) = 22 * 256

17f:
  Stage1 nonstreaming_aligned -> (1, 1280, 1536) = 5 * 256
  Stage3 wrapper streaming    -> (1, 1024, 1536) = 4 * 256
```

这不是 dtype、随机种子、FlashAttention、top-k、DMD noisy point 或完整 DiT 数值容差的问题。token 数在进入 DiT 之前已经不同，所以完整 teacher forward 不可能严格等价。

### 17.2 对当前 v7-D2 的准确命名

当前 v7-D2 更准确的描述是：

```text
Stage1 checkpoint weights + Stage3/Stage2 streaming wrapper
```

而不是严格的：

```text
Stage1 v5.3.5 full-attention teacher/copy
```

因此当前 v7-D2 可以继续作为 shared-noise DMD path 和 trainable fake critic 的 ablation 观察，但不能在汇报或文档里称为“完全作者对齐的 Stage3 DMD”。

### 17.3 是否应该热改当前 v7-D2

不建议热改正在跑的 v7-D2。

原因：

- 当前 v7-D2 已经是一个能跑的 shared-noise DMD ablation；
- 直接热改 teacher wrapper 会改变 DMD loss 的核心语义；
- teacher 23 positions 和 student 22 effective positions 之间还需要显式 alignment；
- 这个 alignment 不能随便写 `[:, :-1]` 或 `[:, 1:]`，必须按 frame/latent 语义验证。

建议复制新线，例如：

```text
v7-D3
```

或者如果语义变化更大，叫：

```text
v7-E
```

### 17.4 新线应该怎么改

新线目标：

```text
G_real/G_fake 保持 Stage1 v5.3.5 nonstreaming_aligned teacher 语义；
student/G_one 保持 Stage3 streaming one-step 语义；
DMD loss 内显式对齐 teacher/student 的 latent positions。
```

建议改法：

1. 给 `FlashVSRStage2Pipeline.from_pretrained()` 或 Stage3B wrapper 增加可配置的 `lq_proj_temporal_mode`。
2. 构造 `G_real/G_fake` 时传入：

```text
lq_proj_temporal_mode=nonstreaming_aligned
```

3. 构造 student 时仍使用 Stage3 streaming 语义。
4. 在 DMD probe / DMD student loss 里显式处理：

```text
teacher: 23 temporal positions
student: 22 effective temporal positions
```

5. 这个 23->22 的 alignment 必须写成独立函数，并配套单元/日志验证。

### 17.5 后续必须补的验证

在新线正式上 48GPU 前，至少补以下验证：

1. `projector-level check`
   - 修复后，`G_real/G_fake` wrapper 必须输出 Stage1 语义：
   - 89f -> 23 positions；
   - 17f -> 5 positions。

2. `latent alignment check`
   - 明确 teacher 23 positions 中哪 22 个参与 DMD；
   - 明确它们对应 student 22 positions 的 frame/latent 关系；
   - 不允许隐式 trim。

3. `DMD zero-diff sanity`
   - 在 `G_fake` 初始等于 `G_real` 时；
   - 同一个 noisy point / timestep 下；
   - DMD real/fake 差值应接近 0；
   - fake 更新后再变成非零。

4. `optimizer ownership check`
   - `G_real` 必须 frozen/no-grad；
   - `G_fake` 只在 fake update step 更新；
   - student optimizer 不应更新 fake；
   - fake optimizer 不应更新 student。

5. `random latent decode window check`
   - 首帧已验证通过：
     - `latent_window=[0,2)`
     - `frame_window=[0,5)`
     - `first_frame_pixel_weight=4.0`
     - `first_frame_lpips_weight=4.0`
   - 还需要补中间 window 和尾部 window：
     - 验证 `frame_window` 没错位；
     - 验证尾部不越界；
     - 验证非首帧 window 不错误套用 4 倍首帧权重。

### 17.6 当前建议

短期：

- 不热改正在跑的 v7-D2；
- 继续观察 v7-D2 作为 shared-noise DMD ablation；
- 在文档中明确它不是严格 Stage1 teacher wrapper。

下一步：

- 新建 `v7-D3` / `v7-E`；
- 先修 teacher wrapper temporal mode；
- 再修 teacher/student latent alignment；
- 完成上面的五类验证后，再上 48GPU。

## 18. v7-D3 实现与 6a smoke 验证结果

我按你第 17 节的结论新建了 `v7-D3`，没有热改正在跑的 `v7-D2`。核心目标是把 `G_real/G_fake` 从：

```text
Stage1 checkpoint weights + Stage2/3 streaming wrapper
```

改成：

```text
Stage1 checkpoint weights + dense_full attention + nonstreaming_aligned projector wrapper
```

同时保留 student/G_one 的 Stage3 one-step streaming 语义，并在 DMD 对比处显式做 teacher/student latent alignment。

### 18.1 代码改动

共享底座：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
```

新增能力：

- `FlashVSRStage2Pipeline.from_pretrained()` / `FlashVSRStage2TrainingModule` 增加 `lq_proj_temporal_mode` 参数；
- 默认仍是 `streaming`，避免改变旧分支；
- `flashvsr_stage2_model_fn()` 增加 `lq_latent_alignment` 参数；
- `lq_latent_alignment=trim_front_to_match` 时，显式把 teacher 的 23-position LQ conditioning 裁到 student 的 22 effective positions；
- `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1` 时打印对齐日志。

D3 新线：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_lora_89f_videoonly_authorweights_stage1teacher_aligned_offlinewandb.yaml
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_teacher_aligned.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-TeacherAligned.sh
```

D3 config 中：

```yaml
stage3_real_lq_proj_temporal_mode: "nonstreaming_aligned"
stage3_fake_lq_proj_temporal_mode: "nonstreaming_aligned"
```

DMD probe、DMD student loss、以及 generic `probe_model.forward()` 路径都会根据 wrapper temporal mode 自动设置：

```text
lq_latent_alignment=trim_front_to_match
```

### 18.2 full attention / FlashAttention

D3 没有把 teacher 改成慢的手写 full attention。`G_real/G_fake` 仍设置：

```text
attention_mode=dense_full
```

当前 Stage2 patched attention 中，`dense_full` 分支仍调用 `flash_attention(...)`。因此这里的含义是：

```text
full-attention mask semantics + FlashAttention kernel
```

不是：

```text
Python/PyTorch 手写 dense attention
```

### 18.3 6a smoke 过程

远端机器：

```text
6ai5mpi47f
```

远端代码目录：

```text
/mnt/task_runtime/lucidvsr
```

本地和远端静态检查都已通过：

```text
python -m py_compile train_flashvsr_stage2_v6_4_lora.py train_flashvsr_stage3_v7_d3_lora.py
bash -n FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh
bash -n FlashVSR-Stage3-Smoke-2GPU-v7-D3-TeacherAligned.sh
```

第一次 smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_smoke1
```

失败原因是有效发现：

```text
Stage2 requires exact LQ/DiT token match ... x=84480, lq=88320 ... alignment=exact
```

这是因为 `real_probe_every` 在 step0 触发 generic `probe_model.forward()`，该路径还没带 D3 alignment。我随后在 `FlashVSRStage3BTrainingModule.forward()` 里补齐，避免 probe path 和 DMD path 语义不一致。

主 smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_smoke2
```

通过 step 1。关键日志：

```text
G_real ... attention_mode=dense_full lq_proj_temporal_mode=nonstreaming_aligned
G_fake ... attention_mode=dense_full lq_proj_temporal_mode=nonstreaming_aligned
[stage3_teacher_align] mode=trim_front_to_match grid=(22, 48, 80) expected_tokens=84480 aligned_lq_shape=(1, 84480, 1536)
real/fake timestep=368.000000 shared_point=True
noise_pred=(1, 16, 22, 96, 160)
x0=(1, 16, 22, 96, 160)
[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 fake_scale=0.000000 real_probe=0.107180 fake_probe=0.059744 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
```

这个结果关闭了三件事：

- `G_real/G_fake` 确认使用 `nonstreaming_aligned`；
- teacher 23 positions 确认显式对齐到 student 22 positions；
- 初始 `G_fake == G_real` 时，shared noisy point 下 DMD zero-diff sanity 成立。

### 18.4 window 验证

中间 window：

```text
FLASHVSR_STAGE3_FORCE_RECON_START=10
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_mid10_smoke
```

关键日志：

```text
latent_window=[10,12) frame_window=[37,45) decode_latents=[0,12) local_frame_window=[0,8) detached_context_latents=10 first_frame_pixel_weight=1.0 first_frame_lpips_weight=1.0
```

尾部 window：

```text
FLASHVSR_STAGE3_FORCE_RECON_START=20
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_tail20_smoke
```

关键日志：

```text
latent_window=[20,22) frame_window=[77,85) decode_latents=[0,22) local_frame_window=[0,8) detached_context_latents=20 first_frame_pixel_weight=1.0 first_frame_lpips_weight=1.0
```

加上之前 D2/D3 首帧 smoke：

```text
latent_window=[0,2) frame_window=[0,5) first_frame_pixel_weight=4.0 first_frame_lpips_weight=4.0
```

现在首帧、中间、尾部三个 decode window 都已覆盖。

### 18.5 对你第 17.5 节验证清单的回应

| 你列的检查项 | 当前状态 |
|---|---|
| projector-level check | D3 日志确认 real/fake wrapper 为 `nonstreaming_aligned`，alignment 前后 token 数符合 23 -> 22 |
| latent alignment check | D3 使用显式 `trim_front_to_match`，日志显示 `88320 -> 84480` tokens |
| DMD zero-diff sanity | step1 `dmd_student=0.000000`，`dmd_grad=0.000000` |
| optimizer ownership check | smoke 中 `fake_update=1`、`fake_loss` 非零；`G_real` 仍走 frozen probe path；后续可补参数 diff 级别检查 |
| random latent decode window check | 首帧 / 中间 / 尾部均已 smoke |

### 18.6 当前结论

我同意你“必须改”的判断。`v7-D3` 已经把 `v7-D2` 最大的语义偏差修掉：`G_real/G_fake` 不再是 streaming wrapper，而是按 Stage1 v5.3.5 的 `nonstreaming_aligned` projector 语义构造，再在 DMD 入口显式对齐到 student 22 positions。

当前仍需诚实标注两点：

- `v7-D2` 只能作为 shared-noise DMD ablation，不应称为严格 Stage1 teacher/copy 复现；
- `v7-D3` 已通过 2GPU smoke 和五类关键 sanity，但还没有启动 48GPU 长训；如果要更严谨，可在 48GPU 前再补一个完整 Stage1 module deterministic forward 对比。

## 19. Codex 对同事 v7-D3 代码的二次审查（2026-05-15 23:31）

本节是对第 18 节同事实现的独立复核。结论是：`v7-D3` 的主方向是对的，但它目前只能说“修复了 D2 最大的 teacher wrapper 偏差，并通过 shape / shared-noise smoke”，还不能说已经完整关闭所有 Stage1 teacher 等价性风险。

### 19.1 已确认正确的改动

我实际检查到的新增代码位置：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_lora_89f_videoonly_authorweights_stage1teacher_aligned_offlinewandb.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh
```

代码层面确认：

- `FlashVSRStage2Pipeline.from_pretrained()` 新增了 `lq_proj_temporal_mode`，默认仍是 `streaming`，不会改变旧 Stage2/Stage3 分支。
- `G_real` 构造时传入 `lq_proj_temporal_mode=args.stage3_real_lq_proj_temporal_mode`。
- `G_fake` 构造时传入 `lq_proj_temporal_mode=args.stage3_fake_lq_proj_temporal_mode`。
- D3 yaml 设置了：

```yaml
stage3_real_lq_proj_temporal_mode: "nonstreaming_aligned"
stage3_fake_lq_proj_temporal_mode: "nonstreaming_aligned"
```

- DMD probe path、fake-FM path、以及 generic `probe_model.forward()` path 都会写入：

```python
merged["lq_latent_alignment"] = _stage3d3_teacher_lq_alignment_mode(pipe)
```

这修掉了 D2 的核心问题：`G_real/G_fake` 不再无意中使用 Stage3 streaming projector wrapper。

### 19.2 需要谨慎解释的 alignment

D3 当前 alignment 是：

```python
if temporal_mode == "nonstreaming_aligned":
    return "trim_front_to_match"
```

底层调用的是 `v5._align_lq_latents_to_dit_tokens()`。当 teacher LQ tokens 比 student tokens 多一个 temporal position 时，它执行的是：

```python
return layer_latents[:, trim_tokens:, :]
```

也就是从前面裁掉一个 temporal position，把 23-position teacher conditioning 裁成 22-position student conditioning。

这个方向从 shape 上是成立的：

```text
teacher 23 * H * W -> student 22 * H * W
88320 tokens -> 84480 tokens
```

但这里还需要明确语义：裁掉的是 teacher 的 warm-up / first-position conditioning。它和当前 Stage3 student 的 22 effective latents 是否逐位置语义完全一致，目前只通过 smoke 日志证明了 shape，不等于证明了数值语义完全等价。

因此 D3 还需要一个更严格的检查：

- 打印 teacher 23 个 temporal positions 对应的 raw/LQ frame 区间；
- 打印 student 22 个 temporal positions 对应的 raw/LQ frame 区间；
- 明确 `trim_front_to_match` 后 teacher position 1..22 是否正好对齐 student position 0..21。

### 19.3 同事已经验证了什么

第 18 节 smoke 已验证：

- `G_real/G_fake` wrapper 打印为 `nonstreaming_aligned`；
- `stage3_teacher_align` 日志显示 23 -> 22 shape alignment 成功；
- real/fake 使用 shared `dmd_point`，即同一个 timestep / noise / noisy_latents；
- 初始 `G_fake == G_real` 时，`dmd_student=0`、`dmd_grad=0`，shared-noise zero-diff sanity 成立；
- 首帧、中间、尾部 random latent decode window 都跑通过。

这些验证是有价值的，但它们不是完整等价证明。

### 19.4 还没有完全验证的东西

还缺以下检查：

- 完整 `G_real/G_fake` DiT forward 和 Stage1 v5.3.5 teacher 的 deterministic 数值对比。当前 `check_stage1_stage3_teacher_projector_equivalence.py` 是 projector 级别验证，而且脚本内容仍偏向证明 D2 不等价，不是 D3 完整验证。
- optimizer ownership 参数级检查。当前只看到 `fake_update=1` 和 `fake_loss` 非零；还没看到“G_fake 参数更新、G_real 参数不变、student 参数不被 fake optimizer 更新”的参数 diff 级验证。
- `trim_front_to_match` 的 temporal semantic map。当前日志只有 token 数，没有列出 frame/latent position 对齐表。
- validation 仍是 Stage3 one-step direct decode 路径，不是严格的 Stage2/FlashVSR streaming KV-cache validation。这不影响 DMD training path，但不能用它证明最终 streaming inference 质量。

### 19.5 发现的小问题和风险

1. D3 的 `stage3_fake_checkpoint` 在 yaml 里是 `null`，但 release/smoke sh 会通过 CLI 传入 `--stage3_fake_checkpoint "${STAGE3_FAKE_CHECKPOINT}"`。所以通过 sh 启动是对的；如果以后有人只用 yaml 直接启动，会退回 scalar fake placeholder，不是 D3 预期。建议 D3 后续加 guard：当 `stage3_fake_fm_weight > 0` 或 `stage3_dmd_weight > 0` 时必须显式提供 `stage3_fake_checkpoint`。

2. D3 48GPU sh 的 snapshot 仍复制：

```text
diffsynth/models/wan_video_dit_stage2_v6_1.py
```

但 D3 代码依赖的是 `train_flashvsr_stage2_v6_4_lora.py`。这不一定影响训练，但 snapshot 不完整，后续复现实验时容易误导。建议同时 snapshot `train_flashvsr_stage2_v6_4_lora.py` 和实际使用的 `diffsynth/models/wan_video_dit_stage2_v6_1.py`。

3. 当前 D3 validation meta 仍写：

```text
validation_mode: stage3_v7_b_one_step_recon
```

这只是旧命名，容易误解。建议改成 D3 自己的名字，或者明确它只是 one-step direct validation，不是 streaming KV-cache validation。

### 19.6 当前判断

我对同事 D3 代码的判断：

- 主改动方向正确，确实修了 D2 最大的 `G_real/G_fake` wrapper 语义偏差。
- smoke 证明了代码能跑、shape 能对齐、shared noisy point 生效、首/中/尾 decode window 能跑。
- 但 `trim_front_to_match` 只证明了 token 数对齐，还没有严格证明 teacher/student temporal position 语义对齐。
- optimizer ownership 还没有做到参数 diff 级验证。
- validation 不是最终 streaming KV-cache validation，不能拿它直接代表最终推理路径。

下一步如果要把 D3 升级成正式候选，建议先补两个轻量验证：

1. `teacher/student temporal map check`：打印 23 -> 22 裁剪前后的 latent index / frame range 对齐表。
2. `optimizer ownership check`：跑 1 step，保存前后参数 hash，确认 G_real 不变、G_fake trainable params 变化、student 只被 student optimizer 更新。

## 20. 对第 19 节二次审查的补丁与验证

我同意第 19 节的新增问题，其中有几项属于应该马上补的工程风险。已经按以下方式处理。

### 20.1 已补代码

`stage3_fake_checkpoint` guard：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py
```

现在 parse 阶段如果满足以下任一条件：

```text
stage3_fake_fm_weight > 0
stage3_dmd_weight > 0
```

就必须显式提供：

```text
stage3_fake_checkpoint
```

否则直接 `parser.error(...)`。这样可以避免只用 yaml 启动 D3 时，因为 yaml 里 `stage3_fake_checkpoint: null` 而退回 scalar fake placeholder。

temporal map 日志：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
```

`FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1` 时，`[stage3_teacher_align]` 不再只打印 shape，还会打印：

```text
teacher_tokens_before
teacher_positions_before
drop_teacher_positions
keep_teacher_positions
student_positions
```

optimizer ownership debug：

```text
FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=1
```

第一步 optimizer 前后会计算少量 trainable / frozen 参数 checksum，并打印：

```text
student_changed
fake_changed
real_changed
```

validation meta：

原来的旧名：

```text
stage3_v7_b_one_step_recon
```

已改成：

```text
validation_mode=stage3_v7_d3_one_step_direct_decode
validation_mode_detail=not_streaming_kvcache_validation
```

这样不会误导成最终 streaming KV-cache validation。

snapshot：

D3 release / smoke 脚本现在都会 snapshot：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py
diffsynth/models/wan_video_dit_stage2_v6_1.py
```

### 20.2 静态检查

本地通过：

```text
python3 -m py_compile train_flashvsr_stage2_v6_4_lora.py train_flashvsr_stage3_v7_d3_lora.py
bash -n FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh
bash -n FlashVSR-Stage3-Smoke-2GPU-v7-D3-TeacherAligned.sh
```

6a 远端也通过同样的 py_compile / bash -n。

### 20.3 6a patchcheck smoke

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_patchcheck_smoke
```

运行设置：

```text
CUDA_VISIBLE_DEVICES=0,1
FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1
FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=1
FLASHVSR_STAGE3C_DMD_DEBUG=0
```

temporal map 关键日志：

```text
[stage3_teacher_align] mode=trim_front_to_match grid=(22, 48, 80) expected_tokens=84480 aligned_lq_shape=(1, 84480, 1536) teacher_tokens_before=88320 teacher_positions_before=23 drop_teacher_positions=[0,1) keep_teacher_positions=[1,23) student_positions=[0,22) note=teacher_position0_is_nonstreaming_aligned_warmup
```

这说明当前 D3 的实际 alignment 是：

```text
drop teacher position 0
keep teacher positions 1..22
map to student positions 0..21
```

也就是说我们显式把 `nonstreaming_aligned` 多出来的 warm-up position 丢掉，而不是默默按 shape 裁剪。

optimizer ownership 关键日志：

```text
[stage3d3_optimizer_ownership] student_changed=True fake_changed=True real_changed=False student_before=40.94118383526802 student_after=40.94704721868038 fake_before=40.983435437083244 fake_after=40.98426878452301 real_before=-196.83386889845133 real_after=-196.83386889845133 fake_update=1
```

这关闭了第 19 节里“只看到 fake_update=1，不是参数 diff 级验证”的问题：至少在 smoke 的 checksum 级别，student/fake 确实变化，frozen real 不变。

训练 step 仍正常：

```text
[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 fake_scale=0.000000 real_probe=0.107180 fake_probe=0.059744 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
```

### 20.4 当前剩余边界

第 19 节里最重的“完整 Stage1 module deterministic forward 对比”仍没有做。这不是这次补丁能顺手关闭的问题，因为它需要完整构造 Stage1 v5.3.5 teacher 和 Stage3 D3 teacher wrapper，在同一输入下对比 forward 中间/输出数值。

当前 D3 的状态应更新为：

- teacher wrapper temporal mode 已修；
- 23 -> 22 alignment 的裁剪语义已记录；
- optimizer ownership 已做 smoke checksum 级验证；
- validation meta 已避免误导；
- 仍未声明“完整 deterministic forward 等价性已经证明”。

## 21. v7-D3.1 干净正式线与 48GPU 启动

按用户要求，我没有删除 `v7-D3`。`v7-D3` 保留为带 temporal map / optimizer ownership 审查日志的已验证版本；正式长训另复制为 `v7-D3.1`。

### 21.1 新文件

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_1_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb.yaml
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-Clean-OfflineWandb.sh
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-1-TeacherAligned-Clean.sh
```

D3.1 保留：

- `G_real/G_fake` 的 `nonstreaming_aligned` Stage1 teacher wrapper；
- DMD 入口的 `trim_front_to_match`；
- shared-noise DMD；
- trainable `G_fake`；
- `stage3_fake_checkpoint` guard；
- D3 validation meta 命名。

D3.1 清理：

- 移除 D3 中只为 smoke 审查服务的 optimizer checksum 代码；
- release/smoke 脚本默认关闭：

```text
FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0
FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0
```

### 21.2 D3.1 clean smoke

6a clean smoke：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean_20260515_v7d31_clean_smoke
```

结果：

```text
[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 fake_scale=0.000000 real_probe=0.107180 fake_probe=0.059744 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
stage3_teacher_align count = 0
stage3d3_optimizer_ownership count = 0
```

这证明干净线仍能跑，并且不会刷审查日志。

### 21.3 停 D2

已停止旧 48GPU：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb_20260515_v7d2_sharednoise_48gpu
```

6 节点：

```text
t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk
```

均确认：

```text
PIDS_D2=0
```

### 21.4 启动 D3.1 48GPU

新实验：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu
```

启动：

```text
MASTER_ADDR=240.12.138.137
MASTER_PORT=29531
RUN_TS_OVERRIDE=20260515_v7d31_clean_48gpu
```

6 节点均启动远端 tmux：

```text
v7d31_48gpu
```

当前状态：

```text
PIDS_D31=9 on each node
PIDS_D2=0 on each node
```

主节点已出：

```text
step=1 loss=0.854109 student=0.722192 fake_loss=0.13191710 fake_update=1 real_probe=0.093054 fake_probe=0.101467 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0
```

已看到：

```text
output/training_state/step-1
output/training_state/step-2
output/validation/step-1
output/validation/step-2
```

主节点 run.log 计数：

```text
stage3_teacher_align = 0
optimizer_ownership = 0
```

### 21.5 W&B

主节点：

```text
wandb_package_v7d31_stage1teacher_clean
```

6a：

```text
wandb_sync_from_s3_v7d31_stage1teacher_clean
```

S3：

```text
s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu.tar.gz
```

已手动触发一次 package + sync，6a 同步成功：

```text
offline-run-20260515_011822-6nu0uhd7
offline-run-20260515_091237-60budl7y
```

### 21.6 仍需诚实保留的边界

D3.1 可以作为当前正式候选，因为它已经修正了 D2 最大的 teacher wrapper 偏差，并且 D3/D3.1 已通过 shape、temporal map、optimizer ownership、clean smoke、48GPU step1。

但仍不能说已经数学上完全证明 Stage1 teacher forward 等价。完整 Stage1 teacher deterministic forward 数值对比还没有做。

## 21. W&B 环境差异复查：6a 可同步、6 节点主机一度不可同步的原因判断（2026-05-16）

用户提出疑问：之前 6 节点母机可以写 W&B，后来不行；现在 6a 可以。是否缺少某些验证文件或配置文件。

我对比检查了：

```text
6a: 6ai5mpi47f
6 节点主机 rank0: t5qdtykjsw
```

检查项包括：

- `/root/.netrc`
- `/root/.config/wandb/settings`
- `wandb` Python 包版本
- 远端到 `https://api.wandb.ai/graphql` 的 HTTPS 连通性
- `wandb` CLI 是否在 flashvsr 环境里可用

### 21.1 检查结果

两台机器都有 W&B 凭证文件：

```text
/root/.netrc exists True size 132
/root/.config/wandb/settings exists True size 11
```

两台机器都能 import W&B：

```text
wandb_version 0.23.1
```

两台机器到 W&B API 的 HTTPS 网络是通的。访问 `https://api.wandb.ai/graphql` 返回：

```text
HTTP Error 405: Method Not Allowed
```

这个返回本身不是失败，反而说明已经连到 W&B 服务端，只是用 GET 访问 GraphQL endpoint 方法不对。

两台机器在显式使用 flashvsr 环境时，W&B CLI 都存在：

```text
/mnt/conda_envs/flashvsr/bin/python -m wandb --version
python -m wandb, version 0.23.1

/mnt/conda_envs/flashvsr/bin/wandb --version
wandb, version 0.23.1
```

但在未激活 flashvsr 环境的非交互 shell 里，裸跑：

```text
wandb
```

可能会失败，因为默认 `PATH` 不包含：

```text
/mnt/conda_envs/flashvsr/bin
```

### 21.2 当前判断

这次不像是缺 W&B 验证文件。

更可能的原因是：

1. 某些训练脚本或后台 tmux 同步脚本没有继承 flashvsr 环境的 `PATH`，裸写 `wandb sync` 时找不到命令。
2. online `wandb.init()` 可能被网络/代理瞬时卡住；offline 模式绕过了训练进程内的在线握手，因此更稳定。
3. 本机有时带着 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 连接 Bolt，会被指到 `127.0.0.1:7897`，导致本机侧 `bolt task ssh` 失败。这是本机 Bolt 连接问题，不是远端 W&B 凭证问题。

### 21.3 给后续脚本的固定写法

后续 48 卡长训建议继续使用 offline W&B，并固定：

```bash
export PATH="/mnt/conda_envs/flashvsr/bin:$PATH"
export WANDB_DIR="${RUN_DIR}"
export WANDB_MODE=offline
```

同步脚本不要裸写：

```bash
wandb sync ...
```

应显式写成：

```bash
/mnt/conda_envs/flashvsr/bin/wandb sync ...
```

或者：

```bash
/mnt/conda_envs/flashvsr/bin/python -m wandb sync ...
```

这样可以避免非交互 tmux / 非 login shell 环境下 PATH 不一致导致同步失败。

### 21.4 给同事的结论

目前没有证据说明 6 节点母机缺 W&B key 或配置文件。6a 和 t5 的 W&B 文件、包版本、API 网络连通性都基本一致。

如果后续 D3/D2/7D 系列 W&B 同步失败，优先检查：

- 同步命令是否使用 `/mnt/conda_envs/flashvsr/bin/wandb`；
- `WANDB_DIR` 是否指向 artifacts 内的实验目录；
- 是否在 training rank0 的远端 tmux 里运行，而不是本机环境；
- 本机执行 `bolt task ssh` 前是否需要清掉本机代理变量。

## 22. D3.2 数据分片修复版：低利用率复查、48GPU 启动与当前边界（2026-05-16）

用户要求：

- 不减少 save 点；
- 不取消 `stage3_decoder_cpu_offload`；
- 不 resume，修复后干净重启；
- 对 48 卡低利用率必须定位根因，不做表面参数调整。

### 22.1 低利用率定位

增加了两个默认关闭的 timing 开关：

```text
FLASHVSR_STAGE3_TIMING_DEBUG=1
FLASHVSR_DATA_TIMING_DEBUG=1
```

2GPU 诊断显示主要等待来自 data 阶段，而不是 save：

```text
data=91.296s -> 0.286s -> 62.721s
save_sched=0
```

进一步拆数据阶段发现：

- decode 通常是数秒到十秒级；
- online CPU degradation 可到 `61s/99s/116s`；
- convert 约 `1s`。

因此这次“周期性掉 0”主要是数据等待，根因是 CPU 退化很慢，再叠加跨 rank 重复取样导致 CPU/I/O 被浪费。

### 22.2 修复内容

`wanvideo/data/flashvsr/datasets/streaming_dataset.py` 修复：

- DataLoader `spawn` worker 内 `torch.distributed` 未初始化时，使用环境变量 `RANK/WORLD_SIZE` fallback；
- worker RNG seed 纳入 rank；
- tar URL datapipe 先按 distributed rank 分片，再按 worker 分片；
- 诊断日志确认 rank0/rank1 不再处理同一视频样本。

2GPU 诊断结果：

```text
dataset_num_workers=4, prefetch=1, stage3_decoder_cpu_offload=true
data=91.664s -> 0.287s -> 0.376s
```

首步仍可能受预热/慢样本影响，但后续 data wait 明显改善。

### 22.3 D3.2 正式文件

正式 config：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml
```

正式 wrapper：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-2-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-OfflineWandb.sh
```

保留项：

```text
save_steps: 500
extra_save_steps: "1,2,5,10,20,50,100,200"
stage3_decoder_cpu_offload: true
validation_num_samples: 1
```

正式数据配置：

```text
dataset_num_workers: 4
dataloader_prefetch_factor: 1
```

### 22.4 48GPU 当前 run

当前正式 run：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2
```

启动参数：

```text
MASTER_ADDR=240.12.138.137
MASTER_PORT=29541
RUN_TS_OVERRIDE=20260516_v7d32_datafix_48gpu_fresh2
```

启动中遇到一次 `V7D32_EXIT=126`，原因是 D3.2 wrapper `exec` 到 D3.1 base script，而 base script 远端缺 executable bit。已在本地 `chmod +x` 并等同步到远端，t5/a9 均确认：

```text
-rwxr-xr-x
```

之后使用 fresh2 重新干净启动。

### 22.5 当前训练状态

主节点已出 step/loss，且暂无错误：

```text
step=2 loss=0.847753 student=0.752921 fake_loss=0.09247246 dmd_student=0.002360 dmd_grad=0.050241 dmd_skip=0
step=3 loss=0.877787 student=0.828751 fake_loss=0.04516934 dmd_student=0.003866 dmd_grad=0.064943 dmd_skip=0
step=4 loss=0.502095 student=0.477836 fake_loss=0.01839858 dmd_student=0.005860 dmd_grad=0.081814 dmd_skip=0
step=5 loss=0.851825 student=0.661957 fake_loss=0.17458852 dmd_student=0.015279 dmd_grad=0.124042 dmd_skip=0
step=6 loss=0.425419 student=0.248533 fake_loss=0.16397065 dmd_student=0.012915 dmd_grad=0.119689 dmd_skip=0
```

`grep` 未见：

```text
Traceback / RuntimeError / CUDA out / Killed / ValueError
```

显存/利用率观察：

- 6 个 48 卡节点显存稳定占用，约 `146GB-157GB / 183GB` 每卡；
- 多轮 watch 中大多数 GPU 为 `97%-100%`；
- 个别 GPU 某一秒低到 `9%/55%/72%`，但不是全节点一起掉 0，下一轮又恢复；
- 当前看更像 step 内同步/局部阶段，不像数据管线整体卡死或空挂。

### 22.6 W&B 状态

这是 offline W&B，不是训练进程在线直连 W&B。

本地 run 目录中已有 offline run：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2/wandb/offline-run-20260515_123145-y2alwjkp
```

后台 package session 存在：

```text
wandb_package_v7d31_stage1teacher_clean
```

虽然 session 名仍沿用 D3.1，但它绑定的 `RUN_DIR` 是 D3.2 fresh2：

```text
run_dir=/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2
interval=3600s
```

S3 初始包已上传：

```text
s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2.tar.gz
```

注意：这个机制是“后台每小时打包上传 offline W&B 目录到 S3”，不是自动在线同步到 W&B cloud。要在 W&B 网页直接看到，需要后续对该 offline run 执行 `wandb sync`。

### 22.7 当前结论与仍需保留的边界

当前 D3.2 已经修复 D2/D3.1 之后发现的最大实现问题：

- DMD real/fake shared noisy latent/timestep；
- Stage1 teacher wrapper 的 LQ 对齐改成 nonstreaming aligned；
- dataset distributed rank sharding；
- fake critic 每步更新；
- validation/offload/save 策略按用户要求保留；
- 48GPU 已出 loss，DMD 项非零且未 NaN；
- 显存和利用率当前正常。

但仍需诚实保留边界：

1. 完整 Stage1 teacher deterministic forward 数值等价仍未做。
2. 当前 DMD 实现是按 DMD2/OSEDiff 的 fake critic 训练语义解释 FlashVSR 的 `L_FM(z_pred, G_fake)`；FlashVSR 论文没有给出足够代码级细节证明这是唯一解释。
3. 目前只确认早期 step 正常；长训稳定性、后续 checkpoint 保存、下一轮 hourly W&B package、以及更长时间 loss 行为仍需继续观察。

因此可以说：D3.2 是目前最接近 FlashVSR Stage3/DMD 论文目标、且已经通过关键 smoke/48GPU 早期运行验证的版本；不能说已经数学上完全证明和作者实现等价。

## 2026-05-16 交接补充：Stage1 v5.3.5 USMGT / Takano 20250205 微调实验

这条记录是给后续接手同事看的，避免把本轮 Stage1 微调和 Stage3 DMD 代码混在一起。

### 实验目的

- 从 Stage1 v5.3.5 89f 稳定母本 `step-10000` warm-start。
- 视频源切到新的 Takano 4K 数据：
  - `s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/`
- 图像 branch 仍沿用旧 Takano image manifest。
- 在进入 LQ 退化前，对 GT 做 Real-ESRGAN 风格 USM sharpening：
  - sharpen 后的 GT 作为 VAE target；
  - 同一个 sharpen 后的 GT 再生成 LQ；
  - 目的不是改退化强度，而是假装 GT 数据更锐、更干净，观察 Stage1 质量是否能继续提升。

### 关键代码边界

- 不再修改全局 `streaming_dataset.py` / `tar_streaming_dataset_v53.py`。
- 本轮 USMGT 逻辑只放在隔离文件：
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53_usmgt.py`
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`
- 原因：全局 dataset 被 Stage1/2/3 多条线复用，不能为了一个 Stage1 微调实验把全局数据流改坏。

### 正在跑的 16GPU 实验

- 机器：
  - rank0：`bfs6vaz4d6`
  - rank1：`i6hf4scd4y`
- 启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh`
- 配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml`
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
- W&B：
  - online 正常；run id：`8dnrur64`

### 当前确认过的敏感参数

- `degradation_device: auto`
  - 远端 CUDA 可用时，worker 内部会按 `LOCAL_RANK` 使用 `cuda:<LOCAL_RANK>`。
- `gt_sharpen: true`
- `gt_sharpen_backend: torch`
- `gt_sharpen_device: auto`
- `dataset_num_workers: 2`
- `dataloader_multiprocessing_context: spawn`
  - 这个是关键修复。
  - 没有 spawn 时，worker 内部 `.to(cuda)` 会触发：`Cannot re-initialize CUDA in forked subprocess`。
- `validation_num_samples: 3`
- `lq_proj_temporal_mode: nonstreaming_aligned`
- `learning_rate: 5e-6`
- `num_frames: 89`
- `image_branch_num_frames: 5`

### 已确认训练状态

- 2GPU smoke 成功：
  - `step=1 loss=0.061863`
  - `step=2 loss=0.012738`
- 16GPU 正式实验已出 loss：
  - `step=1 loss=0.081164`
  - `step=5 loss=0.065258`
  - `step=7 loss=0.074081`
- 主节点 GPU util 后续采样基本达到 `98% - 100%`，说明 GPU 退化 + worker=2 + spawn 路线能跑。

### manifest

- 新视频 manifest：
  - 本地：`wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt`
  - S3：`s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt`
  - 共 `7593` 个 tar shard。

### 可视化检查

- 已新增只读导出脚本：
  - `tools/flashvsr_export_usmgt_samples.py`
- 用途：导出训练前的 `gt_usm` / `lq_degraded`，以及 GT raw vs GT USM 对比。
- 注意：训练本身使用 GPU auto；可视化导出为了不抢正在训练的显存，改用 CPU/opencv 路线生成检查视频。
- 远端输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/usmgt_checks/takano20250205_usmgt_checks_20260516`
- 本机目标目录：
  - `/Users/lixiaohui/Desktop/takano20250205_usmgt_checks_20260516`

### 操作注意

- 如果 conductor 在后台 tmux 报凭证问题，不要直接判定 conductor 不可用。
- 用户手动 ssh/zsh 进去 conductor 可用，说明问题通常是非交互 shell 没继承认证环境或 PATH。
- 后续上传/下载建议：
  - 用正常 bolt ssh 进入远端 zsh；
  - 显式使用 flashvsr 环境或确认 conductor 凭证；
  - 不要在没有确认环境的后台裸跑 conductor。

## 2026-05-16 补充：D3.2 W&B 远端 relay 已接通

用户反馈 W&B 页面未看到 D3.2/`v7d32` 实验后，重新检查了远端同步链路。

结论：

- t5 的 conductor 在 `zsh -lc` 下正常；
- t5 有 `~/.netrc`；
- t5 的 `wandb status` 显示 `api_key: null`，但这不能代表 netrc 登录态不可用；
- 直接实测 t5 对当前 offline run 执行 `wandb sync` 成功，所以 t5 也能 cloud sync；
- 新 16GPU 主机 `bfs6vaz4d6` 的 conductor 正常，且存在 `~/.netrc`；
- `bfs6vaz4d6` 上执行 `wandb sync` 成功。

当前 W&B run：

```text
https://wandb.ai/veralee/flashvsr/runs/y2alwjkp
```

当前自动链路：

```text
t5qdtykjsw tmux: wandb_sync_d3_2_t5_direct
  every 900s: wandb sync local offline-run directly
  timeout 180s per sync attempt

t5qdtykjsw tmux: wandb_package_d3_2_fresh2
  every 900s: package D3.2 wandb/ -> s3://lxh/tmp/wandb_offline/...fresh2.tar.gz
  kept as backup/archive path

bfs6vaz4d6 tmux: wandb_sync_d3_2
  every 900s: download S3 tar -> wandb sync offline-run-20260515_123145-y2alwjkp
  now redundant/backup relay
```

已确认：

- t5 新 package loop 首轮成功，S3 包更新到 `133620` bytes；
- t5 直接 `wandb sync /mnt/task_wrapper/.../wandb/offline-run-20260515_123145-y2alwjkp` 返回 `done`；
- `bfs6vaz4d6` 首轮 sync 和手动刷新 sync 都返回：

```text
Syncing: https://wandb.ai/veralee/flashvsr/runs/y2alwjkp ... done.
```

conductor 经验更新：

- 主机、从机、本地通常都有 conductor 凭证；
- 报 `Unable to locate credentials` 时，优先怀疑是非登录/非交互 shell 没继承环境；
- 远端先用正常 `zsh -lc 'conductor s3 ls ...'` 验证；
- 本机需要先在 zsh 里 `proxy_off`，再测 conductor。
- 如果本机当前 shell 里 `bolt task list` 卡住，到本机 `lxh` tmux session 新开窗口，先 `proxy_off`，再运行 `bolt task list`。

## 2026-05-17 补充：v7-D4.1 turn-isolated DMD runner 给同事审阅

用户要求：

- 希望最后的 `v7-D4.1` 除了 FlashVSR 本身必须有的 pixel/reconstruction loss 以外，其他 DMD 相关组织方式尽量和 DMD2 官方 runner 相似；
- 目标不是“工程上能跑就算”，而是尽量减少与官方 DMD/DMD2 训练方式的无谓偏差。

当前新增文件：

- 训练代码：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
- 48 卡配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_turnisolated_dfake5_offlinewandb.yaml`
- 48 卡启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-TurnIsolated-Dfake5-OfflineWandb.sh`

### D4.1 相比 D3.2 / D4 的关键变化

D3.2 和最终回退后的 D4 都偏工程稳定：

- `student_loss`、`dmd_student_loss`、`fake_loss` 在同一个 runner iteration 内组织；
- 通过 `detach()` / `torch.no_grad()` 保证梯度归属大体正确；
- 但整体形态仍不像 DMD2 官方那样清晰地区分 generator optimizer 和 fake/guidance optimizer 的更新阶段。

D4.1 改成 turn-isolated：

- generator/student turn：
  - 只计算和回传 student 侧 loss；
  - 只 step student optimizer；
  - 不计算、不回传 fake FM loss。
- fake-only turn：
  - student forward 放在 `torch.no_grad()` 里，只用于得到当前 student fake latent distribution；
  - fake FM loss 使用 `student_z_pred.detach()`；
  - 只回传 `G_fake`；
  - 只 step fake optimizer；
  - 不更新 student。

当前 `stage3_dfake_gen_update_ratio: 5` 在 D4.1 中的语义是：

```text
dfake_fake_updates_per_generator = 5

runner_step=0: generator/student update
runner_step=1: fake-only update
runner_step=2: fake-only update
runner_step=3: fake-only update
runner_step=4: fake-only update
runner_step=5: fake-only update
runner_step=6: next generator/student update
```

这比之前直接把 `stage3_fake_update_every_n_steps` 改成 5 的语义正确：现在数字越大，代表 fake/guidance 相对 generator 更新越频繁，和 DMD2 的 `dfake_gen_update_ratio` 意图一致。

### 和 DMD2 官方相似的地方

当前 D4.1 已经做到：

- 两个 optimizer 物理隔离：
  - student/generator optimizer 只吃 FlashVSR reconstruction loss + DMD student loss；
  - fake/guidance optimizer 只吃 fake FM loss。
- fake/guidance 更新使用 detached student fake sample：
  - 对应 DMD2 中用 generator 产生的 fake sample 来训练 fake score/guidance model；
  - 不让 fake loss 反向污染 generator/student。
- DMD student loss 不更新 `G_real/G_fake`：
  - `G_real` 是 frozen Stage1 teacher；
  - `G_fake` probe 在 DMD student loss 路径中走 no-grad；
  - DMD student loss 只作为 student/generator 的优化信号。
- fake 更新频率可以高于 generator：
  - D4.1 的 `dfake_fake_updates_per_generator=5` 实际是 1 次 student 更新配 5 次 fake 更新。

### 仍然和 DMD2 官方不完全一样的地方

需要诚实保留这些边界：

- FlashVSR student/generator loss 不可能只剩 DMD loss：
  - 我们仍保留 Stage3 原本的 flow/MSE/LPIPS/first-frame pixel/first-frame LPIPS；
  - 这些属于 FlashVSR 的 reconstruction/pixel supervision，是用户明确希望保留的部分。
- D4.1 不是 DMD2 官方同一个 iteration 内“先 generator 再 guidance”的完全同构写法：
  - 官方 DMD2 runner 在一个训练循环中可以先更新 generator，再用 fake sample 更新 guidance；
  - 我们之前尝试把这种物理两段 backward 直接搬进当前 DeepSpeed ZeRO2 runner，出现过 ZeRO2 `IndexError`，改普通 fake backward 后又在第二个 generator turn 遇到 NCCL abort；
  - 所以 D4.1 采用跨 runner turn 的物理隔离，避免在同一个 ZeRO2 iteration 混用 student DeepSpeed backward 和手工 fake backward。
- fake-only turn 需要重新跑一次 no-grad student forward：
  - 这是为了每次 fake 更新都看到当前数据/当前 student 分布；
  - 代价是算力更重，GPU 利用率和吞吐需要单独观察。

### 当前我对“已经做到没有明显问题了吗”的回答

从代码组织原则看，D4.1 已经比 D3.2 / D4 更接近 DMD2 官方：

- 梯度图隔离更清楚；
- optimizer ownership 更清楚；
- fake 更新频率语义和 DMD2 更一致；
- 除了 FlashVSR 必须保留的 reconstruction/pixel loss 外，DMD 相关部分没有再故意混成一个 total backward。

但还不能说“已完全验证没问题”：

- 目前只做了本地语法检查：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-TurnIsolated-Dfake5-OfflineWandb.sh`
- 还没跑远端 2GPU smoke；
- 正式训练前至少要 smoke 到 `runner_step=0..6`：
  - 确认日志顺序是 `generator, fake, fake, fake, fake, fake, generator`；
  - 确认第二个 generator turn 不再触发 DeepSpeed/NCCL 问题；
  - 确认 fake-only turn 只更新 fake optimizer，generator turn 只更新 student optimizer。

另外，用户计划 D4 系列正式训练换新的 Stage1 pretrain。当前 D4.1 launch 默认仍指向旧 Stage1：

```text
train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

正式启动 D4.1 前必须替换 `STAGE3_REAL_CHECKPOINT` 与 `STAGE3_FAKE_CHECKPOINT` 到新的 Stage1 结果。

## 2026-05-17 Codex 0 补充：D4.2 以后固定使用 USMGT Stage1 step-3000

用户明确要求：`v7-D4.2` 以及后续 Stage3 训练/验证，不再默认使用旧 48GPU Stage1 `step-10000`，而是统一切换到新的 Takano20250205 USMGT Stage1 微调模型：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors
```

该 ckpt 在 BFS 机器本地。为保证其他机器也能复用，当前脚本默认 S3 fallback 为：

```text
s3://lxh/tmp/usmgt_stage1_takano20250205_step3000_20260517/step-3000.safetensors
```

已做的本地改动：

- 新增 D4.2 release config：
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_turnisolated_dfake5_offlinewandb.yaml`
- 新增 D4.2 release launch：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-TurnIsolated-Dfake5-OfflineWandb.sh`
- 更新 GradCheck launch 默认 Stage1 来源：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-1.sh`

注意：稳定训练源码 `train_flashvsr_stage3_v7_d4_1_lora.py` 没有被塞入 GradCheck 临时逻辑。后续验证必须继续使用独立复制文件，例如：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_gradcheck_lora.py
```

同事后续如果要做第二组/第三组 loss 或 ghost 验证，也应使用上述 USMGT Stage1 `step-3000` 作为默认 `G_real/G_fake` 来源。

## 2026-05-17 Codex 补充：FlashVSR 公式与 DMD2 runner 调度的关系

用户追问：FlashVSR 论文把多个 loss 写在一个公式里，是否说明工程上应该把所有 loss 放在一起回传，只靠不同 optimizer 隔离；还是应该像 DMD2 一样两个模型轮换更新。

我的判断如下：

1. FlashVSR 论文里的总目标是 objective-level 表达，不是 optimizer step 伪代码。
   - 论文可以把 reconstruction、DMD/student、fake critic 目标写在同一个公式里；
   - 这只能说明这些项共同定义训练目标，不能推出工程实现必须一次 `total_loss.backward()`；
   - 特别是 `L_FM(z_pred, G_fake)` 这类 fake diffusion critic 训练项，天然属于 `G_fake/guidance` optimizer，不应反向更新 student。

2. 真正可执行的参考应看 DMD2 官方 runner。
   - `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py` 中：
     - `COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0`
     - generator phase：`accelerator.backward(generator_loss)`，然后 `optimizer_generator.step()`
     - guidance/fake phase：每个 runner step 都执行 `accelerator.backward(guidance_loss)`，然后 `optimizer_guidance.step()`
   - 所以 DMD2 是“同一个 runner iteration 内的两段式更新”：可选 generator，再必做 guidance/fake；不是把 generator loss 和 fake loss 合成一个总 loss 一次 backward。

3. 因此 D4.1 应按两段式组织，而不是简单版。
   - 最新 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py` 已改为：
     - generator/student phase：`student_loss + dmd_student_loss` 单独 `accelerator.backward(total_loss)`，只 step student optimizer；
     - fake/guidance phase：使用 detached `student_z_pred` 计算 fake FM loss，单独 backward，且只 step fake optimizer；
     - fake 模型因 Accelerate 1.12 + DeepSpeed 不支持同一 `Accelerator()` prepare 多模型，采用 PyTorch DDP 包装，student 仍走 DeepSpeed。

4. 旧文档里关于 dfake=5 的 runner 序列需要更正。
   - DMD2 官方 `step % ratio == 0` 语义下：
     - `runner_step=0`: generator+fake
     - `runner_step=1..4`: fake-only
     - `runner_step=5`: generator+fake
   - 不是 `0` 后等到 `6` 才下一个 generator。

当前 smoke 状态：

- 远端机器：`6ikhpjzv3z`
- tmux：`stage3_smoke:d41accel_06`
- run tag：`20260517_d41_smoke_g01_accel06`
- run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_accel06`
- 已确认日志：
  - `dfake_gen_update_ratio=5`
  - `fake_ddp=1`
  - `distributed_timeout_seconds=7200`
- 结果：`d41accel_06` 失败在第一个 runner 的 fake phase。
- 具体原因：
  - 我尝试把 fake phase 也改成 `accelerator.backward(fake_loss)`，希望更像 DMD2 官方；
  - 但在当前项目里 student 是 DeepSpeed ZeRO2 engine，`G_fake` 是独立 DDP 模型；
  - Accelerate 遇到 DeepSpeed distributed type 时会把 `accelerator.backward(fake_loss)` 路由到 student DeepSpeed engine，ZeRO2 会 reduce student optimizer bucket；
  - fake loss 图中没有 student ZeRO2 参数，于是触发 `IndexError: list index out of range`。
- 结论：
  - DMD2 官方的“两个 phase 都 `accelerator.backward`”成立，是因为它的 feedforward/guidance 在同一个官方 runner/model 组织里；
  - 我们当前为了兼容 FlashVSR Stage3 的 student DeepSpeed + trainable fake resident，只能让 student phase 走 DeepSpeed backward，fake phase 走 DDP 自己的 `fake_loss.backward()`；
  - 这仍然是两段式、两个 optimizer、两个梯度图隔离，不是把所有 loss 合并回传的简单版。
- 已改回：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - fake phase 使用 `fake_loss.backward()`，DDP hook 同步 fake 梯度；非 DDP 时保留手工 average fallback。
- `20260517_d41_smoke_g01_ddp07` 结果：
  - `runner_step=0` 的 `generator+fake` 已通过；
  - `runner_step=1..4` 的 fake-only 已通过；
  - 到 runner 5 前，fake DDP process group 触发 PyTorch 默认 600s NCCL watchdog：`WorkNCCL(... ALLREDUCE, NumelIn=287838720 ... Timeout(ms)=600000)`；
  - 这个错误不是 loss 图混合错误，而是 `G_fake` DDP 自己的 process group 没继承 `FLASHVSR_DIST_TIMEOUT_SECONDS=7200`。
- 已补：
  - `_stage3d41_wrap_fake_ddp_if_needed()` 显式 `dist.new_group(... timeout=timedelta(seconds=FLASHVSR_DIST_TIMEOUT_SECONDS))`；
  - DDP 构造传入 `process_group=fake_process_group`。
- 正在跑的新 smoke：
  - `20260517_d41_smoke_g01_pg08`
  - 目标：确认 fake DDP timeout 修正后能跑到 `runner_step=5` 的第二个 `generator+fake`。

`20260517_d41_smoke_g01_pg08` 后续结果：

- runner 0-4 仍然通过；
- fake DDP 的 NCCL watchdog 仍显示 `Timeout(ms)=600000`，说明单独 `dist.new_group(timeout=...)` 没有改变当前 NCCL split group 的实际 watchdog；
- 新补丁：
  - 在 fake phase 的 `fake_loss.backward()` 前，如果 `G_fake` 是 DDP，则先执行 `dist.barrier()`；
  - 这个 barrier 走默认主 process group，目的是把两个 rank 在 fake backward 前对齐；
  - 避免 rank 0 已经进入 287M 元素 fake 梯度 allreduce，而 rank 1 还卡在数据/decode/offload 长尾，导致 rank 0 空等 600s 后被 watchdog 杀掉。

仍需重新 smoke 到 runner 5。

`20260517_d41_smoke_g01_bar09` 后续结果与新的实现判断：

- `bar09` 使用 fake DDP + fake backward 前 barrier；
- runner 0-4 仍正常通过，但之后仍长时间没有进入 runner 5 日志，GPU0/1 出现单 rank 长时间 100% / 0%；
- 这说明 fake DDP hook 路径对当前 89f Stage3 smoke 仍不够稳，不能作为“正规版”交付；
- 已安全停止 `bar09`：
  - 先对 `stage3_smoke:d41bar_09` 发 Ctrl-C；
  - 未响应后只对只读确认出的精确 PID `1153163 1153373 1153374 1153151 1153154` 发 `kill -TERM`；
  - 没有使用 `pkill`、`pgrep | kill` 或模糊正则；
  - GPU0/1 释放，GPU2/3 占卡保持不动。

我对“官方式轮换”的修正判断：

- FlashVSR 论文的 loss 公式是 objective-level，不是 optimizer step 伪代码；
- DMD2 官方 runner 明确是两段式：
  - generator turn 时先 generator/student loss backward + generator optimizer step；
  - 然后每个 runner step 都 guidance/fake loss backward + guidance/fake optimizer step；
- 因此 D4.1 不应把所有 loss 合成一个 total backward；
- 但当前 FlashVSR 架构也不能机械照搬 DMD2 的两个 `accelerator.backward()`：
  - student 已在 DeepSpeed ZeRO2 engine 里；
  - `G_fake` 独立于该 engine；
  - `accelerator.backward(fake_loss)` 会被路由进 student ZeRO2，已在 `d41accel_06` 触发 `IndexError: list index out of range`；
- 新实现采用：
  - student/generator phase：`accelerator.backward(student_loss + dmd_student_loss)`，只 step student optimizer；
  - fake/guidance phase：`fake_loss.backward()`，然后显式 `_average_stage3c_fake_gradients()`，只 step fake optimizer；
  - `G_fake` 初始化后通过 rank0 `dist.broadcast()` 同步参数/buffer；
  - 日志应显示 `fake_ddp=0 fake_manual_grad_sync=1`。

这个实现不是“简单版合 loss”，也不是 DDP hook 版本；它是当前 DeepSpeed student + standalone fake critic 架构下更稳的 DMD2-style 两段式实现。仍需 `bar10` smoke 跑过 runner 5 确认第二个 generator turn。

`20260517_d41_smoke_g01_bar10` 后续结果：

- bar10 确认进入新路径：`fake_ddp=0 fake_manual_grad_sync=1`；
- runner 0-4 均通过：
  - runner 0 是 `generator+fake`；
  - runner 1-4 是 fake-only；
- 但仍在 runner 4 后触发同样量级的 NCCL watchdog：
  - `WorkNCCL(... OpType=ALLREDUCE, NumelIn=287838720, Timeout(ms)=600000)`；
- 这说明 DDP hook 不是唯一问题；手写 `_average_stage3c_fake_gradients()` 如果对单个超大 fake 参数直接整块 all_reduce，仍然会制造一个 287M 元素的 NCCL work；
- 已修正为分块 fake grad sync：
  - 默认 `FLASHVSR_STAGE3_FAKE_GRAD_SYNC_CHUNK_NUMEL=4194304`；
  - 每个 fake grad flatten 后按 4M 元素切块；
  - 每个 chunk 单独 `dist.all_reduce(SUM)`，再除以 world size；
  - 同步后显式 `torch.cuda.synchronize()`；
- 清理 bar10 时只使用精确 PID：
  - TERM: `1260242 1260452 1260453 1260230 1260233`
  - KILL: `1260242 1260452 1260453`
  - 没有使用模糊 kill。

新 smoke：

- run tag：`20260517_d41_smoke_g01_bar11`
- run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_bar11`
- 目标：确认分块 fake grad sync 是否能跑过 runner 5 第二个 generator turn。

`20260517_d41_smoke_g01_bar11` 后续结果：

- runner 0-4 均通过；
- 分块 fake grad sync 后，fake-only 路径没有再由 fake DDP hook 触发失败；
- 但 runner 5 仍失败，且 traceback 已经明确不是 fake phase：
  - 失败点：`accelerator.backward(total_loss)`；
  - 栈：DeepSpeed ZeRO2 `allreduce_bucket()` -> `torch.distributed.all_reduce()`；
  - 报错仍是 `WorkNCCL(... NumelIn=287838720, Timeout(ms)=600000)`；
- 新结论：
  - D4.1 当前 loss/optimizer 组织已经是两段式，不是合 loss；
  - fake optimizer ownership 已隔离；
  - 剩余 smoke 阻塞是 runner 5 的 generator/student phase 进入 DeepSpeed ZeRO2 allreduce 前 rank skew 太大。

新补丁：

- 在 generator/student phase 的 `accelerator.backward(total_loss)` 前加入 `dist.barrier()`；
- 目的：用默认主 process group 先等慢 rank 对齐，再进入 DeepSpeed ZeRO2 的 PG ID 1 allreduce；
- fake phase 继续使用分块 `_average_stage3c_fake_gradients()`。

注意：这不是改变 loss 数学定义，而是解决 Stage3 真实数据/decode/offload 导致的多 rank 同步时序问题。

`20260517_d41_smoke_g01_bar12` 结果：

- 使用 generator backward 前 barrier + fake grad 分块同步；
- runner 0-4 正常通过；
- runner 4 后没有再立刻触发 DeepSpeed ZeRO2 watchdog，但也长时间没有进入 runner 5；
- 进程保持运行，GPU0 长时间 100%、GPU1 0%，因此判断为 barrier/上游 rank skew hang；
- 这说明 barrier 避免了原先直接进 DeepSpeed allreduce 的 600s timeout，但还没有解决 rank 长尾本身；
- 已安全停止 bar12，未使用模糊 kill。

新增诊断：

- `FLASHVSR_STAGE3_SYNC_DEBUG=1`，默认关闭；
- 打印每个 rank 的：
  - `before_next_data`
  - `after_next_data`
  - `enter_turn`
  - `before_generator_barrier`
  - `after_generator_barrier`
- 新 smoke `20260517_d41_smoke_g01_bar13_syncdebug` 早期观察：
  - runner 0 rank1 先到 `before_generator_barrier`；
  - rank0 后到 barrier；
  - 两 rank 都成功打印 `after_generator_barrier`；
  - runner 1 fake-only 也正常进入；
- 初步说明：
  - turn 没有一开始就分叉；
  - generator barrier 本身不是立即死锁；
  - 真正需要定位的是后续 runner 4 -> runner 5 的数据/前向长尾。

## 2026-05-18 给同事：D4.2 single-runner dfake=5 版本

我这边没有继续硬修 D4.1 的彻底 turn-isolated runner。原因是 D4.1 已经证明“数学/optimizer ownership 可以拆开”，但在当前 FlashVSR Stage3 的真实 89f + DeepSpeed ZeRO2 + resident fake/teacher 组合下，两段式 runner 很容易被 rank skew、fake 大梯度 collective、DeepSpeed allreduce 时序拖进 NCCL watchdog 或长 hang。

D4.2 的取舍是：形式上回到单 runner，但保留 DMD2 的核心梯度归属。

- fake critic 每个 runner step 都更新；
- student/generator 只在 `runner_step % stage3_dfake_gen_update_ratio == 0` 时更新；
- `stage3_dfake_gen_update_ratio=5` 的 runner 序列为：
  - runner 0：student/generator + fake
  - runner 1-4：fake-only
  - runner 5：student/generator + fake
- fake loss 使用当前 step 的 `student_z_pred`，但在 `_stage3c_fake_fm_loss()` 内部 detach，因此 fake loss 不向 student 传梯度；
- student backward 只包含 `student_loss + dmd_student_loss`，走 `accelerator.backward(...)` 和 student optimizer；
- fake backward 只包含 fake FM loss，走普通 `fake_loss.backward()` + 显式 fake grad averaging，然后只 step fake optimizer。

这个版本不是把所有 loss 又合成一个 total backward；它是单 runner 调度下的两套 optimizer/两张梯度图隔离。

相关文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_singlerunner_dfake5_offlinewandb.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-SingleRunner-Dfake5-OfflineWandb.sh`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-2-SingleRunner-Dfake5.sh`

Stage1 teacher/fake 初始化已换成用户指定的新 USMGT pretrain：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- S3 fallback：`s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors`

Smoke 过程：

- 第一轮 `20260518_d42_smoke_g01b` 跑过 runner 0-4，但没等到 runner 5；原因不是 D4.2 更新逻辑报错，而是 smoke config 仍 50% 抽 Yubari conductor，`dataset_num_workers=0` 下 `next(data)` 极慢。已用 tmux Ctrl-C 停止，没有使用模糊 kill。
- 我把 D4.2 smoke config 改成只走本地 Takano manifest：`yubari_video_prob=0.0`、`takano_video_prob=1.0`。正式 release config 没改。
- 第二轮 `20260518_d42_smoke_g01c_takano` 完整通过到 runner 5。

关键 smoke 日志：

- runner 0：`generator_update=1`，`loss=1.084645`，`student=1.038355`，`fake_loss=0.04628955`，`dmd_student=0.000000`
- runner 1-4：fake-only，fake loss 约 `0.021428 / 0.013311 / 0.006936 / 0.082231`
- runner 5：第二次 generator turn 通过，`generator_update=1`，`loss=1.557251`，`student=1.426915`，`fake_loss=0.10539128`，`dmd_student=0.024945`，`dmd_grad=0.171409`，`dmd_skip=0`
- `step-1.safetensors` 和 `step-2.safetensors` 均保存成功，进程正常退出。

我的结论：

- D4.2 比 D4.1 更适合作为下一轮 48 卡候选，因为它保留 DMD2 的 dfake=5 更新语义和梯度归属，同时避免 D4.1 turn-isolated/DDP fake 版本暴露出的 rank 对齐与大 collective 问题。
- 这个结论不等于“完全数学证明了 Stage1 teacher wrapper 等价”；完整 deterministic forward 数值等价仍是未完成边界。
- D4.2 的 smoke 数据源改动只用于 smoke 提速，正式训练数据比例需要按 release config 审阅后再启动。

## 2026-05-18 给同事：D4.2 teacher 对齐规则纠正为前 22

用户明确纠正了我之前“teacher 后 22 对齐 student 22”的判断。新的结论如下：

- Stage2 v6.4 student 的 22 个 latent 不是对齐 `GT 89 -> VAE 23 -> drop z0` 的后 22；
- v6.4 实际监督目标是 `GT 前 85 帧 -> WAN VAE -> 22 latents`；
- 因此 Stage3 里用 Stage1 `nonstreaming_aligned` teacher 时，teacher 89 帧会产生 23 positions，但应保留 teacher 前 22 个 positions `[0,22)` 对齐 student `[0,22)`，丢弃 teacher 最后一个 position `[22,23)`。

我已在 D4.2 新线上修正：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
  - `nonstreaming_aligned` teacher alignment 改为 `trim_tail_to_match`。
- `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
  - 新增 `trim_tail_to_match` 支持。
  - debug 日志会显示 `keep_teacher_positions=[0,22)` 和 `drop_teacher_positions=[22,23)`。

本地 `py_compile` 已通过。

同时新增中文验证计划：

- `doc/flashvsr_stage3_v7d42_validation_plan_20260518.md`

该文档把后续实验编号为 E0-E7：

- E0：D4.2 前 22 对齐 smoke；
- E1：Stage1 teacher deterministic forward 数值等价；
- E2：梯度归属/互不干扰检查；
- E3：loss 有效回传检查；
- E4：pixel 梯度与 DMD 梯度冲突检查；
- E5：DMD 方向性 sanity check；
- E6：48 卡 100-step 四组 ablation；
- E7：10 个合成测试集视觉/指标对照。

注意：旧 D3/D4.1 线目前仍保留原历史实现，没有在这次一起改，避免混淆已有结果。

## 2026-05-18 给同事：D4.3 dual DeepSpeed engine 尝试结果

用户要求验证更干净的实现：student/generator 一个 DeepSpeed engine，trainable `G_fake` 一个 DeepSpeed engine；两个 optimizer、两个 scheduler、两个 checkpoint/save/load 路径分开。

我先澄清：之前 D4.1 不是这个方案。D4.1 是 turn-isolated / 双 optimizer 路线，没有把 `G_fake` 单独交给 DeepSpeed engine 管。

D4.3 新增文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_3_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-3-DualEngine-Dfake5.sh`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_3_lora_89f_videoonly_usmgtpretrain_dualengine_dfake5_offlinewandb.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-3-Lora-89f-VideoOnly-USMGTPretrain-DualEngine-Dfake5-OfflineWandb.sh`

实现要点：

- student 仍走现有 `accelerator.prepare(model, optimizer, ...)`；
- `G_fake` 在 student prepare 后用 `deepspeed.initialize(...)` 单独包成 fake engine；
- fake phase 改成 `fake_engine.backward(fake_loss)` / `fake_engine.step()`；
- fake DeepSpeed state 单独保存到 `output/stage3_fake_deepspeed/`；
- 继续使用 Stage1 USMGT step-3000 初始化 real/fake；
- teacher 对齐仍是用户纠正后的前 22：`keep_teacher_positions=[0,22)`、`drop_teacher_positions=[22,23)`。

smoke 过程里修了两个问题：

- `ZeRO-Offload + client-provided AdamW` 被 DeepSpeed 拒绝，已在 fake DS config 加 `zero_force_ds_cpu_optimizer=False`；
- `G_fake` 包成 DeepSpeedEngine 后，原来的 `isinstance(fake_model, FlashVSRStage3BTrainingModule)` 判断导致 `fake_update=0`，已改成先 `_unwrap_stage3c_model(fake_model)` 再判断底层 module。

当前有效 smoke：`20260518_d43_dualengine_e2`，机器 `6ikhpjzv3z` GPU0/1。

关键日志：

- runner 0：
  - `generator_update=1`
  - `loss=1.069140`
  - `student=1.038355`
  - `fake_loss=0.03078421`
  - `fake_update=1`
  - `real_probe=0.115747`
  - `fake_probe=0.149991`
- runner 1 fake-only：
  - `generator_update=0`
  - `fake_loss=0.00752554`
  - `fake_update=1`
- timing：
  - runner 0：`fake=8.778s`、`fake_backward_sync=175.037s`、`student_backward=7.823s`
  - runner 1：`fake=8.750s`、`fake_backward_sync=46.805s`

我的判断：

- D4.3 证明“真正 dual DeepSpeed engine”在工程上能跑起来，`G_fake` 可以不再手写 fake grad all-reduce；
- 但当前速度没有比 D4.2 好，fake backward 仍是主要耗时；
- 因此 D4.3 现在应该作为继续优化/验证线，不建议直接替代 D4.2 上 48 卡；
- 如果继续推进 D4.3，下一步应查 fake engine 的 ZeRO-Offload 配置、optimizer 选择、checkpoint 开销，以及 rank 等待是否仍来自 fake FM full-attention backward。

## 2026-05-18 给同事：e2/e3 不是版本，双 DeepSpeed 参考复查

先把命名说清楚，避免后面讨论混乱：

- `e2` / `e3` 是 D4.3 dual-engine smoke 的实验后缀，不是代码版本；
- 代码版本仍是 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_3_lora.py`；
- `e2` 用 fake DeepSpeed 原始 ZeRO2/offload 配置；
- `e3` 在 fake DeepSpeed engine 上关闭 CPU/param offload，日志会显示 `fake_ds_offload=0`。

实验结果：

- `e2` run：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5_20260518_d43_dualengine_e2`
  - runner 0：`fake_loss=0.03078421`、`fake_update=1`、`fake_backward_sync=175.037s`
  - runner 1 fake-only：`fake_loss=0.00752554`、`fake_update=1`、`fake_backward_sync=46.805s`
- `e3` run：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5_20260518_d43_dualengine_nooffload_e3`
  - runner 0：`fake_loss=0.03078421`、`fake_update=1`、`fake_backward_sync=174.472s`、`optim=0.034s`

结论：

- 关 fake offload 只明显改善了 optimizer step 时间；
- 首个 generator turn 的 fake backward/sync 仍约 174-175s，核心瓶颈没有消失；
- 因此 e3 不能证明 D4.3 已经适合上 48 卡，只能说明 fake engine offload 不是主要慢点。

我重新查了三个参考：

1. DeepSpeed 官方 Training API：
   - 支持 multiple models；
   - 核心写法是每个模型一个 `DeepSpeedEngine`；
   - 分别 `engine.backward(loss)` / `engine.step()`。
2. HuggingFace Accelerate 官方 multiple DeepSpeed models：
   - 支持多个 `DeepSpeedPlugin`；
   - 对 disjoint models，官方说明需要第二个 `Accelerator`，因为一个 `Accelerator` 同时只能携带一个 DeepSpeed engine/plugin。
3. 本地 DMD2 官方代码：
   - 路径：
     - `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py`
     - `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_unified_model.py`
   - DMD2 本身不是 dual DeepSpeed engine；
   - 它是 `feedforward_model` / `guidance_model` 两个子模型，两套 optimizer/scheduler，经 Accelerate/DDP/FSDP 管；
   - 训练语义是：
     - `self.step % self.dfake_gen_update_ratio == 0` 时更新 generator；
     - 每步更新 guidance/fake；
     - guidance/fake 训练输入用 `generated_image.detach()`；
     - generator turn 会临时冻结 guidance model。

因此当前判断是：

- D4.3 的 direct `deepspeed.initialize()` 方案符合 DeepSpeed 官方 multiple-engine API；
- 但它不是“照 DMD2 原代码抄”，因为 DMD2 没用双 DeepSpeed；
- 如果继续追求更贴近 Accelerate 官方的双 DeepSpeed 写法，下一条线应该是两个 `DeepSpeedPlugin` + 两个 `Accelerator` 的 D4.4 风格实验；
- 这会比 D4.3 改动更大，需要单独 smoke，不应该直接覆盖 D4.2 或 D4.3。

## 2026-05-18 给同事：D4.4 dual Accelerator + dual DeepSpeedPlugin 结果

用户要求继续尝试更贴近 Accelerate 官方 multiple DeepSpeed models 的方案。我已写 D4.4，不覆盖 D4.2/D4.3。

参考依据：

- HuggingFace Accelerate 官方 multiple DeepSpeed models 文档：训练多个 disjoint models 时，需要两个 `DeepSpeedPlugin`，并且需要第二个 `Accelerator`，因为一个 `Accelerator` 同时只能携带一个 DeepSpeed engine/plugin。
- DeepSpeed 官方 training API：支持 multiple models/engines，D4.3 的 raw `deepspeed.initialize()` 是这条路线。
- 本地 DMD2 官方代码不是 DeepSpeed 双 engine；它是 `feedforward_model` / `guidance_model` 两套 optimizer/scheduler，经 Accelerate/DDP/FSDP 管理。D4.4 只能借鉴 DMD2 的更新语义，不能直接照抄 DMD2 的 DeepSpeed 双 engine 代码。

D4.4 新文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`

关键实现：

- `main()` 里构造 `{"student": DeepSpeedPlugin(...), "fake": DeepSpeedPlugin(...)}`；
- 主 `accelerator` 选择 `student` plugin，包 student/generator；
- 第二个 `fake_accelerator = Accelerator()` 选择 `fake` plugin，包 trainable `G_fake`；
- student phase 用 `accelerator.backward(student_total_loss)`；
- fake phase 用 `fake_accelerator.backward(fake_loss)`，随后只 step fake optimizer/scheduler；
- fake loss 仍使用 `_stage3c_fake_fm_loss()`，内部对当前 `student_z_pred` detach；
- fake checkpoint 仍单独写 `output/stage3_fake_deepspeed/`；
- fake DeepSpeed 默认关闭 offload，日志显示 `fake_ds_offload=0`。

远端检查：

- 机器：`6ikhpjzv3z`
- `accelerate==1.12.0`
- `Accelerator` 支持 `deepspeed_plugins` 参数；
- `AcceleratorState.select_deepspeed_plugin` 存在；
- 本地和远端 `py_compile` 通过，D4.4 smoke/release launch script `bash -n` 通过。

smoke：`20260518_d44_dualaccel_e0`

- run dir：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_dualaccel_e0`
- 日志确认：
  - `D4.4 dual Accelerate DeepSpeed engine`
  - `fake_backward=fake_accelerator_deepspeed`
  - `fake_ds_zero_stage=2`
  - `fake_ds_offload=0`
- runner 0：
  - `loss=1.069140`
  - `student=1.038355`
  - `fake_loss=0.03078421`
  - `fake_update=1`
  - `real_probe=0.115747`
  - `fake_probe=0.149991`
  - timing：`data=127.236s`、`student=9.485s`、`probe=18.013s`、`dmd=18.062s`、`fake=9.017s`、`fake_backward_sync=172.442s`、`student_backward=9.438s`
- runner 1 fake-only：
  - `fake_loss=0.00751893`
  - `fake_update=1`
  - timing：`data=57.597s`、`student=7.800s`、`fake=9.002s`、`fake_backward_sync=46.520s`

结论：

- D4.4 已经实现用户要求的“两个 Accelerator + 两个 DeepSpeedPlugin”，并通过 2GPU smoke 到 runner 1；
- 包装方式更官方，但没有解决性能问题；
- D4.4 的 fake backward 时间与 D4.3 同量级：runner 0 约 172s，runner 1 fake-only 约 46s；
- 因此当前瓶颈不在 raw DeepSpeed vs Accelerate，也不在 fake optimizer offload，而更可能在 trainable full-attention `G_fake` 的大 backward/ZeRO2 gradient sync 本身，或 Stage3 每步都跑 heavy fake FM 的结构成本；
- D4.4 不应直接上 48 卡，除非后续接受这个成本或进一步改变 fake 更新/同步策略。

## 2026-05-18 给同事：D4.4 fake 不是 full WAN 训练，flash-attn 也已生效

用户提出一个重要怀疑：`fake_backward_sync` 这么慢，会不会是我们把 full WAN body 也放进了 fake optimizer / ZeRO sync，或者 fake full-attention 没走 flash-attn。

我补了显式参数分组日志，并用 Stage1 v5.3.5 同款 no-offload + activation checkpointing DeepSpeed config 重新 smoke：

- 代码：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`
- smoke run：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_nooffload_actckpt_e1`

日志结论：

- fake DeepSpeed：
  - `zero_stage=2`
  - `offload_optimizer_device=None`
  - `offload_param_device=None`
  - `has_activation_checkpointing=True`
- fake 可训练参数：
  - `fake_trainable_params=570961408`
  - `fake_trainable_groups={"lora": 283115520, "lq_proj_in": 287845888}`
  - 没有 `dit_base_unexpected`
- fake attention debug：
  - `branch=flash_attn_2`
  - shape `(1, 84480, 12, 128)`
  - dtype `torch.bfloat16`

所以这里可以比较确定地排除两件事：

- 没有误把 full WAN/DiT base body 打开训练；当前 fake trainable 范围和 Stage1 LoRA 训练一致，都是 LoRA + `lq_proj_in`。
- fake dense_full attention 已经走 `flash_attn_2`，不是因为 fallback 到慢 attention。

但性能仍然没有改善：

- runner 0：
  - `fake_backward_sync=172.897s`
  - `student_backward=6.129s`
- runner 1 fake-only：
  - `fake_backward_sync=45.945s`

新的判断：

- `fake_backward_sync` 的根因不在“全模型误训”、不在 fake offload、也不在缺 flash-attn；
- 更可能在 570M trainable 参数的 full-attention fake backward + ZeRO2 gradient reduction，或 dual-engine/dual-Accelerator 下 fake 分支 collective 等待；
- 如果继续定位，建议做一个同机同 2GPU 的 Stage1 v5.3.5 backward timing 对照。若 Stage1 同样慢，则是 Stage1-like fake backward 本身成本；若 Stage1 明显快，则继续 profile D4.4 fake backward 和 dual-engine 通信路径。

清理状态：

- e1 smoke 已停止；
- 没有使用 `pkill` 或模糊 kill；
- GPU0/1 已恢复占卡。
