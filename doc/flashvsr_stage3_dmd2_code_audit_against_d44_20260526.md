# DMD2 代码拆解与 FlashVSR Stage3 D44 风险对照 2026-05-26

## 0. 这份文档回答什么

你现在不是要再看一份更复杂的 D44 代码清单，而是要先搞清楚 DMD2 原作者代码到底怎么做，然后拿它逐条回答我们 D44 里面的风险点。

本文件只做三件事：

1. 把 DMD2 的训练原理按代码拆开讲清楚；
2. 把 `flashvsr_stage3_d44_dmd_code_audit_20260526.md` 里的 DMD/fake 风险点逐条映射到 DMD2；
3. DMD2 里找不到答案的地方，明确列成后续实验项。

DMD2 源码位置：

`/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2`

核心代码：

| 文件 | 作用 |
| --- | --- |
| `main/train_sd.py` | 训练循环，决定 generator 和 fake/guidance 的更新顺序 |
| `main/sd_unified_model.py` | 把 generator、real score、fake score 包成一个训练模块 |
| `main/sd_guidance.py` | DMD loss 和 fake loss 的真正公式 |
| `main/edm/edm_guidance.py` | EDM 版本，同样能验证 DMD 公式方向和 fake loss 语义 |

## 1. DMD2 里有哪三个模型

DMD2 的名字容易让人误解。它不是一个普通的 teacher-student loss，而是三个角色：

| 角色 | DMD2 代码名字 | 是否训练 | 我们 FlashVSR 对应 |
| --- | --- | --- | --- |
| Student / Generator | `feedforward_model` | 训练 | Stage3 student，初始化自 Stage2 641/USMGT pretrain |
| Real score teacher | `guidance_model.real_unet` | 不训练 | `G_real`，Stage1 teacher，frozen |
| Fake score critic | `guidance_model.fake_unet` | 训练 | `G_fake`，fake critic，学习当前 student 生成分布 |

DMD2 的核心思想是：

- student 先生成一个 fake clean latent/image；
- 把这个 fake clean latent 加噪到某个 timestep；
- frozen `G_real` 预测这个 noisy fake 应该往真实数据分布哪里去；
- trainable `G_fake` 预测这个 noisy fake 在当前 fake 分布里应该往哪里去；
- 两个方向的差值就是给 student 的 DMD 梯度；
- 同时，`G_fake` 自己也要不断训练，否则它不能代表当前 fake distribution。

一句话版本：

`G_real` 告诉 student “真实分布往哪边”，`G_fake` 告诉 student “你现在的假分布往哪边”，两者相减，student 才知道怎么从假分布推向真分布。

## 2. DMD2 的训练 loop 是什么顺序

代码位置：

`main/train_sd.py:322-408`

每个 DMD2 runner step 做两件事：

1. 可能更新 generator。
2. 一定更新 guidance/fake。

关键代码：

```python
COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0
```

代码位置：

`main/train_sd.py:330`

如果 `dfake_gen_update_ratio=5`：

- step 0：generator 更新一次，fake/guidance 也更新一次；
- step 1-4：generator 不更新，只更新 fake/guidance；
- step 5：generator 再更新一次，fake/guidance 也更新一次。

所以 DMD2 的 “dfake=5” 不是 “generator turn 禁止 fake 更新”。它的真实含义是：

`fake/guidance 每个 runner step 都更新，generator 每 5 个 runner step 才更新一次。`

这点和我们 D4.4 现在的解释一致。

## 3. DMD2 generator turn 做什么

代码位置：

`main/train_sd.py:354-381`

generator turn 先调用：

```python
generator_loss_dict, generator_log_dict = self.model(..., generator_turn=True)
```

然后如果当前 step 需要更新 generator：

```python
generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight
self.accelerator.backward(generator_loss)
generator_grad_norm = accelerator.clip_grad_norm_(self.model.feedforward_model.parameters(), self.max_grad_norm)
self.optimizer_generator.step()
```

代码位置：

`main/train_sd.py:367-381`

重要结论：

- generator/student 只吃 `loss_dm`，不吃 fake loss；
- generator 更新后会清掉 generator 和 guidance 的梯度；
- DMD2 也有 gradient clipping，默认 `max_grad_norm=10.0`；
- DMD2 明确把 generator optimizer 和 guidance optimizer 分开。

## 4. DMD2 guidance/fake turn 做什么

代码位置：

`main/train_sd.py:385-408`

每个 runner step 都会调用：

```python
guidance_loss_dict, guidance_log_dict = self.model(..., guidance_turn=True)
guidance_loss += guidance_loss_dict["loss_fake_mean"]
self.accelerator.backward(guidance_loss)
guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
self.optimizer_guidance.step()
```

关键结论：

- fake/guidance 每个 runner step 都更新；
- fake loss 只更新 guidance/fake，不更新 generator；
- DMD2 不是把 generator loss 和 fake loss 混到同一个 backward 里；
- 但 generator update step 上 fake 也会更新一次。

这回答了我们之前的争论：D4.4 里 `student_update_turn` 上 `fake_delta != 0` 并不一定是错，DMD2 原始代码也是 generator step 后继续做 guidance update。

## 5. DMD2 的 DMD loss 公式

代码位置：

`main/sd_guidance.py:168-255`

核心步骤：

1. 对 student 生成的 clean latent/image 采样 timestep 和 noise。

```python
timesteps = torch.randint(self.min_step, min(self.max_step+1, self.num_train_timesteps), [batch_size])
noise = torch.randn_like(latents)
noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
```

2. fake score model 预测 noisy fake 的 x0。

```python
pred_fake_noise = predict_noise(self.fake_unet, noisy_latents, ...)
pred_fake_image = get_x0_from_noise(noisy_latents, pred_fake_noise, ...)
```

3. real score teacher 预测同一个 noisy fake 的 x0。

```python
pred_real_noise = predict_noise(self.real_unet, noisy_latents, ...)
pred_real_image = get_x0_from_noise(noisy_latents, pred_real_noise, ...)
```

4. 构造两个方向。

```python
p_real = (latents - pred_real_image)
p_fake = (latents - pred_fake_image)
```

5. 构造 DMD gradient。

```python
grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
grad = torch.nan_to_num(grad)
```

6. 用 synthetic target 注入梯度。

```python
loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")
```

DMD2 的 EDM 版本也一样：

`main/edm/edm_guidance.py:102-112`

```python
p_real = (latents - pred_real_image)
p_fake = (latents - pred_fake_image)
weight_factor = torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
grad = (p_real - p_fake) / weight_factor
loss = 0.5 * F.mse_loss(original_latents, (original_latents-grad).detach(), reduction="mean")
```

对我们最重要的结论：

- 我们现在 D44 用 `(p_real - p_fake)` 的符号，和 DMD2 源码一致；
- 我们现在用 `abs(p_real).mean(...)` 做归一化，也和 DMD2 源码一致；
- DMD2 也用 `0.5 * mse(original, original - grad)` 这种 synthetic target trick；
- 所以 OF-D 崩坏不太像是最简单的“DMD 符号写反”。

## 6. DMD2 的 fake loss 是什么

代码位置：

`main/sd_guidance.py:257-310`

核心代码：

```python
latents = latents.detach()
noise = torch.randn_like(latents)
timesteps = torch.randint(0, self.num_train_timesteps, [batch_size])
noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
fake_noise_pred = predict_noise(self.fake_unet, noisy_latents, ...)
loss_fake = torch.mean((fake_noise_pred.float() - noise.float())**2)
```

关键点：

- fake loss 的输入 latent 是 student 生成结果，但先 `detach()`；
- fake loss 不回传 student；
- fake loss 是普通 diffusion/FM 训练目标；
- fake timestep/noise 和 DMD generator loss 里的 timestep/noise 不是同一个采样点；
- DMD2 原代码允许这个不一致。

这回答 D44 audit 中一个风险点：

`fake FM 自己采样 timestep/noise，DMD student loss 另采样 shared dmd_point` 这个设计在 DMD2 里是存在的，不是我们独创的明显 bug。

## 7. DMD2 怎么避免 generator 和 fake 梯度互相污染

DMD2 有三层隔离。

第一层：generator forward 中，算 DMD loss 时冻结 guidance model。

代码位置：

`main/sd_unified_model.py:303-310`

```python
self.guidance_model.requires_grad_(False)
loss_dict, log_dict = self.guidance_model(generator_turn=True, guidance_turn=False, ...)
self.guidance_model.requires_grad_(True)
```

第二层：fake loss 里显式 detach generator output。

代码位置：

`main/sd_guidance.py:267`

```python
latents = latents.detach()
```

第三层：两个 optimizer 分开。

代码位置：

`main/train_sd.py:191-202`

```python
self.optimizer_generator = torch.optim.AdamW(feedforward_model params)
self.optimizer_guidance = torch.optim.AdamW(guidance_model params)
```

我们的 D4.4 对应要求：

- fake-only turn 上 student 参数不能变；
- generator/student turn 上 fake 是否变，要看是否也跑 fake update；
- G_real 永远不能变；
- fake loss 必须用 `clean_latents.detach()`；
- DMD student loss 里 G_real/G_fake probe 必须 no-grad 或等价冻结。

## 8. DMD2 对 gradient accumulation 的态度

代码位置：

`main/train_sd.py:689`

```python
assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"
```

结论很明确：

- DMD2 不支持 gradient accumulation；
- 原因不是数学上绝对不可能，而是 generator/guidance 两套目标交替时，accumulation 很容易把梯度生命周期搞错；
- 我们 D44 系列必须保持 `gradient_accumulation_steps=1`，否则优先判定为高风险。

## 9. DMD2 对 loss weight 的做法

DMD2 SD 训练里 generator loss 是：

代码位置：

`main/train_sd.py:367-370`

```python
generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight
```

默认参数：

`main/train_sd.py:664`

```python
parser.add_argument("--dm_loss_weight", type=float, default=1.0)
```

实验脚本里常见：

- `dfake_gen_update_ratio=5` 或 `10`；
- `max_grad_norm=10.0`；
- `generator_lr/guidance_lr` 可能相同，也可能非常小。

DMD2 本身没有 pixel/LPIPS，因此它不能直接回答我们 `flow + MSE + LPIPS + DMD` 四种 loss 怎么配比。这个是 FlashVSR Stage3 自己新增的问题，需要靠 OF-E、loss grad probe、长短训对比实验回答。

## 10. D44 风险点逐条对照 DMD2

| D44 风险点 | DMD2 源码答案 | 我们当前判断 |
| --- | --- | --- |
| DMD 符号是不是反了 | DMD2 用 `grad=(p_real-p_fake)/abs(p_real).mean` | 我们符号和 DMD2 一致，DMD-2 sign probe 也显示当前符号比反符号更合理 |
| DMD normalization 是不是乱写 | DMD2 同样除以 `abs(p_real).mean` | 公式对齐，但视频 latent 维度和 SD image latent 不同，仍需看 grad 范围 |
| fake loss 是否应该 detach student output | DMD2 明确 `latents=latents.detach()` | 我们必须保持；已在 D44 路径实现 |
| fake loss 是否和 DMD point 共用同一个 timestep/noise | DMD2 不共用，fake loss 自己采样 | 我们不共用不构成直接 bug |
| G_real 是否训练 | DMD2 `real_unet.requires_grad_(False)` | 我们 G_real 必须 frozen/no-grad |
| G_fake 是否训练 | DMD2 `fake_unet.requires_grad_(True)` | 我们 G_fake trainable 是对齐的 |
| fake 是否每 step 更新 | DMD2 每个 runner step 更新 guidance/fake | D44 fake 每 runner step 更新是合理的 |
| generator 是否每 step 更新 | DMD2 按 `dfake_gen_update_ratio` 间隔更新 | D44 student 每 5 runner step 更新是合理的 |
| generator turn 上 fake_delta 是否允许非零 | DMD2 generator step 后仍会 guidance update | 允许，日志命名应叫 `student_update_turn`，不是 fake 禁止更新 turn |
| generator/fake optimizer 是否必须分开 | DMD2 两个 optimizer | 我们 D44 dual optimizer 是必要方向 |
| gradient accumulation 是否安全 | DMD2 assert 只能为 1 | 我们必须保持 1 |
| 是否需要 grad clip | DMD2 对 generator/guidance 都 clip grad norm | 我们 D44 的 DMD spike guard/grad norm guard 有必要 |
| DMD loss 数值是否等于真实优化目标 | DMD2 也是 synthetic target trick | 重点看 `grad` 和参数梯度，不只看 loss 标量 |
| real/fake x0 是否要可视化 | DMD2 记录 `dmtrain_pred_real_image/fake_image` 并可 decode 到 wandb | 我们 DMD-1V decode 是正确排查方向 |

## 11. DMD2 找不到答案的 D44 独有问题

下面这些不是 DMD2 能直接回答的，因为 DMD2 是单图像 SD/EDM，FlashVSR 是视频、LQ 条件、chunk causal attention 和 Wan VAE。

| 我们的问题 | DMD2 是否有答案 | 需要怎么验证 |
| --- | --- | --- |
| `G_real/G_fake` 应该用 Stage1 dense_full 还是 Stage2 streaming/block sparse | 没有 | 对同一 batch decode `G_real_x0/G_fake_x0`，和 Stage1/Stage2 原生推理对齐 |
| fake `lq_proj_in` 是否该训练 | 没有 | DMD4 fakeproj frozen / every1 / every5 三组对比 |
| fake `lq_proj_in` 每 5 step 更新是否合理 | 没有 | 看 DMD4 三组颜色、fake loss、fake_x0 是否漂移 |
| DMD-only 是否应该能 overfit 视频 | DMD2 理论上 DMD 可以独立训练，但 FlashVSR 有 LQ 条件和视频结构 | OF-D / DMD4 长训验证 |
| pixel/LPIPS 和 DMD 混合是否会掩盖 DMD 错误 | DMD2 没 pixel/LPIPS | OF-A vs OF-E，外加每个 loss 的参数梯度量级 probe |
| 89f streaming mask / KV cache 是否要进入 student/fake/real | DMD2 没视频流式结构 | Stage2 641 forward 对齐检查、teacher wrapper 对齐检查 |
| Wan decoder 首帧权重和 selected window decode | DMD2 没 Wan VAE causal decoder | v7-B/Stage3 decode 对齐单独验收 |
| DMD gradient 在视频 latent 上是否应按 token/frame 改 normalization | DMD2 没视频 latent | DMD-2 norm probe，比较 absmean/std/sigma/token normalization |
| DMD 是否应该只在有效 85 帧上算 | DMD2 没 89->85 | 检查 D44 DMD 输入 latent 是否和 641 有效帧语义一致 |

## 12. DMD4 实验到底计划了几种，做了几种

`flashvsr_stage3_dmd_fake_branch_debug_plan_20260526.md` 里 DMD-4 的目的，是检查 fake `lq_proj_in` 是否导致 DMD-only 色偏/灰屏。

计划上应该拆成三种：

| DMD4 变体 | 目的 | 当前状态 |
| --- | --- | --- |
| fake `lq_proj_in` frozen | 看 fake projector 不动时是否还偏色 | 已启动独立 DMD4：`stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_frozen_v7_d4_4_20260525_233413` |
| fake `lq_proj_in` every1 | 看 projector 每步跟 fake LoRA 一起更新是否更稳定 | 已启动独立 DMD4：`stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every1_v7_d4_4_20260525_234500` |
| fake `lq_proj_in` every5/current | 对照 D44/OF-D 默认行为 | 已补跑独立 DMD4：`stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4_20260526_ownership_cmp` |

所以当前状态：

`DMD4 按计划应有 3 组；现在 3 组都已经有独立实验。every5/current 早期 runner5 的 dmd_student=0.932228，runner10 触发 dmd_loss_clamp=1，需要继续看保存点和 validation 视频。`

## 13. 现在 DMD debug 应该怎么继续

根据 DMD2 对照，已经基本排除的简单问题：

- DMD 符号不是明显反；
- `abs(p_real).mean` normalization 不是我们凭空发明；
- fake loss detach student output 是对的；
- fake 每 runner step 更新、student 每 5 step 更新是 DMD2 风格；
- generator turn 上 fake 也更新不是错误。

还需要继续查的高优先级问题：

1. `G_real/G_fake` wrapper 是否和对应 Stage1/Stage2 原生推理语义一致。

   DMD2 没有 LQ projector，所以这是我们最可能偏的地方。

2. `G_fake` 学到的是不是合理 fake score。

   需要看 `fake_x0` 随 fake step 变化的 decode，而不是只看 fake loss 下降。

3. fake `lq_proj_in` 是否引入颜色漂移。

   DMD4 第三组 every5/current 需要补齐，然后和 frozen/every1 比。

4. DMD-only 崩坏是来自 fake critic，还是来自 DMD 梯度注入。

   做法是固定 `G_fake` 不更新，只用 frozen fake probe 跑 DMD-only；以及固定 fake samples 单训 fake critic。

5. DMD 在视频 latent 上的 normalization 是否过强。

   DMD2 的公式对 SD image latent 成立，但视频 latent 的时间维、首帧语义、有效帧裁剪都不同，需要 norm probe。

## 14. 关键澄清：DMD 不是“50-step teacher 直接监督 1-step student”

用户提出一个重要疑问：如果 Stage3 是蒸馏，是否应该让 50-step teacher 输出结果，再监督 1-step student？我们当前是不是错误地用 teacher 的“一步”去指导 student 的“一步”？

对照 DMD2 代码后，结论如下：

- DMD2 训练时不是跑完整多步 teacher sample，再用该 sample 做 pixel target；
- DMD2 的 `G_real / G_fake` 是 score / denoiser / x0 predictor critic；
- student/generator 先生成一个 fake clean sample；
- 然后把这个 fake clean sample 加噪到随机 timestep；
- `G_real` 和 `G_fake` 在同一个 noisy point 上各做一次 denoise/x0 prediction；
- DMD 梯度来自两个 score 的差：
  `grad = ((z - real_x0) - (z - fake_x0)) / mean(abs(z - real_x0))`。

所以 DMD loss 里的 teacher 不是 50-step sampling teacher，而是单次 noisy-point score teacher。这不是偷懒，而是 DMD/VSD 方法的核心：用 diffusion teacher 的 score field 指导 generator，而不是每步都生成完整 teacher 视频。

需要区分：

| 概念 | 是否 50-step | 在 DMD2 / 当前 Stage3 中的作用 |
| --- | --- | --- |
| sampling teacher | 是 | 可用于生成完整高质量样本或 benchmark，不是 DMD loss 的常规路径 |
| score teacher / G_real | 否，单次 noisy-point denoise | 提供 real distribution score |
| G_fake | 否，单次 noisy-point denoise | 学 student fake distribution score |
| student | 目标是 1-step | 输出 clean latent，接受 flow / pixel / LPIPS / DMD 约束 |

因此，“teacher 的一步和答案没关系”这句话要分开理解：

- 如果说 teacher 一步 sample 不等于最终 50-step SR 视频，这是对的；
- 但 DMD 用的不是最终 sample，而是 teacher 在某个 noisy point 的 score / denoise direction；
- 这个 direction 对 generator 优化有意义，DMD2 就是这么实现的。

## 15. student / teacher mask 的边界

用户判断“causal mask 应该进入 student，但不应该进入 teacher”，当前看是合理的。

- student 是 Stage2 streaming 能力的 one-step 版本，应继承 Stage2 的 `block_sparse_chunk_causal` 和 LQ projector 流式语义；
- `G_real/G_fake` 是 critic / score model，不是最终 streaming student；
- 如果它们按当前设计来自 Stage1 teacher，则不应该强行使用 Stage2 causal mask；
- 如果后续改成 Stage2 teacher critic，则需要重新定义其输入、mask、LQ projector cache 和 teacher score 语义，不能混用。

当前风险不是“teacher 没有跑 50 step”，而更像是：

- fake critic 更新后质量/颜色是否稳定；
- fake `lq_proj_in` 是否把 LQ 条件学歪；
- D4.4 dual accelerator 下 fake 更新尺度是否和 D4.2 单 runner 一致；
- DMD score difference 是否在长期训练中变成错误或过强方向。

## 16. G_real / G_fake 差距变大会怎样

会直接推高 DMD。

DMD 梯度来自 `real_x0 - fake_x0` 的差。若 `G_fake` 变黄、偏灰、或偏离 `G_real` 很远，则：

- `real_minus_fake` 变大；
- `dmd_grad` 变大；
- `dmd_student` loss 也可能变大；
- student 会被沿着这个 score difference 强行更新。

如果 `G_fake` 的偏差正确描述 student fake distribution，这是 DMD 该做的事；
如果 `G_fake` 自己学歪了，大 DMD loss 就会变成错误方向，可能导致绿色/灰屏或细节突然消失。

所以后续 fake 分支验证不能只看 `loss_fake_fm` 是否下降，还要看：

- `G_fake` 预测的 `fake_x0` 是否视觉稳定；
- `G_fake` 和 `G_real` 的差是否对应真实 student 缺陷，而不是系统性色偏；
- D4.2/D4.4 fake 更新尺度是否一致；
- fake `lq_proj_in` frozen / every1 / every5 哪个最不容易引入颜色漂移。

## 17. 最短结论

DMD2 源码支持我们 D44 当前几个核心设计：DMD 符号、fake detach、fake 高频更新、student 低频更新、两个 optimizer、gradient accumulation=1。

但 DMD2 不能替我们证明 FlashVSR 的视频条件、`G_real/G_fake` projector、89f streaming 语义、fake `lq_proj_in` 更新方式是对的。OF-D/DMD4 的色偏更像是这些 FlashVSR 独有适配层出了问题，而不是 DMD2 最基础公式写反。
