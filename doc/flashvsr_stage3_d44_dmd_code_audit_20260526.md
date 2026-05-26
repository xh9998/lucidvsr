# FlashVSR Stage3 v7-D4.4 DMD/Fake 代码审阅文档 2026-05-26

## 0. 文档目的

这份文档只回答一个问题：当前 `v7-D4.4` 从 `Stage2 v6.4.1 / 641` 发展到现在，所有和 `DMD / G_real / G_fake / fake critic` 相关的新增逻辑到底在哪里、做了什么、风险在哪里、应该怎么验证。

核心代码：

`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`

验证专用 overfit 代码：

`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py`

当前 40 卡 D4.4 配置：

`wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`

昨天 OF 固定 LQ/GT 过拟合实验使用同一套 overfit 专用入口，不修改正式 D4.4 训练文件。

## 1. 641 基线是什么

`641` 是 Stage2 流式训练版本。它的核心目标是让 student 仍然按 diffusion / flow matching 训练：

- 输入：89 帧 LQ/GT。
- 输出语义：Stage2 streaming 形状，实际有效输出按 85 帧使用。
- 训练目标：`noise_pred` 拟合 scheduler 的 `training_target`。
- 推理/validation：Stage2 风格 streaming / KV-cache / chunk 逻辑。
- 模型主体：DiT + LoRA + `lq_proj_in`。
- 注意力：`block_sparse_chunk_causal`。
- 没有 DMD。
- 没有 G_real。
- 没有 G_fake。
- 没有 pixel/LPIPS decode loss。

因此 `v7-D4.4` 相对 641 新增了三大块：

1. one-step student 输出 `z_pred` 与 pixel/LPIPS reconstruction loss；
2. DMD score difference：冻结 `G_real`、冻结或 trainable `G_fake` probe，给 student 提供 DMD gradient；
3. fake critic 训练：单独优化 `G_fake`，让 fake critic 学当前 student 生成分布。

当前你看到的 OF-D 色偏/模糊，问题范围已经缩小到第 2/3 块，即 DMD/fake 分支。

## 2. Stage3B one-step reconstruction 不是 DMD，但它是 DMD 的输入来源

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:344-500`

函数：

`Stage3BOneStepReconLoss(...)`

主要逻辑：

1. 随机采样训练 timestep。

   代码：`363-366`

   ```python
   timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
   timestep = pipe.scheduler.timesteps[timestep_id]
   ```

2. 对 GT latent 加噪。

   代码：`368-371`

   ```python
   target_latents = inputs["input_latents"]
   noise = torch.randn_like(target_latents)
   inputs["latents"] = pipe.scheduler.add_noise(target_latents, noise, timestep)
   training_target = pipe.scheduler.training_target(target_latents, noise, timestep)
   ```

3. student DiT 预测 flow / noise。

   代码：`373-376`

   ```python
   noise_pred = pipe.model_fn(...)
   flow_loss = F.mse_loss(noise_pred.float(), training_target.float())
   ```

4. 用 `scheduler.step(..., to_final=True)` 做 one-step latent prediction。

   代码：`378-387`

   ```python
   z_pred = pipe.scheduler.step(noise_pred, timestep, inputs["latents"], to_final=True)
   pipe._stage3_last_z_pred = z_pred
   ```

   这个 `z_pred` 是 DMD 分支后续使用的 `clean_latents`。

5. 随机选 2 个 latent window 做 Wan decoder pixel/LPIPS。

   代码：`389-432`

   当前语义：

   - 随机选 `stage3_recon_num_latents=2`；
   - decode 时使用 full-prefix context；
   - 只对选中 window 对应帧算 MSE/LPIPS；
   - 如果 window 从全局首帧开始，首帧 pixel 和 LPIPS 权重乘 4。

6. student 自身 loss：

   代码：`445-449`

   ```python
   total = flow_weight * flow_loss + mse_weight * mse_loss + lpips_weight * lpips_loss
   ```

重要结论：

- `Stage3BOneStepReconLoss` 内部没有真正训练 `G_fake`。
- `stage3_fake_fm_weight != 0` 在这里会直接报错，因为 fake 必须由双 optimizer runner 管。
- DMD 分支读取的是 `pipe._stage3_last_z_pred`。
- 如果 `z_pred` 自身错，DMD 后面一定错。

风险点：

- `z_pred` 的尺度和真实 clean latent 是否对齐，决定 DMD score 的基础。
- `z_pred` 是 one-step 结果，不是 50-step diffusion 结果。
- pixel/LPIPS 只监督选中 window，DMD 是 latent 全局 loss，两者作用范围不同。

## 3. Stage3BTrainingModule 对 641 forward 的改动

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:503-579`

类：

`FlashVSRStage3BTrainingModule(v6.FlashVSRStage2TrainingModule)`

关键点：

- 它继承 Stage2 训练模块，所以 student 的 LQ projector、Stage2 attention、pipeline unit runner 都沿用 641 的结构。
- forward 里仍然走：

  代码：`557-562`

  ```python
  inputs = self.get_pipeline_inputs(data)
  self.pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)
  merged_inputs = self.transfer_data_to_device(...)
  for unit in self.pipe.units:
      merged_inputs = self.pipe.unit_runner(...)
  merged_inputs[0]["lq_latent_alignment"] = _stage3d31_teacher_lq_alignment_mode(self.pipe)
  ```

- 然后调用 `Stage3BOneStepReconLoss`。

和 641 的区别：

- 641 forward 只返回 flow/FM loss；
- D4.4 forward 额外保存 one-step `z_pred`；
- D4.4 forward 可额外 decode selected window 算 MSE/LPIPS。

## 4. G_real / G_fake 的模型构造

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:2205-2362`

### 4.1 G_real

代码：`2205-2252`

`G_real` 只有在 `--stage3_real_checkpoint` 非空时创建。

当前目的：

- 冻结 teacher / real score network；
- 用 Stage1 checkpoint 表示真实分布 score；
- 不进 optimizer；
- 不反传。

关键参数：

```python
resume_stage1_checkpoint=args.stage3_real_checkpoint
freeze_lq_proj_in=True
stage2_attention_mode=args.stage3_real_attention_mode
lq_proj_temporal_mode=args.stage3_real_lq_proj_temporal_mode
stage3_compute_z_pred=False
stage3_flow_weight=1.0
stage3_mse_weight=0.0
stage3_lpips_weight=0.0
```

当前配置中：

- `stage3_real_attention_mode: dense_full`
- `stage3_real_lq_proj_temporal_mode: nonstreaming_aligned`

这表示我们认为 Stage1 teacher 是非流式 full-attention 语义。

风险点：

- 如果 Stage1 teacher 实际测试时不是这个 projector temporal mode，DMD teacher score 就会错。
- 如果 `nonstreaming_aligned` 对 89->85 的对齐有问题，DMD 可能给颜色/内容错误方向。

### 4.2 G_fake

代码：`2253-2307`

`G_fake` 在 `--stage3_fake_checkpoint` 非空时创建为 trainable model。

当前目的：

- 初始化为 Stage1 checkpoint；
- 用 fake FM loss 训练成 current student fake distribution 的 score / critic；
- 和 student 使用独立 optimizer / independent DeepSpeed engine。

关键参数：

```python
resume_stage1_checkpoint=args.stage3_fake_checkpoint
freeze_lq_proj_in=not bool(args.stage3_fake_train_lq_proj_in)
stage2_attention_mode=args.stage3_fake_attention_mode
lq_proj_temporal_mode=args.stage3_fake_lq_proj_temporal_mode
stage3_compute_z_pred=False
stage3_flow_weight=1.0
stage3_mse_weight=0.0
stage3_lpips_weight=0.0
```

当前配置中：

- `stage3_fake_attention_mode: dense_full`
- `stage3_fake_lq_proj_temporal_mode: nonstreaming_aligned`
- `stage3_fake_train_lq_proj_in: true`
- `stage3_fake_lq_proj_update_every_n_runner_steps: 5`

风险点：

- G_fake 和 G_real 初始相同，但 G_fake 会被 fake FM 更新。
- 如果 fake FM 的训练目标、timestep、noise 或 LQ 条件错，G_fake 会学错。
- 如果 G_fake 学错，DMD gradient 方向会错，OF-D 最容易暴露为色偏/灰屏。

### 4.3 fake_probe_model

代码：`2308-2362`

如果 `stage3_fake_probe_checkpoint` 非空，会创建冻结 fake probe。

如果没有单独 fake probe，但 `fake_model` 是完整 Stage3B 模型：

代码：`2356-2362`

```python
fake_probe_model = fake_model
```

当前 D4.4 常见路径是：trainable `G_fake` 也作为 no-grad DMD fake probe 使用。

风险点：

- DMD student loss 里的 fake probe 使用的是正在被 fake optimizer 更新的同一个 G_fake。
- 这符合“critic 跟随当前 fake distribution”的思路，但也意味着 fake branch 不稳定会直接污染 DMD direction。

## 5. DMD probe：G_real/G_fake 如何预测 x0

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:1086-1179`

函数：

`_stage3c_probe_predict_x0(...)`

这是 DMD 的核心 forward。

输入：

- `probe_model`：G_real 或 G_fake；
- `clean_latents`：student one-step 输出 `z_pred`；
- `dmd_point`：可选，共享 timestep/noisy_latents；
- `data`：同一批 LQ/GT condition。

执行流程：

1. probe model 切 eval。

   代码：`1106-1109`

2. 重建 pipeline inputs。

   代码：`1119-1138`

   如果 dataset 来自 cache，就直接用 inputs；否则重新跑 `get_pipeline_inputs`。

3. 强制设置 LQ latent alignment。

   代码：`1138`

   ```python
   merged["lq_latent_alignment"] = _stage3d31_teacher_lq_alignment_mode(pipe)
   ```

4. 如果没有传入 `dmd_point`，采样同一个 timestep/noise/noisy_latents。

   代码：`1141-1153`

5. 如果传入 `dmd_point`，复用相同 timestep/noisy_latents。

   代码：`1154-1156`

6. 将 `clean_latents` 放到 `input_latents`，将 noisy latent 放到 `latents`。

   代码：`1157-1158`

7. probe DiT 预测 noise/flow。

   代码：`1168`

8. `scheduler.step(..., to_final=True)` 得到 probe 的 `x0_pred`。

   代码：`1171`

9. 返回 detach 后的 `x0_pred`。

   代码：`1174-1177`

关键结论：

- DMD 比较的是 G_real 和 G_fake 对同一个 student `z_pred` 加噪点的 one-step x0 prediction。
- v7-D2 之后，G_real/G_fake 已经共享同一个 `dmd_point`，不是各自采样不同 noise/timestep。

风险点：

- probe 内部 `pipe.units` 和 student 走同样 LQ condition，但 teacher/fake attention mode 是 dense_full。
- `x0_pred` 是否应该用 `to_final=True` 与作者实现一致，需要继续验证。
- `lq_latent_alignment` 是自动由 projector temporal mode 判断的；如果 Stage1 teacher 对齐模式错，DMD 方向会错。

## 6. fake FM loss：G_fake 如何训练

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:1182-1228`

函数：

`_stage3c_fake_fm_loss(...)`

当前逻辑：

1. 如果 `stage3_fake_fm_weight == 0`，直接返回 `None`。

   代码：`1197-1200`

2. G_fake 使用 student 生成的 `clean_latents`，但 detach。

   代码：`1214`

   ```python
   fake_clean_latents = clean_latents.detach()
   ```

3. 对 fake clean latent 随机加噪。

   代码：`1215-1221`

4. G_fake 预测 flow。

   代码：`1225`

5. 用 flow MSE 训练 G_fake。

   代码：`1226-1228`

   ```python
   fake_loss = F.mse_loss(fake_pred.float(), training_target.float())
   fake_loss = fake_loss * pipe.scheduler.training_weight(timestep)
   return fake_loss * fake_weight
   ```

关键结论：

- fake loss 不回传 student。
- fake loss 训练目标是“当前 student fake distribution 上的 FM target”。
- D4.4 中 G_fake 每个 runner step 都更新，student 每 5 个 runner step 更新一次。

风险点：

- fake FM 用自己采样的 timestep/noise，不一定和 DMD student loss 的 `dmd_point` 是同一个点。
- 这在 DMD2 类训练里可以成立，但需要确认作者是否这样实现。
- 如果 fake critic 没训好或过度漂移，DMD-only 会直接崩。

## 7. DMD student loss：真正给 student 的 DMD 梯度

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:1260-1334`

函数：

`_maybe_run_stage3c_dmd_student_loss(...)`

当前逻辑：

1. `stage3_dmd_weight == 0` 时关闭。

   代码：`1280-1282`

2. 只在满足 `global_step % stage3_dmd_probe_every == 0` 时运行。

   代码：`1283-1285`

3. no-grad 跑 G_real，返回 `real_x0` 和共享 `dmd_point`。

   代码：`1286-1294`

4. no-grad 跑 G_fake，复用同一个 `dmd_point`。

   代码：`1295-1302`

5. 构造 DMD score 差。

   代码：`1303-1308`

   ```python
   z_detached = clean_latents.detach()
   p_real = z_detached.float() - real_x0.float()
   p_fake = z_detached.float() - fake_x0.float()
   weight_factor = p_real.abs().mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
   dmd_grad = torch.nan_to_num((p_real - p_fake) / weight_factor)
   ```

6. DMD spike guard。

   代码：`1309-1331`

   - `stage3_dmd_grad_norm_max`
   - `stage3_dmd_spike_policy`
   - `stage3_dmd_loss_max`

7. 用 synthetic target 把 DMD gradient 注入 student。

   代码：`1332-1334`

   ```python
   target = (clean_latents.float() - dmd_grad).detach()
   dmd_loss = 0.5 * F.mse_loss(clean_latents.float(), target, reduction="mean")
   return dmd_loss * dmd_weight
   ```

重要结论：

- DMD loss 数值本身不是普通“预测错多少”，而是把指定的 `dmd_grad` 伪装成 MSE target 后回传。
- `dmd_grad` 的方向和尺度比 `dmd_loss` 数值更重要。
- 当前 DMD 归一化除以的是 `p_real.abs().mean(...)`，不是统一 token 数或 pixel 数。

最关键风险点：

1. 公式方向可能错。

   如果 `(p_real - p_fake)` 的符号和作者/DMD2 实现相反，DMD-only 会稳定把 student 推坏。

2. 归一化可能错。

   `weight_factor = mean(abs(p_real))` 会让 real_x0 很接近 z 时 DMD grad 被放大。

3. fake_x0 如果不可信，DMD grad 就是错的。

   OF-D 色偏/灰屏最符合这个风险。

4. DMD guard 只能限制 spike，不能修正错误方向。

   如果方向错，clamp 只会让模型慢慢错，不会让它对。

## 8. 双 optimizer runner

代码位置：

`train_flashvsr_stage3_v7_d4_4_lora.py:1337-1815`

函数：

`launch_stage3c_dual_optimizer_task(...)`

### 8.1 两个 optimizer

代码：`1382-1388`

```python
optimizer = AdamW(model.trainable_modules(), ...)
fake_optimizer = AdamW(fake_trainable_params, ...)
```

student 和 G_fake 是两个 optimizer。

### 8.2 两个 DeepSpeed engine

代码：`1427-1438`

```python
_stage3d44_select_deepspeed_plugin(accelerator, "student")
model, optimizer, ... = accelerator.prepare(...)

_stage3d44_select_deepspeed_plugin(fake_accelerator, "fake")
fake_model, fake_optimizer, fake_scheduler = fake_accelerator.prepare(...)
```

student 和 fake 分别由不同 accelerator 管。

### 8.3 dfake 调度

代码：`1518-1520`

```python
current_runner_step = runner_step
generator_turn = _stage3d4_is_generator_turn(args, current_runner_step)
fake_lq_proj_turn = _stage3d44_should_update_fake_lq_proj(args, current_runner_step)
```

当前 D4.4 配置：

- `stage3_dfake_gen_update_ratio=5`
- runner step 每步都会跑 fake update；
- 只有 `runner_step % 5 == 0` 时 student/generator 更新。

这意味着：

- fake update 频率是 5 倍于 student；
- `global_step` 是 student step；
- `runner_step` 是 fake/student runner step。

### 8.4 student forward

代码：`1534-1545`

- generator turn：student forward 有 grad；
- fake-only turn：student forward 在 `torch.no_grad()` 里，仅用于生成 fake sample。

### 8.5 DMD student loss 只在 generator turn 上加

代码：`1568-1576`

```python
dmd_student_loss = ... if generator_turn else None
```

### 8.6 fake loss 每个 runner step 都算

代码：`1596-1610`

```python
fake_loss = _stage3c_fake_fm_loss(...)
```

### 8.7 fake backward 和 student backward 分开

代码：`1615-1632`

```python
fake_accelerator.backward(fake_loss)
...
student_total_loss = student_loss + dmd_student_loss
accelerator.backward(student_total_loss)
```

### 8.8 optimizer step

代码：`1641-1644`

```python
if generator_turn:
    optimizer.step()
if fake_did_update:
    fake_optimizer.step()
```

重要结论：

- D4.4 不是“纯 generator turn 不更新 fake”的实现。
- 它是“fake 每个 runner step 更新，student 每 5 个 runner step 更新”。
- 这更接近 dfake=5 的频率语义。

风险点：

- fake-only turn 虽然 student 不反传，但仍要先跑 student no-grad 生成 fake sample。
- 如果 no-grad student forward 和 generator-turn student forward 输入不一致，fake 会学错分布。
- fake 每步更新会让 critic 比 student 更新快很多，但如果 fake target 本身错，它也会更快学错。

## 9. D4.4 当前配置的 DMD 相关参数

配置文件：

`stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`

关键参数：

```yaml
stage3_flow_weight: 1.0
stage3_mse_weight: 1.0
stage3_lpips_weight: 2.0

stage3_fake_attention_mode: "dense_full"
stage3_fake_lq_proj_temporal_mode: "nonstreaming_aligned"
stage3_fake_train_lq_proj_in: true
stage3_fake_lq_proj_update_every_n_runner_steps: 5
stage3_fake_fm_weight: 1.0
stage3_dfake_gen_update_ratio: 5
stage3c_fake_learning_rate: 1e-5

stage3_real_attention_mode: "dense_full"
stage3_real_lq_proj_temporal_mode: "nonstreaming_aligned"

stage3_dmd_probe_every: 1
stage3_dmd_weight: 1.0
stage3_dmd_grad_norm_max: 5.0
stage3_dmd_loss_max: 3.0
stage3_dmd_spike_policy: "skip"
```

解释：

- `flow/mse/lpips` 是 student 自身 loss；
- `fake_fm_weight=1.0` 训练 G_fake；
- `dmd_weight=1.0` 给 student 加 DMD gradient；
- `dfake_gen_update_ratio=5` 表示 fake 更新 5 次，student 更新 1 次；
- `dmd_grad_norm_max=5.0` + `skip` 只跳过极端 DMD grad；
- `dmd_loss_max=3.0` 会按 loss magnitude 缩放 DMD grad。

## 10. 昨天 OF-A/B/C/D 和新增 OF-E 的意义

固定数据：

`/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`

昨天固定 LQ/GT 的结果：

| 编号 | Loss | 目的 | 观察 |
| --- | --- | --- | --- |
| OF-A | flow + MSE + LPIPS + DMD + fake | 完整 D4.4 小样本 | 相对较好，但 DMD 可能被其他 loss 锚住 |
| OF-B | MSE + LPIPS | pixel reconstruction 是否正常 | 用于判断 decode/recon 对齐 |
| OF-C | flow only | one-step flow 是否会拉糊 | 用于判断 flow 本身 |
| OF-D | DMD + fake only | DMD/fake 是否独立成立 | 最后严重色偏和模糊 |
| OF-E | flow + MSE + LPIPS，去掉 DMD/fake | 验证“只去 DMD”后 full loss 是否稳定 | 本轮新增 |

OF-D 的结果说明：

- DMD/fake-only 当前不能作为独立稳定目标；
- 这不是 pixel/LPIPS 造成的，因为 OF-D 没有 pixel/LPIPS；
- 也不是固定 LQ/GT 数据差异，因为四组现在已经共用同一批 `.pt`；
- 下一步应聚焦 DMD/fake 分支，而不是继续泛泛讨论 Stage3 全部 loss。

OF-E 的意义：

- 如果 OF-E 稳定，说明 flow+pixel 本身没有明显问题；
- 如果 OF-E 也模糊，则说明不是 DMD 独有，可能是 one-step flow 或 decoder/pixel 组合；
- 如果 OF-E 好但 OF-A 变差，DMD/fake 是负贡献；
- 如果 OF-A 好于 OF-E，但 OF-D 崩，说明 DMD 只有在 pixel/flow 锚定下有弱正贡献，不能单独依赖。

## 11. 当前最可疑的问题列表

### 11.1 DMD 方向符号

当前：

```python
dmd_grad = ((z - real_x0) - (z - fake_x0)) / norm
```

等价：

```python
dmd_grad = (fake_x0 - real_x0) / norm
```

如果作者/DMD2 需要相反方向，OF-D 会非常容易崩。

验证方法：

- 做 DMD sign flip probe，不长训；
- 固定 batch 单步更新，比较 real/fake score objective 是否朝预期改善；
- 不先改正式代码，只写 DMD debug 专用脚本。

### 11.2 DMD normalization

当前除以：

```python
p_real.abs().mean(...)
```

风险：

- 如果 `p_real` 很小，梯度放大；
- 如果 fake_x0 偏色，DMD 会把偏色放大；
- loss 数值看起来不大，但梯度方向仍可能有害。

验证方法：

- 记录 `p_real.mean/std/absmean`、`p_fake.mean/std/absmean`、`real_x0/fake_x0/z_pred` 范围；
- 对比不归一化、除以 latent std、除以 scheduler sigma 的方向和尺度。

### 11.3 G_fake target 与 DMD probe point 不一致

fake FM loss 自己采样 timestep/noise。

DMD student loss 另采样一个 shared `dmd_point`。

这并不必然错，但如果 fake critic 训练不足，DMD probe 在未学好的点上会给错 direction。

验证方法：

- 固定 fake FM timestep/noise 与 DMD point；
- 或记录同一 dmd_point 上 fake loss 是否下降。

### 11.4 G_real/G_fake projector temporal mode

当前：

- student: Stage2 streaming/block sparse；
- G_real/G_fake: dense_full + nonstreaming_aligned；
- DMD 的 `lq_latent_alignment` 自动按 projector temporal mode 设置。

如果 Stage1 teacher 的正确 LQ projector 行为不是 `nonstreaming_aligned`，DMD 会错位。

验证方法：

- 对同一 LQ/GT，用 G_real dense_full/nonstreaming_aligned 输出 x0 preview；
- 和 Stage1 官方测试路径对比；
- 检查 89->85 latent/frame 对齐。

### 11.5 fake lq_proj 训练频率

当前 fake LoRA 每个 runner step 更新，fake lq_proj 每 5 runner step 更新。

风险：

- fake critic 的 LoRA 和 projector 学习速率不同；
- DMD probe 用的 fake_model 处于不断变化状态；
- 色偏可能来自 fake lq_proj 漂移。

验证方法：

- OF-D with fake lq_proj frozen；
- OF-D with fake lq_proj 每步更新；
- 比较颜色漂移是否显著变化。

## 12. 下一步实验顺序

### 实验 E：OF-E flow+MSE+LPIPS no-DMD

目的：

验证“只去掉 DMD/fake 后，flow+pixel 是否能稳定 overfit”。

设置：

- 固定同一批 LQ/GT `.pt`；
- `flow=1, mse=1, lpips=2, dmd=0, fake_fm=0`；
- 4 GPU；
- 保存 `1,2,5,10,20,50,100,150,200,220`；
- validation 从训练 batch 来。

### DMD-1：DMD tensor dump

目的：

不训练，固定 batch 输出：

- `z_pred`
- `real_x0`
- `fake_x0`
- `p_real`
- `p_fake`
- `dmd_grad`
- 对应 mean/std/min/max/absmean
- x0 decode preview

验收：

- real/fake x0 不应有明显色偏；
- dmd_grad 不应在颜色通道上有系统偏置；
- 如果 fake_x0 已经偏绿，先修 fake branch。

### DMD-2：sign / norm 单步更新 probe

目的：

固定 batch 上比较：

- 当前符号；
- 反符号；
- 当前 normalization；
- 无 normalization；
- sigma/std normalization。

验收：

- 哪个方向让 teacher-defined objective 改善；
- 哪个方向让 pixel preview 不崩。

### DMD-3：fake critic sanity

目的：

冻结 student，单训 fake critic，看 G_fake 是否能在固定 fake samples 上稳定下降，不产生颜色漂移。

验收：

- fake_loss 降；
- fake_x0 不绿；
- fake output norm 不爆。

## 13. 当前结论

现在不能继续只说“Stage3 有问题”。根据 OF-D，问题已经明确缩小到：

1. DMD gradient 方向/尺度；
2. G_fake critic 训练目标；
3. G_real/G_fake projector / attention wrapper 语义；
4. DMD 归一化和 spike guard。

OF-E 会帮助判断：没有 DMD 的情况下，flow+pixel 是否足够稳定。如果 OF-E 正常，后续就应集中修 DMD/fake，而不是再怀疑 pixel/LPIPS 主链路。
