# FlashVSR Stage3 v7-D4.4 重新审阅记录（2026-05-20）

## 1. 本次审阅目标

这次审阅刻意不沿用之前关于 ghost、残影、7D/7D1/7D4 的假设，而是重新从三类依据出发：

- 论文逻辑：FlashVSR、DMD2、OSEDiff。
- 参考代码：本地 `mac_code/DMD2`、`mac_code/OSEDiff`，以及联网确认的官方 `tianweiy/DMD2`。
- 当前代码：`train_flashvsr_stage3_v7_d4_4_lora.py`、对应 40GPU YAML 和启动 SH。

当前被审阅版本：

- YAML：`wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- SH：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-40GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- PY：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- 远端实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`

## 2. 外部依据

### FlashVSR 论文给出的 Stage3 目标

FlashVSR Stage3 的目标是把 Stage2 sparse-causal DiT 蒸馏为 one-step streaming model。论文中 Stage3 损失是：

`L = LDMD(zpred, Gone, Greal, Gfake) + LFM(zpred, Gfake) + MSE(xpred, xgt) + lambda * LPIPS(xpred, xgt)`

关键点：

- `Gone` 是 one-step student。
- `Greal` 是真实分布 score probe。
- `Gfake` 是 fake latent distribution 的 score probe，需要跟随 student 生成分布。
- pixel / LPIPS 需要 decode 到像素空间后计算。
- `lambda = 2`。
- 每轮随机选 2 个 latent decode，previous latents detached，用于控制显存。
- Stage3 仍然要求 train / inference 语义对齐：训练时使用 block-sparse causal attention mask，并让 one-step student 在统一 timestep 下学习 streaming 语义。

### DMD2 论文和代码给出的双支路逻辑

联网确认的官方仓库：`https://github.com/tianweiy/DMD2`。本地镜像路径：`/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2`。

DMD2 的重要逻辑：

- DMD2 明确把 generator 和 guidance / fake score model 分成两类模型。
- fake score model 需要用 two-time-scale rule 更频繁更新，论文推荐过 `5 fake score updates per generator update`。
- 官方训练代码中，generator turn 和 guidance turn 是两个明确阶段：
  - 先 generator forward，可选 generator backward/update。
  - 再用 generator forward 产生的 fake sample 训练 guidance/fake model。
- DMD 梯度公式在代码中是：
  - `p_real = latents - pred_real`
  - `p_fake = latents - pred_fake`
  - `grad = (p_real - p_fake) / mean(abs(p_real))`
  - `loss = 0.5 * mse(latents, (latents - grad).detach())`
- fake guidance loss 明确 `latents = latents.detach()`，不允许回传到 generator。
- 官方代码对 generator 和 guidance 都做 `clip_grad_norm_`。

### OSEDiff 给出的 pixel / LPIPS + latent regularization 参考

本地路径：`/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/OSEDiff`。

OSEDiff 的相关参考点：

- pixel reconstruction 是 `MSE + lambda_lpips * LPIPS`。
- regularization / distribution loss 在 latent space 做。
- generator loss 和 regularizer loss 有独立 optimizer step。
- OSEDiff 也对两支路都做 gradient clipping。

### Accelerate / DeepSpeed 多模型官方建议

HuggingFace Accelerate 官方文档 `Using multiple models with DeepSpeed` 明确说：

- 多模型 DeepSpeed API 仍然是 experimental。
- 如果是训练多个 disjoint models，需要两个 DeepSpeedPlugin，并且通常需要第二个 `Accelerator`。
- 需要通过 `select_deepspeed_plugin()` 切换当前 DeepSpeed plugin 后再 `prepare()` 对应模型。

参考链接：`https://huggingface.co/docs/accelerate/v1.9.0/en/usage_guides/deepspeed_multiple_model`

这说明 v7-D4.4 使用 dual accelerator + two DeepSpeed plugins 的方向是合理的，不是明显错路。

## 3. v7-D4.4 当前实现摘要

### 配置

当前 40GPU YAML 的关键设置：

- student：`stage2_attention_mode=block_sparse_chunk_causal`
- `stage3_flow_weight=1.0`
- `stage3_mse_weight=1.0`
- `stage3_lpips_weight=2.0`
- `stage3_first_frame_pixel_weight=4.0`
- `stage3_first_frame_lpips_weight=4.0`
- `stage3_dmd_weight=1.0`
- `stage3_fake_fm_weight=1.0`
- `stage3_dfake_gen_update_ratio=5`
- `stage3_dmd_grad_norm_max=5.0`
- `stage3_dmd_loss_max=3.0`
- `stage3_dmd_spike_policy=skip`
- `stage3_fake_lq_proj_update_every_n_runner_steps=5`
- `Greal/Gfake attention_mode=dense_full`
- `Greal/Gfake lq_proj_temporal_mode=nonstreaming_aligned`

### Student loss

`FlashVSRStage3BTrainingModule` 继承 Stage2 训练模块。student forward 内部：

- 使用 Stage2 block-sparse causal attention。
- 计算 flow loss。
- 计算 one-step `z_pred`。
- decode 随机 2 个 latent window。
- 对选中的像素窗口计算 MSE 和 LPIPS。
- 如果 window 命中首帧，则 pixel 和 LPIPS 都乘 4。

### DMD student loss

`_maybe_run_stage3c_dmd_student_loss()` 中：

- `Greal` 和 `Gfake` 都在 `torch.no_grad()` 下预测 `x0`。
- `Greal` 和 `Gfake` 使用同一个 noisy latent / timestep。
- DMD 梯度公式与 DMD2 代码一致：
  - `p_real = z - real_x0`
  - `p_fake = z - fake_x0`
  - `grad = (p_real - p_fake) / mean(abs(p_real))`
  - `loss = 0.5 * mse(z, (z - grad).detach())`
- 额外加了 spike guard / skip / clamp，这是工程稳定性增强，不是论文原始公式。

### Fake FM loss

`_stage3c_fake_fm_loss()` 中：

- 输入是当前 student `z_pred`。
- `z_pred` 会 detach。
- 对 fake latent 加随机噪声和 timestep。
- G_fake 预测 flow target。
- 只更新 G_fake。

这点与 DMD2 “fake score model 要拟合当前 generator fake distribution”基本一致。

### 双 DeepSpeed / 双 optimizer

`launch_stage3c_dual_optimizer_task()` 中：

- student 由 `accelerator.prepare(model, optimizer, scheduler)` 管。
- fake 由 `fake_accelerator.prepare(fake_model, fake_optimizer, fake_scheduler)` 管。
- generator turn 每 5 个 runner step 一次。
- fake 每个 runner step 都更新。
- generator turn 时，fake backward 先执行，student backward 后执行，最后两个 optimizer 分别 step。

这与 DMD2 官方代码“两个 optimizer + fake 更频繁更新”方向一致，但更新顺序和工程组织不是完全一样。

## 4. 当前我认为“基本正确”的部分

### DMD 梯度公式基本对齐 DMD2

v7-D4.4 的 DMD student loss 公式和 DMD2 `edm_guidance.py` 中的公式是同构的。符号方向没有看到明显反号问题。

### fake loss 对 student detach 是正确的

`_stage3c_fake_fm_loss()` 明确 `clean_latents.detach()`，fake loss 不应该回传 student。这个符合 DMD2 fake score update。

### dual accelerator 方向合理

HuggingFace 官方多 DeepSpeed 模型文档也建议 disjoint model training 使用第二个 Accelerator。因此 v7-D4.4 的双 accelerator 不是明显错误。

### Stage3 loss 权重基本对齐论文

当前 `MSE=1, LPIPS=2, DMD=1, fake_FM=1`，比之前 debug 权重更接近论文语义。首帧 pixel 和 LPIPS 都乘 4，也符合我们对 Wan 首 latent 特殊语义的处理。

## 5. 需要怀疑的问题

### 问题 1：没有 optimizer-level grad clipping

DMD2 官方代码对 generator 和 guidance 都调用 `clip_grad_norm_`。OSEDiff 也对 generator 和 regularizer 分支做 grad clipping。

v7-D4.4 当前只对 DMD 梯度本身做 spike skip / clamp，但没有看到 student optimizer 和 fake optimizer 的统一 `clip_grad_norm_`。这不是同一件事：

- DMD spike guard 只管 DMD 梯度项。
- optimizer grad clipping 管所有 loss 汇总后的参数梯度，包括 flow、pixel、LPIPS、fake FM。

风险：

- fake score branch 可能出现大梯度但不被限制。
- LPIPS / pixel decoder branch 也可能造成局部大梯度。
- DeepSpeed ZeRO2 下不加 clipping，长训稳定性弱于 DMD2/OSEDiff 参考实现。

建议验证：

- 打印 student/fake 两支路 grad norm。
- 开一个短实验只加 grad clipping，不改其它设定，对比 loss spike、fake_loss、val 视频。

### 问题 2：`stage3_fake_lq_proj_update_every_n_runner_steps=5` 是自定义策略，不是 DMD2 标准策略

DMD2 的核心是 fake score model 比 generator 更新更频繁。当前 v7-D4.4 的 G_fake 主体每个 runner step 更新，但 fake 的 LQ projector 只每 5 个 runner step 更新一次。

这意味着：

- G_fake 的 DiT LoRA 每步适应 fake latent distribution。
- G_fake 的 conditioning projector 只有 generator turn 那一步才适应。

这可能是为了稳定，但也可能让 fake score model 的条件分布跟不上 LR conditioning，尤其 VSR 强依赖 LQ projector。

风险：

- G_fake score 对 fake latent 的估计不准，DMD gradient 偏。
- fake critic “看错条件”，DMD 对 student 的方向可能不稳定。

建议验证：

- `lqprojfreq5` vs `lqprojfreq1` 做短训对比。
- 记录 fake_lq_proj 的 grad norm 和参数 delta。
- 对同一个 batch 比较 `fake_x0` 与 `real_x0` 的差值随训练是否更稳定。

### 问题 3：generator / fake 更新顺序与 DMD2 不完全一样

DMD2 官方训练是：

1. generator forward。
2. 如果到 generator turn，先 generator backward/update。
3. guidance/fake turn 使用 generator forward 留下的 fake sample 更新 fake model。

v7-D4.4 当前是：

1. student forward 得到 `z_pred`。
2. fake loss backward。
3. generator turn 时 student backward。
4. 两个 optimizer step。

如果两个图完全 detach 且 optimizer 分离，这在梯度归属上应接近等价。但它不是 DMD2 官方的完全同序实现。

风险：

- DeepSpeed engine 内部状态、梯度同步、accumulate 上下文可能导致边界不够清晰。
- fake backward 在 student backward 之前，调试时更难判断两个 engine 是否互相污染。

建议验证：

- 做一次单 step 梯度归属检查：
  - fake backward 后 student trainable params grad 必须为 0/None。
  - student backward 后 fake trainable params grad 不应新增。
  - fake optimizer step 前后只有 fake 参数变化。
  - student optimizer step 前后只有 student 参数变化。

### 问题 4：Greal/Gfake 用 dense_full + nonstreaming_aligned 是设计选择，需要确认不是错误

当前 student 是 Stage2 sparse-causal streaming 结构，Greal/Gfake 是 dense_full + nonstreaming_aligned。

这可能合理：

- DMD2 的 fake/real score model 通常是 score model，不一定要和 generator 完全同结构。
- Greal 用更强的 dense score 可能更稳定。

但也有风险：

- fake score 需要拟合 student fake distribution，如果条件投影/attention 语义与 student 差太大，DMD gradient 可能带来结构偏差。
- FlashVSR Stage3 论文强调 block-sparse causal training / inference 对齐，Gfake 是否也应该同样 streaming，并没有从当前代码里得到直接证明。

建议验证：

- 只换 Gfake attention：`dense_full` vs `block_sparse_chunk_causal`，短训看 DMD grad、fake_loss、val。
- Greal 保持 dense，Gfake 换 streaming；如果改善稳定性，说明 fake score 应更贴近 student 分布。

### 问题 5：fake-only turn 仍完整跑 student forward + reconstruction，效率上不理想

fake-only turn 用 `torch.no_grad()` 跑 student forward，只需要 `z_pred` 给 G_fake。当前 student forward 仍会走 Stage3B loss，包括随机 decode / MSE / LPIPS 的路径。

这不一定影响正确性，但浪费显存和时间。

建议：

- 增加 fake-only `z_pred_only` 模式：不算 pixel/LPIPS、不做 VAE decode，只保留 one-step student output。
- 先作为性能优化，不改变 loss 语义。

### 问题 6：validation_num_samples=1 偏少

当前 40GPU YAML 里 `validation_num_samples=1`。这不是训练逻辑错误，但不足以监控视频稳定性。

建议：

- 稳定后改成 2 或 3 个固定样本。
- val 不要影响训练主循环太久，必要时用外部测试脚本定期扫 ckpt。

### 问题 7：scheduler step 语义与 DMD2 不完全一致，但当前 ConstantLR 影响小

DMD2 官方 generator scheduler 每 runner step 都 step。v7-D4.4 的 student scheduler 只在 generator turn step。当前 scheduler 是 ConstantLR，所以几乎无影响。

风险：

- 如果以后改成非 constant LR，这里会变成实际差异。

建议：

- 文档里明确：当前 ConstantLR 下无实质影响。
- 如果换 scheduler，需要重新定义 runner_step 和 generator_step 的 LR 轴。

## 6. 双 Accelerate / DeepSpeed 优化建议

我认为当前 dual accelerator 方向是对的，但还不够“工程干净”。

优先建议：

1. 加 student/fake 两支路 grad norm 统计和 clipping。
2. 把 fake-only student forward 改为 z_pred-only，减少没必要的 decoder / LPIPS 计算。
3. 验证 `stage3_fake_lq_proj_update_every_n_runner_steps=1` 是否比 5 更合理。
4. 明确 `with accelerator.accumulate(model)` 只适用于 `gradient_accumulation_steps=1` 的当前设置；如果未来要 accumulation，需要重新包 fake_accelerator 的 accumulate。
5. 把 fake/student 参数更新归属做成一个固定 smoke test，不要只靠肉眼看 loss。

## 7. 最小验证计划

### V1 梯度归属验证

目的：确认 fake loss 不回传 student，student loss 不回传 fake。

步骤：

- 用 2GPU 或单机单卡跑一个固定 batch。
- fake backward 后检查：
  - student LoRA grad norm。
  - student LQ projector grad norm。
  - fake LoRA grad norm。
  - fake LQ projector grad norm。
- student backward 后再检查同样四组。
- optimizer step 后比较参数 delta。

通过标准：

- fake backward 后 student grad 为 0/None。
- student backward 后 fake grad 不新增。
- fake step 只改变 fake 参数。
- student step 只改变 student 参数。

### V2 fake LQ projector 更新频率验证

目的：判断 `lqprojfreq5` 是否削弱 G_fake。

对比：

- A：当前 `stage3_fake_lq_proj_update_every_n_runner_steps=5`。
- B：改为 1。

记录：

- fake_loss。
- dmd_grad。
- dmd_skip。
- fake_lq_proj grad norm。
- 100/200 step val。

### V3 Gfake attention 语义验证

目的：判断 fake score model 是否应该更贴近 student streaming 分布。

对比：

- A：Greal dense，Gfake dense。
- B：Greal dense，Gfake block_sparse_chunk_causal。

记录：

- DMD gradient magnitude。
- fake loss convergence。
- same ckpt inference 是否更稳定。

### V4 grad clipping 验证

目的：确认不改变目标函数的情况下提升稳定性。

对比：

- A：当前无 optimizer clipping。
- B：student/fake 都加 `clip_grad_norm_`，阈值先用 DMD2 默认附近的保守值。

记录：

- loss spike 数量。
- fake grad norm。
- student grad norm。
- dmd_skip 次数。

### V5 fake-only z_pred-only 性能验证

目的：只优化性能，不改变目标。

对比：

- A：当前 fake-only 完整 student forward。
- B：fake-only 只输出 `z_pred`，不算 recon loss。

记录：

- runner step time。
- 显存峰值。
- fake_loss 是否一致。

## 8. 当前结论

v7-D4.4 不是“明显错误”的代码。它的主干逻辑已经接近 DMD2 + FlashVSR Stage3：

- one-step student。
- DMD gradient 公式方向正确。
- fake model 用 detach fake sample 训练。
- fake updates 比 generator 更频繁。
- pixel / LPIPS 权重和首帧处理合理。
- dual accelerator + DeepSpeed 方向符合 Accelerate 官方多模型训练建议。

但它仍然有几个严肃的不确定点：

- 缺 optimizer-level grad clipping。
- fake LQ projector 只 5 步更新一次是自定义策略，可能削弱 G_fake。
- fake/student 两路虽然看起来分离，但需要做参数归属验证，而不是只看 loss。
- Greal/Gfake dense_full 与 student streaming 的结构差异需要验证。
- fake-only turn 计算过重，影响效率。

我建议下一步不要直接大规模改目标函数，而是先做 V1/V2/V4 三个最小验证。最优先是 V1，因为它能直接判断双 accelerator / 双 optimizer 是否真的没有串梯度。

## 9. 2026-05-20 V1 16 卡参数归属验证

验证对象：

- 代码：`/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py`
- 启动脚本：`/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh`
- 配置基线：`/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- 机器：`bfs6vaz4d6` + `i6hf4scd4y`，共 16 卡。
- 成功 run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradcheck_16gpu_v7_d4_4_Ownership_20260520_v1b_ownership_16gpu_clean`

第一次启动失败原因：

- `i6hf4scd4y` 上旧的 `occupy_all/gpu_stress` 没被 `tmux kill-session -t lxh` 杀掉。
- 每卡已有约 162GB 占卡显存，导致 DeepSpeed ZeRO2 初始化 flatten optimizer buffer 时 OOM。
- 清掉 `gpu_stress` 的真实 GPU PID 后，16 卡干净重跑成功。

关键观察：

- 首帧强制验证生效：`latent_window=[0,2)`、`frame_window=[0,5)`。
- reconstruction 分支生效：`recon_latents=2`、`decoded_frames=5`、`context_mode=full_prefix`、`decoder_cpu_offload=True`。
- 首帧权重生效：`first_frame_pixel_weight=4.0`、`first_frame_lpips_weight=4.0`。
- `G_real` 没有参数变化：所有记录中 `real_delta=0.000000e+00`。

参数归属结果：

| runner_step | turn | generator_update | student_delta | fake_delta | real_delta | 结论 |
|---:|---|---:|---:|---:|---:|---|
| 0 | generator | 1 | `3.757248e+00` | `1.220477e+01` | `0` | generator turn 同时更新了 student 和 fake。 |
| 1 | fake | 0 | `0` | `1.064757e+02` | `0` | fake-only turn 没有更新 student。 |
| 2 | fake | 0 | `0` | `-2.222283e+02` | `0` | fake-only turn 没有更新 student。 |
| 3 | fake | 0 | `0` | `-1.531622e+02` | `0` | fake-only turn 没有更新 student。 |
| 4 | fake | 0 | `0` | `-2.968045e+01` | `0` | fake-only turn 没有更新 student。 |
| 5 | generator | 1 | `9.786371e+01` | `-1.010206e+00` | `0` | generator turn 仍有小幅 fake 更新。 |

当前判断：

- fake-only turn 的 detach/optimizer 隔离看起来成立：student 不被 fake-only loss 更新。
- `G_real` 冻结成立。
- generator turn 同时出现 `fake_delta`，经 2026-05-20 复核源码后确认不是意外串梯度，而是 D4.4 当前设计：每个 runner step 都会执行 fake loss/backward/optimizer step，只有 student/generator 每 `stage3_dfake_gen_update_ratio=5` 个 runner step 更新一次。
- 因此这里的 `generator_turn` 命名不够准确，更应该理解为 `student_update_turn`：这一轮 student 也更新，而不是这一轮 fake 不更新。
- 当前配置里 `stage3_fake_lq_proj_update_every_n_runner_steps=5`，所以 fake 侧 `lq_proj_in` 也正好在 runner 0/5/10 这类 student update turn 更新；这会让这些 turn 的 `fake_delta` 更明显，属于配置预期。

因此，V1 的修正结论是：

- 没有发现 fake-only turn 反传污染 student。
- 没有发现 `G_real` 被更新。
- student update turn 里出现 fake 参数变化符合当前 D4.4 的 dfake=5 频率设计，不再作为“隔离不彻底”的证据。
- 仍建议后续把日志字段从 `generator_turn/generator_update` 改得更清楚，至少在文档里统一称为 `student_update_turn`，避免继续误判。

## 10. Gemini 审阅意见复核（2026-05-20）

外部审阅文件：

- `/Users/lixiaohui/Desktop/gemini_FlashVSR_Stage3_Review.md`

### 10.1 LPIPS / DMD / Flow 梯度量级失衡

Gemini 观点：

- `flow_loss` / `dmd_loss` 在 latent 全维度做 mean。
- `lpips_loss` 按 frame 做感知距离。
- 因此 LPIPS 对 `z_pred` 的实际梯度可能远大于 DMD / flow，导致 DMD 蒸馏被掩盖。

复核结论：

- 这个怀疑有道理，但不能只从 loss 标量或 mean 维度直接下结论。
- 当前日志里的 `dmd_grad` 是 DMD 构造梯度的 mean-abs，不等于 student 参数最终接收到的总梯度占比。
- 真正需要测的是同一个 batch 上不同 loss 分量对 `z_pred` 和 student trainable 参数的梯度范数。

建议新增验证：

- 固定同一个 batch 和同一个 timestep/noise。
- 分别 backward：
  - flow only；
  - pixel MSE only；
  - LPIPS only；
  - DMD only；
  - full loss。
- 记录：
  - `z_pred.grad.abs().mean()` / `z_pred.grad.norm()`；
  - LoRA grad norm；
  - LQ projector grad norm；
  - 各 loss 的未加权/加权标量。

判据：

- 如果 LPIPS 的 `z_pred` 梯度比 DMD 大 10 倍以上，`stage3_lpips_weight=2.0` 需要重新讨论。
- 如果 LPIPS 和 DMD 在同一量级，则保持论文权重，不要过早调小 LPIPS。

### 10.2 G_fake 梯度累加可能失效

Gemini 观点：

- 当前代码只有 `with accelerator.accumulate(model)`，没有 `with fake_accelerator.accumulate(fake_model)`。
- 如果 `gradient_accumulation_steps > 1`，fake optimizer 每个 micro batch 都 zero_grad/step，会破坏 fake 梯度累加。

复核结论：

- 当前 D4.4 40GPU YAML 中 `gradient_accumulation_steps: 1`，所以这个问题对当前实验没有影响。
- 但如果后续为了显存或 batch size 把 accumulation 改成大于 1，这条会变成真实 bug。

建议：

- 当前不需要改。
- 在文档里标记约束：D4.4 当前 runner 只支持 `gradient_accumulation_steps=1`。
- 如果未来要开 accumulation，需要把 fake 分支也放进 `fake_accelerator.accumulate(fake_model)`，并重写 zero_grad/step 条件。

### 10.3 LPIPS 输入值域是否错误

Gemini 观点：

- LPIPS 通常期望 `[-1,1]`。
- debug dump 里把 `video` clamp 到 `[0,1]`，可能说明 LPIPS 输入也是 `[0,1]`。

复核结论：

- 这条在当前代码里基本是误判。
- `x_pred` 来自 Wan VAE decode：`WanVideoVAE.single_decode()` 返回 `video.clamp_(-1, 1)`。
- `x_gt` 来自 `pipe.preprocess_video(inputs["input_video"])`，默认 `min_value=-1, max_value=1`，会把 dataloader 的 `[0,1]` video 映射到 `[-1,1]`。
- debug dump 的 `clamp(0,1)` 只用于保存 dataloader 原始视频可视化，不代表训练里的 LPIPS 输入值域。

结论：

- 当前 LPIPS 输入值域和标准 LPIPS 预期一致。
- 不需要把 `x_pred * 2 - 1` 之类转换加进训练；那反而会把 `[-1,1]` 错映射到 `[-3,1]`。

### 10.4 三模型共存的 ZeRO-2 显存压力

Gemini 观点：

- Student / G_fake / G_real 三个模型同时驻留，ZeRO-2 不切参数，会造成显存压力。

复核结论：

- 这是已知工程事实，不是新 bug。
- 当前 40 卡 D4.4 能跑起来，说明显存层面暂时可接受。
- 但它解释了为什么 Stage3 比 Stage1/2 慢很多、显存峰值高很多。

后续优化方向：

- 不改变训练目标前提下，优先做：
  - fake-only turn 只算 `z_pred`，跳过 pixel/LPIPS；
  - 检查 frozen probe 是否能更彻底串行释放；
  - optimizer-level grad clipping；
  - 若必要再考虑 fake/real probe offload。

## 11. 当前疑问状态

已解决或基本澄清：

- fake-only turn 没有污染 student：V1 里 `student_delta=0`。
- `G_real` 没被更新：V1 里 `real_delta=0`。
- generator/student update turn 里有 `fake_delta` 是当前 D4.4 设计，不是串梯度证据。
- 首帧 pixel / LPIPS x4 生效。
- LPIPS 输入值域不是 `[0,1]`，而是 `[-1,1]`。
- 当前 `gradient_accumulation_steps=1`，G_fake accumulation 问题不影响当前实验。

仍需验证：

- LPIPS / pixel / flow / DMD 的 `z_pred` 梯度量级已完成首轮验证，见第 12 节；trainable 参数分组梯度量级如仍需要，可另开更重的 per-loss backward 验证。
- optimizer-level grad clipping 是否能减少 DMD / fake / LPIPS 引发的 spike。
- `stage3_fake_lq_proj_update_every_n_runner_steps=5` 是否比 1 更稳或更差。
- Greal/Gfake dense_full 与 student streaming 之间的结构差异是否影响 DMD 方向。
- fake-only turn 性能优化是否能不改变结果地降低训练时间。

## 12. GradScale 量级验证

验证目的：

- 复核 Gemini 提出的 `LPIPS / DMD / Flow` 梯度量级是否失衡。
- 不改正式 D4.4 训练代码，只在独立 gradcheck 代码中加 `GradScale` case。

验证代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh`

远端验证：

- 机器：`bfs6vaz4d6` + `i6hf4scd4y`，共 16 卡。
- 有效实验目录：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradscale_16gpu_v7_d4_4_20260520_1640_gradscale_16gpu_step2`
- 设定：
  - `stage3_flow_weight=1`
  - `stage3_mse_weight=1`
  - `stage3_lpips_weight=2`
  - `stage3_dmd_weight=1`
  - `stage3_fake_fm_weight=1`
  - `max_train_steps=2`
  - 强制首帧 window：`FLASHVSR_STAGE3_FORCE_RECON_START=0`

量级定义：

- Flow 不经过 `z_pred` 这条 tensor，因此 Flow 测的是 `d(flow)/d(noise_pred)`。
- MSE / LPIPS / DMD 都直接或间接作用在 `z_pred`，因此测 `d(loss)/d(z_pred)`。
- fake-only turn 中 student forward 是 `no_grad`，所以那几轮的 `*_to_z_pred=0` 是预期现象，不作为量级结论。

关键结果：

| 项 | mean_abs | l2 | 说明 |
| --- | ---: | ---: | --- |
| `flow_to_noise_pred` | `7.53e-09` | `2.36e-05` | runner 5 student turn，Flow 对 `noise_pred` 的梯度 |
| `mse_to_z_pred` | `6.98e-08` | `9.91e-04` | runner 5 student turn |
| `lpips_to_z_pred` | `6.73e-06` | `8.72e-02` | runner 5 student turn，已包含 `lpips_weight=2` 和首帧 x4 |
| `dmd_to_z_pred` | `9.54e-08` | `2.94e-04` | runner 5 student turn，fake 已更新 5 次后非零 |

补充观察：

- runner 0 的 `dmd_to_z_pred=0` 是预期：初始 `G_real` 和 `G_fake` checkpoint 相同，fake 尚未被训练出差异。
- runner 5 后，`dmd_student=0.234259`、`dmd_raw_grad_mean_abs=0.515619`，DMD 分支确实生效。
- 当前权重下，`LPIPS -> z_pred` 的 mean-abs 梯度约为：
  - MSE 的 `~96x`
  - DMD 的 `~71x`

结论：

- Gemini 关于“LPIPS / DMD / Flow 梯度量级可能失衡”的担心是成立的。
- 当前 D4.4 的 LPIPS 梯度在首帧 window 下显著主导 `z_pred` 方向，尤其相比 DMD 和 MSE。
- 这不一定说明 LPIPS 权重一定错，因为 LPIPS 本来就是感知约束；但后续如果出现残影/过平滑/纹理漂移，需要优先做 LPIPS 权重 ablation。

建议下一步：

- 保留 D4.4 当前跑法作为基线。
- 后续做一组小验证：
  - `lpips_weight=2.0` 当前基线；
  - `lpips_weight=1.0`；
  - `lpips_weight=0.5`；
  - `lpips_weight=0` 只保留 MSE + DMD + Flow。
- 每组只需短训并固定同一批测试视频，观察 ghost、纹理稳定性和运动残影。

## 13. Gemini 新问题复核：LPIPS λ=2、值域、Gradient Accumulation

### 13.1 FlashVSR 论文是否明确 `λ=2`

复核结论：

- 是。FlashVSR 论文 Stage 3 目标写为：
  `L = L_DMD + L_FM + ||x_pred - x_gt||_2^2 + λ L_lpips(x_pred, x_gt)`，
  并明确写了 `where λ = 2`。
- 当前 D4.4 YAML 也设置：
  - `stage3_lpips_weight: 2.0`
- 因此 `lpips_weight=2` 不是我随手设的工程值，而是和论文文本对齐的。

重要解释：

- `λ=2` 说明作者确实希望 LPIPS 有较强作用，尤其 Stage3 要在 one-step 下保住感知质量。
- 但论文没有给出 `LPIPS / DMD / MSE` 的梯度量级，也没有说明是否存在内部 gradient balancing。
- 所以“λ=2 正确”和“LPIPS 梯度可能主导”不矛盾：前者是论文设定，后者是我们在自己代码/数据/首帧 window 下测到的工程事实。

### 13.2 OSEDiff 的 LPIPS 值域问题

Gemini 观点：

- OSEDiff 在算 LPIPS 前，VAE decode 输出被 clamp 到 `[-1,1]`。
- 如果我们的 FlashVSR Stage3 把 LPIPS 输入错误地放在 `[0,1]`，会导致 VGG/LPIPS 激活异常。

复核 OSEDiff：

- `OSEDiff/osediff.py` 中，`output_image = vae.decode(...).sample.clamp(-1, 1)`。
- `OSEDiff/train_osediff.py` 中，`loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips`。
- `lambda_lpips` 默认也是 `2.0`。

复核我们的 D4.4：

- `x_pred` 来自 Wan VAE decode：
  - `diffsynth/models/wan_video_vae.py` 的 `single_decode()` 返回 `video.clamp_(-1, 1)`。
- `x_gt` 来自：
  - `pipe.preprocess_video(inputs["input_video"])`
  - `diffsynth/diffusion/base_pipeline.py` 默认 `min_value=-1, max_value=1`；
  - 对 `[0,1]` tensor 会执行 `video = video * 2 - 1`。
- 因此当前训练里送入 LPIPS 的 `x_pred/x_gt` 都是 `[-1,1]`。

结论：

- Gemini 这条作为排查方向是对的，但套到当前 D4.4 不是 bug。
- 当前 LPIPS 梯度大，不是因为 `[0,1]` 值域错误导致 VGG 激活极端，而是在正确 `[-1,1]` 输入下依然偏大。
- 之前 debug dump 里出现 `[0,1]` 只代表保存可视化，不代表训练 loss 输入。

### 13.3 为什么 GradScale 看起来 LPIPS “超级失衡”

需要注意两个条件：

- GradScale 是强制首帧 window：`FLASHVSR_STAGE3_FORCE_RECON_START=0`。
- 首帧 pixel 和 LPIPS 都按 Wan 首 latent 语义乘了 4。

因此这次量级验证是一个偏保守的 worst-case：

- runner 5 里 `lpips_to_z_pred_mean_abs ≈ 6.73e-06`；
- `mse_to_z_pred_mean_abs ≈ 6.98e-08`；
- `dmd_to_z_pred_mean_abs ≈ 9.54e-08`。

解释：

- 在首帧 window 下，LPIPS 对 `z_pred` 的梯度约为 MSE/DMD 的 `70-100x`。
- 随机 window 不总是包含首帧，所以平均训练过程中的 LPIPS 主导程度可能弱于这个数。
- 这个结果不直接等价于“代码错了”；它说明如果后续出现 ghost、过平滑、纹理漂移，LPIPS 权重/梯度是优先怀疑对象。

当前状态：

- 用户反馈 ghost 已经基本解决。
- 因此当前不建议立即偏离论文把 `λ=2` 改掉。
- 更合理做法是保留 `λ=2` 作为论文对齐主线，同时保留 GradScale 监控和短训 ablation 方案。

### 13.4 DMD2 如何处理稳定性

DMD2 的主要稳定手段不是“把 LPIPS 和 DMD 梯度平衡”。

DMD2 做的是：

- 不依赖 LPIPS regression 主线，或者尽量去掉原始 DMD 中的 regression loss；
- 使用 two-time-scale update rule，让 fake score estimator 多次更新；
- 典型设定是 `dfake=5`，即 fake/guidance 更新 5 次，generator 更新 1 次；
- 对 generator 和 guidance/fake 分别使用 optimizer；
- 使用 `clip_grad_norm_`，默认 `max_grad_norm=10`；
- 明确 `assert gradient_accumulation_steps == 1`，因为交替训练下 accumulation 容易破坏更新语义。

我们 D4.4 的对应状态：

- 已采用 `dfake_gen_update_ratio=5`。
- 已拆成 student optimizer 和 fake optimizer。
- 当前 YAML 是 `gradient_accumulation_steps: 1`，所以 Gemini 关于 accumulation 污染的风险当前不成立。
- 已增加 DMD loss-level clamp，用于处理 DMD 尖峰。

### 13.5 解决策略

当前主线建议：

1. 不立刻改 `stage3_lpips_weight=2.0`，因为论文明确 λ=2，且 ghost 当前已缓解。
2. 保留 `gradient_accumulation_steps=1`，不要为了吞吐打开 accumulation，除非重写 fake 分支的 accumulate 逻辑。
3. 保留 DMD loss-level clamp，继续观察 W&B 中 `dmd_student / dmd_grad / dmd_loss_clamp`。
4. 若后续再次出现 ghost/残影/过平滑，再做 LPIPS ablation：
   - `λ=2.0` 当前主线；
   - `λ=1.0`；
   - `λ=0.5`；
   - `λ=0`。
5. 如果想不偏离论文太多，可以优先做“LPIPS 梯度 cap”而不是直接改 λ：
   - 只在 `lpips_to_z_pred` 极端超过 DMD/MSE 时缩放 LPIPS 梯度；
   - 平时仍保持 `λ=2`。

我的判断：

- 当前没有证据表明 LPIPS 值域错。
- 当前没有 evidence 表明 gradient accumulation 污染。
- LPIPS 梯度大是真现象，但在论文 `λ=2` 和首帧 x4 条件下不必立刻判为错误。
- 如果当前视觉 ghost 已解决，就先不动 LPIPS；后续作为稳定性 ablation 预案。

## 14. DMD loss 的 mean reduction 是否导致 DMD 被压没

用户问题：

- 现在是否应该让 LPIPS 长期淹没 DMD。
- DMD 看起来不生效，是否是因为 DMD loss 里用了除法和 `reduction="mean"`。

### 14.1 对照 DMD2 代码

DMD2 本地参考代码：

- `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_guidance.py`

DMD2 的 generator DMD loss 也是：

- `grad = (p_real - p_fake) / torch.abs(p_real).mean(...)`
- `loss = 0.5 * F.mse_loss(original_latents, (original_latents - grad).detach(), reduction="mean")`

当前 D4.4：

- `train_flashvsr_stage3_v7_d4_4_lora.py` 中同样使用 `dmd_grad = (p_real - p_fake) / weight_factor`；
- 同样使用 `0.5 * F.mse_loss(clean_latents, target.detach(), reduction="mean")`。

结论：

- `mean reduction` 不是我们偏离 DMD2 的 bug。
- DMD 里的除法归一化也不是 bug，它是 DMD2 用来稳定 score gradient 尺度的核心步骤。
- 因此不能通过“把 mean 改成 sum”或随意取消除法来修，这会直接偏离 DMD2，并且很容易制造更大的 DMD spike。

### 14.2 为什么 GradScale 里 DMD 看起来很小

GradScale 量的是 `d(loss)/d(z_pred)` 的 mean-abs。

这有两个重要限制：

- DMD pseudo loss 对整段 latent 所有元素做 mean，因此单元素 `z_pred.grad` 天然会小；
- LPIPS 是感知网络反传到选中像素窗口，尤其强制首帧 window 时又有首帧 x4，因此局部 `z_pred.grad` 会明显大。

所以当前结果应该解释为：

- DMD 分支是生效的：runner 5 已经有 `dmd_student=0.23425940`、`dmd_raw_grad_mean_abs=0.515619`、`dmd_to_z_pred_mean_abs=9.535980e-08`；
- 但在强制首帧 window 下，LPIPS 对 `z_pred` 的局部梯度方向明显更强；
- 这说明 LPIPS 可能主导 reconstruction window 的更新，不说明 DMD 公式写错。

### 14.3 是否应该继续这样训练

不建议盲目长期让 LPIPS 淹没 DMD，但也不建议现在直接改 DMD loss 公式。

当前判断：

- 如果当前视觉 ghost 已解决，D4.4 可以继续作为论文对齐 baseline 训练。
- 继续观察 `dmd_student / dmd_grad / dmd_loss_clamp / loss_lpips / loss_mse`，确认 DMD 长期非零且没有被 clamp/skip 吞掉。
- 如果后续视觉上 DMD 像没效果，优先做短训 ablation，而不是改 DMD reduction。

建议 ablation：

1. `stage3_lpips_weight=2.0`：当前论文对齐主线。
2. `stage3_lpips_weight=1.0`：测试 LPIPS 降半后 DMD 是否更显著。
3. `stage3_lpips_weight=0.5`：进一步压 LPIPS。
4. `stage3_lpips_weight=0`：验证没有 LPIPS 时 DMD + MSE 是否能维持结构和运动。
5. `stage3_dmd_weight=0`：反向验证当前 DMD 到底是否带来可见差异。

更稳的工程方案：

- 如果只想防止极端 window 里 LPIPS 突然压倒其它项，优先考虑 LPIPS gradient cap / schedule。
- 例如前期保持 `λ=2` 但在 `lpips_to_z_pred` 极端超过 DMD/MSE 时缩放 LPIPS 分支。
- 不优先改 DMD mean/sum，因为这不是 DMD2 的原始问题。

### 14.4 还缺一个更直接的验证

GradScale 当前只量到 `z_pred.grad`，还不是最终 trainable 参数梯度。

更直接的验证是：

- 同一个 batch 下分别 backward：flow、MSE、LPIPS、DMD；
- 分别统计 LoRA 参数 grad norm；
- 分别统计 LQ projector 参数 grad norm；
- 判断 DMD 是否真的在 trainable 参数层面被 LPIPS 压没。

这比只看 loss 标量和 `z_pred.grad` 更接近最终参数更新。如果后续要决定是否调权重，应该优先做这个 per-loss parameter grad 验证。

## 15. Gemini2 审阅意见复核：是否没有 DMD、是否需要统一 loss 数量级

Gemini2 主要结论：

- 认为 `Stage3BOneStepReconLoss` 里只有 flow / MSE / LPIPS，没有真正 DMD。
- 怀疑 `to_final=True` 不是正确的 one-step x0 预测。
- 怀疑 prefix no-grad decode 会截断时序梯度。
- 怀疑 loss scaling 不匹配，需要统一数量级。

### 15.1 “没有 DMD loss”是误判

这个结论只看 `Stage3BOneStepReconLoss` 会成立，但不适用于当前 D4.4 训练。

当前 D4.4 的结构是两层：

- `Stage3BOneStepReconLoss`：只负责 student 的 one-step flow / MSE / LPIPS，并把 `z_pred` 暂存在 `pipe._stage3_last_z_pred`。
- `launch_stage3c_dual_optimizer_task()`：外层 runner 读取 `student_z_pred`，再执行 DMD student loss 和 fake FM loss。

关键代码：

- `_maybe_run_stage3c_dmd_student_loss()`：计算 `Greal/Gfake` DMD pseudo loss。
- `_stage3c_fake_fm_loss()`：用 detach 的 student `z_pred` 更新 G_fake。
- runner 中 `student_total_loss = student_loss + dmd_student_loss`。

因此当前 D4.4 不是“只有 L2 + LPIPS 的 v7-B scaffold”。Gemini2 这条是因为没有读到外层 dual-optimizer runner。

### 15.2 `to_final=True` 是需要验证的关键点，但不能直接判错

当前 scheduler：

- `diffsynth/diffusion/flow_match.py`
- `prev_sample = sample + model_output * (sigma_ - sigma)`
- `to_final=True` 时 `sigma_ = 0`

这不是随便欧拉多步积分，而是按 flow-matching 当前速度直接推到 `sigma=0` 的 one-step x0 estimate。这个逻辑符合 one-step distillation 要拿 `z_pred/x0_pred` 算 reconstruction / DMD 的需求。

但它仍然需要实证检查：

- dump `z_pred` decode 后的视频；
- 比较 teacher 50-step / student one-step 的 latent 范围、mean/std、decode 可视化；
- 如果 one-step `z_pred` 明显不在 VAE 可解码分布，再考虑重写 x0 conversion。

当前不能把它直接定为 bug。

### 15.3 prefix no-grad decode 是论文语义，不是明显 bug

FlashVSR Stage3 论文说随机选 2 个 latents decode，previous ones detached。

当前实现：

- prefix 用 no-grad 跑过 decoder causal context；
- selected window 带 grad decode；
- pixel / LPIPS 只对 selected frames 回传。

这确实是 truncated BPTT，但这是为了实现 “previous detached” 和控制显存，不是我们独创的错误路线。风险是 selected window 之外的 z_pred 不吃 pixel/LPIPS 梯度，所以必须依赖 flow / DMD 约束全局 latent；这也是为什么不能让 LPIPS 完全压没 DMD。

### 15.4 是否应该把 loss 数量级强行统一

不建议简单把 loss 标量强行调到同一数量级。

原因：

- DMD2 的 DMD pseudo loss 本身就是 mean reduction，标量小不等于没梯度。
- LPIPS / MSE / flow / DMD 的作用对象不同：flow 是 latent velocity，MSE/LPIPS 是 decode 后像素，DMD 是 distribution gradient。
- 统一标量 loss 数值不等价于统一 trainable 参数梯度。

更合理的判断标准：

- 分别量每个 loss 对 LoRA 参数的 grad norm；
- 分别量每个 loss 对 LQ projector 参数的 grad norm；
- 用参数梯度量级决定是否需要 weight schedule / gradient cap。

当前已知：

- 强制首帧 window 下，LPIPS 对 `z_pred` 的局部梯度远大于 DMD；
- 但这是首帧 x4 的 worst-case，不代表全训练平均；
- ghost 已缓解，所以当前不建议立刻改论文 `lambda=2`。

### 15.5 当前建议

短期：

- 保持当前 D4.4 作为论文对齐 baseline，不立刻强行统一 loss 数量级。
- 不改 DMD mean reduction / normalization。
- 增加或单独跑 per-loss parameter grad check，确认 LoRA / LQ projector 层面是否被 LPIPS 主导。

如果确认 LPIPS 在参数层面也长期压过 DMD：

- 优先做 `stage3_lpips_weight=2/1/0.5/0` 短训 ablation；
- 或引入 LPIPS gradient cap / warmup schedule；
- 不优先改 DMD 公式。

## 16. Per-Loss Parameter Grad 验证结果

目的：验证上一节的 `z_pred.grad` 量级失衡是否真的传到可训练参数，而不是只发生在中间 latent 上。

### 16.1 验证方式

- 不改正式训练文件，复制出独立验证脚本：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_paramgrad_lora.py`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-ParamGrad-2GPU-v7-D4-4.sh`
- 正式 D4.4 使用 ZeRO2 后 `param.grad` 被切分/隐藏，直接读 `.grad` 会得到 0；`safe_get_full_grad` 太慢。
- 因此这次采用诊断模式：`1GPU + no DeepSpeed + small shape`，只为读取真实 trainable parameter grad，不作为正式训练配置。
- 诊断设置：
  - `PARAMGRAD_SINGLEGPU=1`
  - `PARAMGRAD_SMALL=1`
  - `FLASHVSR_STAGE3_PARAM_GRAD_DISABLE_DS=1`
  - 采样参数：前 12 个 LoRA 参数 + 前 2 个 `lq_proj_in` 参数，共 `63,703,040` 个元素。
- DMD 的有效 student 梯度必须等 fake critic 更新后才出现；runner 0 的 `dmd_loss=0` 是预期，正式读取 runner 5。

### 16.2 参数梯度量级

| Loss | all L2 | LoRA L2 | LQ projector L2 |
|---|---:|---:|---:|
| Flow | 1.7389 | 1.3424 | 1.1053 |
| MSE | 0.0602 | 0.0382 | 0.0466 |
| LPIPS | 1.5213 | 0.6436 | 1.3785 |
| DMD | 0.0267 | 0.0117 | 0.0240 |

比例关系：

| 对比 | all | LoRA | LQ projector |
|---|---:|---:|---:|
| LPIPS / DMD | 56.9x | 54.8x | 57.4x |
| Flow / DMD | 65.1x | 114.3x | 46.1x |
| MSE / DMD | 2.25x | 3.25x | 1.94x |

### 16.3 结论

- Gemini 提到的梯度量级失衡在 trainable 参数层面成立：当前设置下，LPIPS 和 Flow 对参数的梯度明显大于 DMD。
- 这不是 LPIPS 输入值域 bug：当前 `x_pred/x_gt` 都是 `[-1,1]`，与 OSEDiff/FlashVSR 的 LPIPS 使用方式一致。
- 这也不是 DMD 公式 bug：DMD2 本身也使用 mean reduction 和 `mean(abs(p_real))` 归一化。
- 普通 `clip_grad_norm_` 不能解决这个问题；它只是在所有 loss 已经合成后统一防爆，不会自动让 DMD 与 LPIPS/Flow 贡献相等。
- `GradNorm algorithm` 可以动态调 loss weight，但这会偏离论文设定，也可能削弱 LPIPS 的细节约束；不建议作为第一步。

### 16.4 下一步建议

先不要直接改正式 D4.4。更稳妥的下一步是短训 ablation：

- DMD 权重增强：例如 `stage3_dmd_weight=5/10/20`，保持 `stage3_lpips_weight=2`，看 DMD 是否开始影响运动残影。
- Flow 权重降低：例如 `stage3_flow_weight=0.1/0.3`，避免 Flow 把 one-step 蒸馏拉回常规 FM。
- LPIPS 权重 ablation：`stage3_lpips_weight=2/1/0.5/0`，仅在 ghost/残影复现时做，不作为默认首选。
- 评估指标必须包含视频视觉结果，而不只看 loss；因为 LPIPS 大不等于一定坏，DMD 小也不等于完全没效果。
