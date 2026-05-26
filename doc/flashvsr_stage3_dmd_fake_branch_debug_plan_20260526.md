# FlashVSR Stage3 DMD/Fake Branch Debug Plan 2026-05-26

## 背景

旧 OF-fast 四组过拟合实验里，`OF-D = DMD + fake-only` 出现绿色/灰屏/颜色错误，而 `OF-A = full loss` 相对更锐利、更鲜艳。这说明 DMD/fake 支路需要单独严肃 debug。

当前不能简单下结论为“DMD 不该用”。更准确的判断是：

- DMD/fake-only 分支在当前实现下不能单独稳定训练 student；
- full loss 没有同样崩掉，可能是 flow / MSE / LPIPS 把颜色和内容锚住了；
- 如果 DMD 正确，它理论上应该能在没有 pixel loss 的情况下提供有意义的分布蒸馏方向，至少不应快速把视频推成绿色/灰屏。

## 总目标

把 DMD/fake 分支拆成可验证的小问题，直到能回答：

1. `G_real` score 是否是正确 teacher 分布。
2. `G_fake` critic 是否训练在正确样本、正确 timestep、正确 noisy latent 上。
3. student 的 DMD gradient 方向是否合理。
4. fake branch 是否因为优化器、detach、更新频率、LQ projector wrapper 或 timestep/noise 不一致而提供错误梯度。
5. DMD-only 是否能在固定小样本上至少稳定改善，而不是颜色崩坏。

## 当前已知风险点

| 风险点 | 为什么关键 | 当前症状 |
| --- | --- | --- |
| fake critic 输入不对 | DMD 的梯度依赖 fake/real score 差；fake 输入错会直接给错方向 | OF-D 颜色崩坏 |
| G_real/G_fake wrapper 语义 | Stage1 checkpoint 如果套 Stage2 streaming wrapper，teacher 语义可能变 | 之前 review 已发现过 projector temporal mode 风险 |
| shared timestep/noisy latent | DMD student 和 fake probe 必须比较同一个 noisy point | D7/D4.4 已修过一轮，但仍需独立验证 |
| fake 更新频率 | fake 太弱或太强都会让 guidance 失真 | D-only 更容易暴露 |
| DMD loss/grad guard | 单次 spike 可能把 student 推坏 | 40 卡 D4.4 曾出现 DMD spike 后 NaN |
| detach/optimizer 隔离 | fake-only turn 不能污染 student；student turn 不能错误更新 G_real | 已验证部分，但 D-only 仍异常 |

## 分阶段验证计划

### DMD-0: 固定输入一致性

目的：排除数据和退化差异。

- 使用预生成固定 LQ/GT，不再在线随机退化；
- 同一批 `.pt` 同时喂给 DMD-only、recon-only、flow-only；
- 固定随机 seed、固定 timestep、固定 noise，用于单步 debug。

验收：

- 四组看到的 `lq_video` / `gt_video` 完全一致；
- debug dump 的 LQ/GT md5 或 tensor checksum 一致；
- DMD-only 绿色问题仍复现时，才能继续归因到 DMD/fake。

### DMD-1: G_real / G_fake forward 等价性

目的：确认 real/fake score 网络的基础 forward 语义没错。

- 固定同一个 `x_t`、timestep、LQ 条件；
- 分别跑 `G_real`、`G_fake`、student；
- 检查输入 shape、LQ projector temporal mode、attention mode、token alignment；
- 对比 Stage1 teacher 应该使用的 `nonstreaming_aligned` projector 和当前 wrapper 实际行为。

验收：

- `G_real` 权重冻结且 forward deterministic；
- `G_fake` 初始权重和 `G_real` / Stage1 checkpoint 的差异符合预期；
- real/fake 不应出现异常颜色尺度、NaN、极端 norm；
- 记录每层输入 token 数和 projector mode。

### DMD-2: fake critic 单独训练 sanity

目的：确认 fake branch 能正确拟合 fake distribution，而不是训练成错误 critic。

- 冻结 student，只生成固定 fake samples；
- 只训练 `G_fake`，不更新 student；
- 固定 20 到 100 fake steps，看 fake loss 是否稳定下降；
- 每隔若干步保存 fake score 统计和 fake model 对同一输入的输出 norm。

验收：

- fake loss 下降但不 NaN；
- fake output norm 不漂到极端；
- fake-only turn 确认 student_delta=0、G_real_delta=0、G_fake_delta>0。

### DMD-3: DMD gradient 方向检查

目的：确认 DMD 给 student 的梯度不是反方向或尺度异常。

- 固定一个 batch 和 timestep；
- 分别计算 flow / recon / DMD 对 trainable params 的 grad norm；
- 对同一参数子集记录 cosine similarity；
- 做一次极小 learning rate 的 DMD-only 单步更新，比较更新前后 real/fake score 差是否朝预期方向移动。

验收：

- DMD grad norm 不应比其他 loss 大数十倍或小到完全无效；
- DMD 单步更新后，teacher-defined objective 应改善；
- 如果 DMD 单步更新直接让 pixel preview 变绿，说明梯度方向/尺度有硬问题。

### DMD-4: DMD-only overfit 最小闭环

目的：DMD-only 必须在固定 4 视频上稳定，而不是依赖 MSE/LPIPS 救场。

- 固定 LQ/GT；
- 只开 `dmd=1, fake_fm=1`；
- 从 1, 2, 5, 10, 20, 50, 100 student steps 保存；
- 同时保存 `z_pred` preview、G_real/G_fake score、DMD grad norm。

验收：

- 至少不能出现绿色/灰屏/颜色崩坏；
- 如果 DMD-only 仍崩，必须先修 DMD/fake，再讨论 full loss。

### DMD-5: 和 full loss 的组合验证

目的：确认 DMD 加入后是正贡献，不只是被 pixel/flow 淹没。

- 固定 LQ/GT；
- 比较：
  - `recon + flow`
  - `recon + flow + DMD`
  - `recon + flow + DMD` 但 DMD weight 小一档
  - `recon + flow + DMD` 但 fake update schedule 改变

验收：

- DMD 加入后不能降低颜色稳定性；
- 如果视觉提升只来自 recon/flow，DMD score 应该降权或推迟加入；
- 如果 DMD 只在某个 warmup 后有效，应记录成正式 schedule。

## 当前优先级

1. 先完成固定 LQ/GT 的 OF 220-step 对照，确认 D-only 绿色是否稳定复现。
2. 同时启动 7E：无 DMD，只在 Stage2 50-step 路径上加 MSE/LPIPS，判断 pixel-level loss 本身是否改善 Stage2。
3. 若 7E 变好而 D-only 仍崩，优先 debug fake critic，而不是继续堆 full loss。

## 2026-05-26 补充：OF-E 与下一轮 DMD 定位

用户观察最新 fixed-LQGT validation 后确认：`OF-D = DMD/fake-only` 到最后仍有严重色偏和模糊。这个现象比 full loss 的 40 卡训练更干净，因为它排除了在线退化差异，也排除了 pixel/LPIPS 分支干扰。

因此新增一组 `OF-E`：

| 编号 | Loss | 目的 |
| --- | --- | --- |
| OF-E | `flow=1, mse=1, lpips=2, dmd=0, fake_fm=0` | 只去掉 DMD/fake，验证 flow+pixel 主链路是否稳定 |

OF-E 的判读逻辑：

- 如果 OF-E 稳定、OF-D 崩：问题集中在 DMD/fake；
- 如果 OF-E 也糊：问题不只是 DMD，可能是 one-step flow 或 pixel decode/window 对齐；
- 如果 OF-A 比 OF-E 更好但 OF-D 崩：DMD 可能只能在 pixel/flow 锚定下提供弱增益，但不能单独作为可靠目标；
- 如果 OF-A 比 OF-E 更差：DMD/fake 在 full loss 里就是负贡献。

新增代码审阅文档：

`doc/flashvsr_stage3_d44_dmd_code_audit_20260526.md`

该文档逐段记录了 `v7-D4.4` 从 641 到现在所有 DMD/fake 相关新增逻辑，包括：

- `z_pred` one-step 来源；
- `G_real/G_fake` 构造；
- `DMD probe` 如何共享 timestep/noisy latent；
- `fake FM loss` 如何训练 G_fake；
- `DMD student loss` 如何构造伪 target；
- dual optimizer runner 的 dfake 调度；
- 当前最可疑的方向符号、归一化、fake critic、projector temporal mode 问题。

下一步实验不再泛泛重训 D4.4，而是拆开做：

1. `DMD tensor dump`：固定 batch 导出 `z_pred / real_x0 / fake_x0 / p_real / p_fake / dmd_grad` 的统计和 preview；
2. `sign/norm probe`：同一 batch 比较当前符号、反符号、不同 normalization 的单步更新效果；
3. `fake critic sanity`：冻结 student，只训练 fake critic，确认 fake branch 本身不会学出颜色偏移；
4. `fake lq_proj ablation`：比较 fake projector 冻结、每步更新、每 5 步更新对颜色漂移的影响。

## 并行实验矩阵

所有实验必须遵守：

- 不修改 `train_flashvsr_stage3_v7_d4_4_lora.py` 正式文件；
- 需要新增 probe / ablation 时，复制出新文件，例如 `train_flashvsr_stage3_v7_d4_4_dmd_debug_*.py`；
- 所有实验统一使用 fixed LQ/GT：`/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`；
- 实验退出后必须启动对应 GPU 组占卡；
- 每个实验的 config、启动命令、run dir 写入 `FLASHVSR_WORKLOG.md`。

| 编号 | 类型 | GPU 建议 | 是否训练 | 目的 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| OF-E | overfit | 4 GPU | 是 | `flow+MSE+LPIPS`，去掉 DMD/fake，判断主链路是否稳定 | 已在 `3ec6pb9art:0-3` 启动 |
| DMD-1 | tensor dump | 1-2 GPU | 否 | 导出 `z_pred/real_x0/fake_x0/p_real/p_fake/dmd_grad` 统计和 preview | 待写独立 debug 文件 |
| DMD-2 | sign/norm probe | 1-2 GPU | 极小单步 | 比较当前符号、反符号、不同 normalization 的方向 | 待 DMD-1 后做 |
| DMD-3 | fake critic sanity | 4 GPU | 是 | 冻结 student，只训练 G_fake，确认 fake branch 是否自己学偏色 | 待写独立 train 文件 |
| DMD-4 | fake lq_proj ablation | 4 GPU x 2-3 | 是 | 比较 fake projector 冻结、每步更新、每 5 步更新 | DMD-3 后做 |
| DMD-5 | DMD-only repaired | 4 GPU | 是 | 在修正 sign/norm/fake 后复跑 DMD-only，看绿色/灰屏是否消失 | 最后做 |

建议并行策略：

- `OF-E` 先跑满 220 step，作为主链路对照；
- `DMD-1` 和 `DMD-2` 可以在另一个节点小卡并行，因为它们不需要长训；
- `DMD-3` 不要和 `DMD-1` 混同一个文件，单独复制训练入口；
- `DMD-4/5` 等 DMD-1/2/3 给出明确证据后再跑，避免盲目开很多长实验。

## 2026-05-26 15:50 更新：DMD-0 / DMD-1 / DMD-2 首轮结果

### DMD-0：固定输入一致性已通过

运行位置：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD0_fixedlqgt_checksum_20260526/fixed_lqgt_checksum.json`

固定 LQ/GT root：

`/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`

结果：

| sample | seed | shape | GT sha256 前缀 | LQ sha256 前缀 | LQ mean |
| --- | ---: | --- | --- | --- | ---: |
| sample_00 | 273071058 | `[89,3,768,1280]` | `53d0675f47b976d1` | `4d0995a6774f4a66` | 0.466836 |
| sample_01 | 1981220916 | `[89,3,768,1280]` | `0c465a3a71b1d865` | `8a31afb4a0b62c08` | 0.437016 |
| sample_02 | 1936481890 | `[89,3,768,1280]` | `dbf3a3d217cd5817` | `f5fca55879578572` | 0.405503 |
| sample_03 | 1657547439 | `[89,3,768,1280]` | `33e5fadbcab5c4b3` | `3d81976b31cbc4c7` | 0.426428 |

结论：后续 DMD probe / OF 对照可以认为使用同一批固定输入，不再把在线退化随机性作为主要解释。

### DMD-1：张量 dump 已产出

新增独立入口：

`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_dmd_debug_lora.py`

这个入口只 monkey-patch overfit 版本里的 DMD student loss，把 `clean_latents / real_x0 / fake_x0 / p_real / p_fake / dmd_grad / noisy_latents / timestep` 落盘；没有修改正式 D44 训练文件。

首个 runner 0 dump：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_fixedlqgt_tensor_dump_20260526/dmd_dump/step_000000`

结果：`G_fake == G_real` 初始状态下 `dmd_grad=0`，这是合理 sanity，不足以判断符号。

after-fake dump：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_fixedlqgt_tensor_dump_20260526/dmd_dump/step_000001`

关键统计：

| tensor | mean | abs_mean | std | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean_latents | 0.073331 | 0.477400 | 0.619659 | -2.718750 | 2.734375 |
| real_x0 | 0.067804 | 0.489419 | 0.634342 | -2.968750 | 2.843750 |
| fake_x0 | 0.057175 | 0.470141 | 0.608172 | -2.765625 | 2.703125 |
| real_minus_fake | 0.010629 | 0.086895 | 0.112576 | -0.695312 | 0.705078 |
| p_real | 0.005527 | 0.090866 | 0.118650 | -1.265625 | 1.289062 |
| p_fake | 0.016156 | 0.103689 | 0.133902 | -1.152344 | 1.203125 |
| dmd_grad | -0.116975 | 0.956290 | 1.238917 | -7.759506 | 7.652033 |

该 batch 的 DMD loss 统计与训练日志一致：

`dmd_student=0.774298, dmd_grad=0.956290, dmd_skip=0, dmd_loss_clamp=0`

### DMD-2：符号/归一化 probe 已跑

运行位置：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD2_afterfake_sign_norm_probe_20260526/sign_norm_report.json`

在 `eps=0.01` 的 proxy 更新下：

| candidate | grad abs_mean | Δ distance to real | Δ distance to fake | 判读 |
| --- | ---: | ---: | ---: | --- |
| current `(p_real-p_fake)/mean(abs(p_real))` | 0.956290 | -0.000803 | +0.002011 | 朝 real 走，远离 fake |
| flipped `(p_fake-p_real)/mean(abs(p_real))` | 0.956290 | +0.001113 | -0.001702 | 反方向，朝 fake 走 |
| current global real norm | 0.956290 | -0.000803 | +0.002011 | 与 per-sample 等价，因为当前 batch/rank 为 1 |
| current fake norm | 0.838029 | -0.000721 | +0.001746 | 方向一致，尺度略小 |
| raw no norm | 0.086895 | -0.000086 | +0.000170 | 方向一致，但尺度小很多 |

首轮结论：

- 当前 D44 的 DMD 符号在这个 after-fake batch 上不是反的；
- `current` 方向会把 `z_pred` 拉近 `G_real`，推远 `G_fake`；
- DMD-only 绿色/灰屏问题更可能来自 fake critic 质量、fake 更新路径、DMD 尺度/归一化长期稳定性，暂时不是简单符号写反；
- 下一步应继续看 DMD-3 fake critic sanity 和 DMD-4 fake lq_proj ablation，而不是盲目翻转 DMD sign。

### DMD-1V：real/fake/student latent 解码可视化

用户进一步怀疑 `real_x0 / fake_x0` 本身可能已经不对，因此新增 decode 可视化脚本：

`wanvideo/model_training/flashvsr/tools/decode_dmd_tensor_dump.py`

该脚本只加载 Wan VAE，把 DMD tensor dump 中的 latent 解码成视频，不更新任何模型参数。

远端输出：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_decode_videos_20260526/step_000001`

本机桌面：

`/Users/lixiaohui/Desktop/stage3_DMD1_afterfake_decode_videos_20260526/step_000001`

包含：

| 文件 | 含义 | 预期用途 |
| --- | --- | --- |
| `gt_sample_00.mp4` | fixed GT 输入 | 参考原始目标 |
| `lq_sample_00.mp4` | fixed LQ 输入 | 参考退化输入 |
| `student_z_pred.mp4` | student one-step `z_pred` decode | 看 student 当前单步输出是否已经偏色/崩坏 |
| `g_real_x0.mp4` | frozen `G_real` 在 shared noisy point 上预测的 x0 | 看 Stage1 teacher wrapper / dense_full / nonstreaming_aligned 是否合理 |
| `g_fake_x0.mp4` | after-fake `G_fake` 在同一 noisy point 上预测的 x0 | 看 fake critic 更新后是否自己偏色/变灰 |
| `shared_noisy_latents_decode.mp4` | 同一 noisy latent 直接 decode | 只作噪声参考，不要求好看 |

判读规则：

- 如果 `g_real_x0` 自己已经明显偏色/灰屏：优先查 `G_real` wrapper、Stage1 checkpoint、`nonstreaming_aligned` 对齐，而不是先怀疑 fake 更新；
- 如果 `g_real_x0` 正常但 `g_fake_x0` 明显偏色/灰屏：问题集中在 fake critic 更新目标、fake optimizer、fake lq projector 或 dual-accelerator 路径；
- 如果 `student_z_pred` 已经偏色，但 `real/fake` 都正常：问题可能在 student one-step 或 DMD 梯度进入 student 的方式；
- 如果三者视觉都正常，但 DMD-only 长训后变绿：问题更可能是长期尺度/调度/梯度累积效应，而不是单步 forward 硬错。

注意：`real_x0 / fake_x0` 是在 timestep 944 的 shared noisy latent 上预测出来的 x0，不等价于完整最终 SR 视频，但作为 sanity check 不应出现系统性绿色/灰屏。

### DMD-6：D4.2 vs D4.4 fake ownership 对照

目的：

- 验证 D4.4 dual accelerator / DeepSpeed 路径是否把 fake loss 梯度错误回传到 student，或错误更新 G_real；
- 同一 fixed LQ/GT batch 下，对比 D4.2 单 runner 和 D4.4 dual accelerator 的 fake 更新归属；
- 不修改正式 D44 文件，只使用独立 debug copy。

debug 文件：

| 对照 | 文件 | 说明 |
| --- | --- | --- |
| D4.2 single-runner | `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_ownership_lora.py` | 从 D4.2 复制，只加 fixed LQ/GT 和 ownership log |
| D4.4 dual-accelerator | `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_ownership_fixed_lora.py` | 从 D4.4 复制，只加 fixed LQ/GT 和 ownership log |

远端运行：

| 对照 | 机器 / GPU | 目录 |
| --- | --- | --- |
| D4.2 | `if8r8fmkiv` GPU `0,1,2,3` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D42_ownership_debug_fixedlqgt_4gpu_singlerunner_20260526_ownership_cmp` |
| D4.4 | `if8r8fmkiv` GPU `4,5,6,7` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D44_ownership_debug_fixedlqgt_4gpu_dualaccelerator_20260526_ownership_cmp` |

共同设置：

- `FLASHVSR_STAGE3_FIXED_LQGT_ROOT=/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`
- `FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG=1`
- `stage3_dfake_gen_update_ratio=5`
- `stage3_fake_fm_weight=1.0`
- `stage3_dmd_weight=1.0`
- `stage3_flow/mse/lpips=0`
- `max_train_steps=2`

关键观测：

| turn | D4.2 student_delta | D4.2 fake_delta | D4.2 real_delta | D4.4 student_delta | D4.4 fake_delta | D4.4 real_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| runner0 generator | `0` | `-3.47` | `0` | `0` | `+20.02` | `0` |
| runner1 fake-only | `0` | `-5.67` | `0` | `0` | `+34.12` | `0` |
| runner2 fake-only | `0` | `-3.58` | `0` | `0` | `+173.31` | `0` |
| runner3 fake-only | `0` | `+4.13` | `0` | `0` | `+74.43` | `0` |
| runner4 fake-only | `0` | `+5.76` | `0` | `0` | `+120.32` | `0` |
| runner5 generator | `+6.28` | `+89.11` | `0` | `-18.04` | `-40.61` | `0` |

训练日志对应：

| 对照 | runner0 real_probe | runner0 fake_probe | runner5 dmd_student | runner5 dmd_grad |
| --- | ---: | ---: | ---: | ---: |
| D4.2 | `0.111476` | `0.127597` | `0.030178` | `0.188745` |
| D4.4 | `0.100243` | `0.023121` | `0.354319` | `0.644264` |

当前结论：

- 没有看到 fake-only turn 污染 student：D4.2/D4.4 的 fake-only turn `student_delta=0`；
- 没有看到 G_real 被更新：所有 turn `real_delta=0`；
- D4.4 的 fake 参数更新幅度明显大于 D4.2，且 runner5 的 DMD loss/grad 明显更大；
- D4.4 的 fake grad norm 打印为 `0`，这是 DeepSpeed engine 下普通 `.grad` 不可见或已被内部消费导致的观测限制，不能据此认为 fake 没有梯度；param delta 证明 fake 实际被更新了；
- 当前更像是 fake optimizer / DeepSpeed dual accelerator 路径的更新尺度、随机 DMD point 或 optimizer state 行为与 D4.2 不同，而不是简单的参数归属串线。

后续需要继续验证：

- 固定 DMD timestep/noise，而不仅固定 LQ/GT batch，再比较 D4.2/D4.4；
- 对比 D4.4 fake optimizer 的实际 LR、AdamW state、ZeRO2 包装前后参数更新尺度；
- 视觉上补做 `initial G_real/G_fake decode` 和 `after fake update decode`，判断“G_fake 更黄”是一次 fake update 后产生，还是初始化 wrapper 就不一致。

### DMD-7：固定 timestep/noise 后复跑 D4.2 vs D4.4

目的：

- 解决 DMD-6 的一个缺口：DMD-6 只固定了 LQ/GT batch，DMD timestep/noise 仍可能不同；
- 本轮显式固定：
  - `FLASHVSR_STAGE3_FIXED_DMD_TIMESTEP_ID=944`
  - `FLASHVSR_STAGE3_FIXED_DMD_NOISE_SEED=2026052601`
- 同时固定 fake FM 和 real/fake DMD probe 的加噪点。

debug 改动：

- 仍然只改 ownership debug copy，不动正式 D44：
  - `train_flashvsr_stage3_v7_d4_2_ownership_lora.py`
  - `train_flashvsr_stage3_v7_d4_4_ownership_fixed_lora.py`
- 新增 env 控制：
  - `FLASHVSR_STAGE3_FIXED_DMD_TIMESTEP_ID`
  - `FLASHVSR_STAGE3_FIXED_DMD_NOISE_SEED`

远端结果：

| 对照 | 目录 |
| --- | --- |
| D4.4 fixed-point | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D44_ownership_fixedpoint_fixedlqgt_4gpu_dualaccelerator_20260526_fixedpoint` |
| D4.2 fixed-point | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D42_ownership_fixedpoint_fixedlqgt_4gpu_singlerunner_20260526_fixedpoint` |

关键结果：

| turn | D4.2 student_delta | D4.2 fake_delta | D4.2 real_delta | D4.4 student_delta | D4.4 fake_delta | D4.4 real_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| runner0 generator | `0` | `-12.61` | `0` | `0` | `-11.50` | `0` |
| runner1 fake-only | `0` | `-10.19` | `0` | `0` | `-30.35` | `0` |
| runner2 fake-only | `0` | `-7.34` | `0` | `0` | `-184.52` | `0` |
| runner3 fake-only | `0` | `-4.23` | `0` | `0` | `-153.37` | `0` |
| runner4 fake-only | `0` | `-2.92` | `0` | `0` | `-117.96` | `0` |
| runner5 generator | `-9.91` | `-15.34` | `0` | `+8.57` | `-104.76` | `0` |

训练日志对应：

| 对照 | runner0 real_probe | runner0 fake_probe | runner5 dmd_student | runner5 dmd_grad |
| --- | ---: | ---: | ---: | ---: |
| D4.2 fixed-point | `0.172602` | `0.092322` | `0.034875` | `0.202384` |
| D4.4 fixed-point | `0.228080` | `0.211460` | `0.355521` | 未单独打印 final mean 以外数值，param delta 明显更大 |

当前结论：

- 固定 timestep/noise 后，仍没有看到参数归属串线：
  - fake-only turn 的 `student_delta=0`；
  - 所有 turn 的 `real_delta=0`；
- D4.4 的 fake 更新在 runner2/3/4 明显大于 D4.2，差距不是随机 DMD point 导致；
- D4.4 runner5 的 `dmd_student` 仍显著大于 D4.2；
- 更强指向：D4.4 dual accelerator / DeepSpeed fake optimizer 的更新尺度、state 或包装方式与 D4.2 单 runner 不等价，需要继续查 fake optimizer step / grad scale / ZeRO2 partition 行为。

### DMD-8：补齐 DMD4 every5/current

目的：

- 补齐 fake `lq_proj_in` 三组 ablation 的第三组：
  - frozen；
  - every1；
  - every5/current。
- 该组对应当前 D44 默认：fake `lq_proj_in` 可训练，但只在 `runner_step % 5 == 0` 时更新。

新增 config：

`wanvideo/model_training/flashvsr/configs/history/stage3_dmd4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4.yaml`

远端目录：

`/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4_20260526_ownership_cmp`

当前早期日志：

| runner | fake_lq_proj_update | dmd_student | dmd_grad | 备注 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1 | `0` | `0` | 初始 probe |
| 5 | 1 | `0.932228` | `1.034729` | DMD 已明显偏大 |
| 10 | 1 | `3.000000` | `1.689442` | 触发 `dmd_loss_clamp=1` |

当前早期判断：

- every5/current 并没有立刻稳定，早期 DMD 比 D4.2 fixed-point 大很多；
- runner10 已触发 DMD loss clamp，说明当前 fake/update 组合会很快产生较强 DMD 信号；
- 还需要等保存点和 validation 视频，才能判断是否对应肉眼色偏/灰屏。

## 2026-05-26 实验台账：OF-E / DMD0-4

更新时间：2026-05-26 17:05，本地只读检查远端 `run.log` / `nvidia-smi`。

| 实验 | 目的 | 机器 / GPU | 卡数 | 当前状态 | 进度摘要 | 实验目录 |
| --- | --- | --- | ---: | --- | --- | --- |
| OF-E fixedpt | `flow + MSE + LPIPS`，完全去掉 DMD/fake，验证主链路是否稳定 | `3ec6pb9art` GPU `0-3` | 4 | 运行中 | 已到 `runner=609 / student step=122`；最新 loss 约 `0.62-0.74`；之前未带 fixedpt 的 OF-E 已失败停用 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4_fixedpt_20260525_231347` |
| OF-E 非 fixedpt | 误启动版本，没有 `FLASHVSR_STAGE3_FIXED_LQGT_ROOT` | `3ec6pb9art` | 4 | 失败/废弃 | `run.log` 有 Traceback，不作为有效对照 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4_20260525_230953` |
| DMD-0 | fixed LQ/GT checksum，确认四个 overfit 样本一致 | `3ec6pb9art` | 0 | 已完成 | 4 个样本均为 `[89,3,768,1280]`，用于排除在线退化差异 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD0_fixedlqgt_checksum_20260526` |
| DMD-1 initial dump | 初始 `G_real == G_fake` sanity，dump `z_pred/real_x0/fake_x0/dmd_grad` | `3ec6pb9art` GPU `4-7` | 4 | 已完成 | 初始 DMD grad 为 0，说明初始 real/fake 数值相等 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_fixedlqgt_tensor_dump_20260526` |
| DMD-1 after-fake dump | fake 更新 5 次后 dump，观察 DMD 是否非零 | `3ec6pb9art` GPU `4-7` | 4 | 已完成 | runner5 `dmd_student=0.774298`、`dmd_grad=0.956290`、未 clamp | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_fixedlqgt_tensor_dump_20260526` |
| DMD-1V decode | 把 DMD-1 after-fake dump 的 latent 解码成视频，看 real/fake/student 是否偏色 | `3ec6pb9art` | 约 1 | 已完成 | 已同步到桌面，用户观察到 `G_fake` 比 `G_real` 更黄 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_decode_videos_20260526/step_000001` |
| DMD-2 sign/norm probe | 比较当前符号、反符号、不同 normalization | `3ec6pb9art` | 0/轻量 | 已完成 | 当前符号会拉近 real、远离 fake；反符号相反；暂不认为 DMD sign 写反 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD2_afterfake_sign_norm_probe_20260526` |
| DMD-3 fakecritic-only | 冻结 student，只训练 fake critic，检查 fake branch 自己是否稳定 | `yagex8unf4` GPU `0-3` | 4 | 运行中 | 已到 `runner=239 / step=48`；fake loss 持续输出；generator turn 打印 `dmd_probe=0.096290` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD3_fixedlqgt_4gpu_fakecritic_only_v7_d4_4_20260525_233413` |
| DMD-4 fakeproj frozen | DMD-only，fake `lq_proj_in` 冻结，判断 projector 是否引入色偏 | `yagex8unf4` GPU `4-7` | 4 | 运行中 | 已到 `runner=239 / step=48`；runner235 `dmd_student=0.084273`、`dmd_grad=0.320342` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_frozen_v7_d4_4_20260525_233413` |
| DMD-4 fakeproj every1 | DMD-only，fake `lq_proj_in` 每个 runner 更新 | `etpf5tf68s` GPU `0-3` | 4 | 运行中 | 已到 `runner=216 / step=44`；runner215 `dmd_student=0.977088`、`dmd_grad=1.106534` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every1_v7_d4_4_20260525_234500` |
| DMD-4 fakeproj every5/current | DMD-only，fake `lq_proj_in` 每 5 runner 更新，补齐 current 对照 | `if8r8fmkiv` GPU `0-3` | 4 | 运行中 | 已到至少 `runner=13 / step=3`；runner5 `dmd_student=0.932228`；runner10 `dmd_student=3.000000` 且 `dmd_loss_clamp=1` | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4_20260526_ownership_cmp` |

当前占卡/运行状态：

- `3ec6pb9art`：GPU `0-3` 跑 OF-E；GPU `4-7` 占卡；
- `yagex8unf4`：GPU `0-3` 跑 DMD-3；GPU `4-7` 跑 DMD-4 frozen；
- `etpf5tf68s`：GPU `0-3` 跑 DMD-4 every1；GPU `4-7` 占卡；
- `if8r8fmkiv`：GPU `0-3` 跑 DMD-4 every5/current；GPU `4-7` 当前已从 fixed-point ownership 退出并占卡。

阶段性判断：

- DMD-0/1/2 是一次性 probe，已经完成；
- OF-E、DMD-3、DMD-4 三组长训还在跑，尚未到最终验收；
- DMD-4 frozen 的早期 DMD 数值明显低于 every1/every5/current，当前更支持“fake `lq_proj_in` 更新策略可能与色偏/不稳定相关”这个怀疑；
- every5/current 已经出现 DMD clamp，优先继续观察它的保存点和 validation 视频。

## 2026-05-26 18:30 更新：OF-A/B/C/D/E/F 1000-step 长 overfit

目的：

- 旧 OF-A/B/C/D/E 只训到 220 student steps，过短，不足以判断“细节出不来”是训练目标问题还是训练时间不足。
- 重新把 OF-A/B/C/D/E 拉长到 1000 student steps。
- 新增 OF-F：与 OF-A 同 loss，但在数据侧启用与 Stage1 USMGT 一致的 GT sharpness，用于判断 sharp GT 是否改善细节。
- A/D/F 额外开启 DMD tensor dump，保存 `student_z_pred / G_real_x0 / G_fake_x0 / shared_noisy_latents` 的 latent；后续用 `wanvideo/model_training/flashvsr/tools/decode_dmd_tensor_dump.py` 解码成视频看 real/fake image。

本轮重要工程处理：

- `stage3_OF_*_20260526_0316xx` 第一轮启动不是有效结果：
  - pfg/qcp 缺少固定 LQ/GT 根目录，出现 `No sample_*.pt files found`；
  - pfg 的 GPU0 有 defunct 占卡进程残留，不能继续用于训练；
  - 3ec 的 GPU4-7 旧占卡残留导致 D 初次 OOM。
- 已把固定 LQ/GT 根目录同步到 `s3://lxh/tmp/stage3_overfit4_medium_fixed_lqgt_20260525`，再拉到 pfg/qcp。
- 后续有效版本以 `r2/r3` 后缀为准。

当前有效长训表：

| 实验 | Loss 组合 | 机器 / GPU | 卡数 | 当前状态 | 有效目录 |
| --- | --- | --- | ---: | --- | --- |
| OF-A-1000 r2 | `flow + MSE + LPIPS + DMD/fake` | `qcpdgx65xx` GPU `0-3` | 4 | 已出 loss，继续跑；开启 DMD tensor dump | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_A_1000_fixedlqgt_4gpu_full_v7_d4_4_r2_20260526_032423` |
| OF-B-1000 r2 | `MSE + LPIPS` | `qcpdgx65xx` GPU `4-7` | 4 | 已出 loss 和 `step-1` validation，继续跑 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_B_1000_fixedlqgt_4gpu_recononly_v7_d4_4_r2_20260526_032427` |
| OF-C-1000 r2 | `flow only` | `pfg986en8d` GPU `1,2,4,5` | 4 | 已出 loss 和 `step-1` validation，继续跑；避开 pfg GPU0/3 异常 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_C_1000_fixedlqgt_4gpu_flowonly_v7_d4_4_r2_20260526_032432` |
| OF-D-1000 r3 | `DMD/fake only` | `3ec6pb9art` GPU `4-7` | 4 | r2 因旧占卡残留 OOM，r3 已出 loss，继续跑；开启 DMD tensor dump | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_D_1000_fixedlqgt_4gpu_dmdfakeonly_v7_d4_4_r3_20260526_032641` |
| OF-E-1000 | `flow + MSE + LPIPS`，无 DMD/fake | `etpf5tf68s` GPU `4-7` | 4 | 已出多条 loss 和 `step-1/2` validation，继续跑 | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_1000_fixedlqgt_4gpu_flow_recon_v7_d4_4_20260526_031700` |
| OF-F-1000 | OF-A + GT sharpness | `if8r8fmkiv` GPU `4-7` | 4 | 已出 loss 和 `step-1` validation，继续跑；开启 DMD tensor dump | `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_F_1000_sharpgt_4gpu_full_v7_d4_4_20260526_031705` |

注意：

- OF-F 没有使用固定 LQ/GT tensor root，因为固定 root 会绕过在线 GT sharpness；它使用同一 4 个 overfit 视频和固定 seed，在线生成一次带 sharp GT 的 batch 后缓存。
- OF-A/E/C/B/D 使用相同固定 LQ/GT root，便于排除在线退化差异。
- pfg 的 GPU0 有无法 kill 的 defunct 进程占显存，当前只用 `1,2,4,5` 跑 OF-C。
