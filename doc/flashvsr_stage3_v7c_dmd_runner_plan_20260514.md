# FlashVSR Stage3 `v7-C` DMD Runner Plan

日期：2026-05-14

## 1. 为什么需要 `v7-C`

`v7-B` 已经完成 one-step student、random latent decode、pixel MSE、LPIPS、CPU degradation 和显存分解。但它仍不是完整 DMD。

完整 DMD 需要：

- `G_one / student`：最终 one-step streaming model；
- `G_real`：冻结的 real teacher；
- `G_fake`：单独训练的 fake distribution model；
- student optimizer；
- fake optimizer；
- 两套 optimizer / scheduler / state 的保存和恢复。

现有 DiffSynth `launch_training_task(...)` 是单 optimizer runner，不能正确表达 DMD2 的 alternating fake update。因此 `v7-C` 必须新写 Stage3 专用 runner。

## 2. 当前 `v7-C0` 范围

`v7-C0` 只做 dual-optimizer skeleton，不打开完整 DMD。

| 项 | 当前 `v7-C0` |
|---|---|
| Student | 复用 `v7-B` one-step reconstruction path |
| Student optimizer | 真实更新 LoRA + `lq_proj_in` |
| `G_fake` | 暂时使用 `Stage3CFakeScalarModel` placeholder |
| Fake optimizer | 独立 AdamW，保存 / 恢复独立 state |
| `G_real` | 暂不接入 |
| DMD loss | 暂不接入 |
| 目标 | 先验证 runner 能管理两套 optimizer/state |

`Stage3CFakeScalarModel` 不是最终 `G_fake`，只是 C0 用来证明第二 optimizer / scheduler / state path 可以工作。

## 3. 新增代码

训练入口：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py
```

配置：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c0_lora_89f_videoonly_dualopt_641data.yaml
```

启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C0-Lora-89f-VideoOnly-DualOpt-641Data.sh
```

## 4. `v7-C0` runner 设计

当前 runner：

```text
launch_stage3c_dual_optimizer_task(...)
```

单 step 逻辑：

1. 读取 `641` 数据 batch。
2. Student forward，计算 `v7-B` reconstruction loss。
3. `fake_model` 计算一个极小 skeleton loss。
4. `accelerator.backward(student_loss + fake_loss)`。
5. `student optimizer.step()`。
6. `fake optimizer.step()`。
7. 保存 student checkpoint。
8. 保存 `accelerator.save_state(...)`。
9. 额外保存 `flashvsr_stage3c_extra.pt`，包含 fake model / optimizer / scheduler state。

## 5. C0 验收标准

| 验收项 | 预期 |
|---|---|
| forward/backward | 能跑出第一条 loss |
| student checkpoint | 能保存 `.safetensors` |
| training state | 能保存 DeepSpeed / optimizer / RNG state |
| fake extra state | `training_state/step-*/flashvsr_stage3c_extra.pt` 存在 |
| fake optimizer | `fake_scale` 发生变化或 fake optimizer state 非空 |
| resume | 从 C0 training_state 恢复后能继续训练 |

## 6. 后续 C1-C5

| 阶段 | 内容 | 验收 |
|---|---|---|
| C1 | 保留 student reconstruction loss | random window / first-frame weight 正常 |
| C2 | 接 `G_real` frozen forward | `G_real` 无梯度、显存可控 |
| C3 | 接 `G_fake` forward，不更新 | DMD loss 可记录 |
| C4 | 打开 `G_fake` optimizer | fake grad norm 和 student grad norm 独立 |
| C5 | 打开完整 DMD | 20 step smoke 可保存完整 state |

## 6.1 C1-C5 具体落地顺序

`v7-C` 不允许一次性把 DMD、`G_real`、`G_fake` 和 dual optimizer 全部打开。每一步都必须有独立 smoke 目录和日志证据。

| 阶段 | 实际改动 | 不允许做的事 | 通过标准 |
|---|---|---|---|
| C1 | 固化 `v7-B` one-step reconstruction 作为 student loss；保留 `stage3_recon_num_latents=2`、prefix no-grad、pixel/LPIPS 首帧权重 | 不接 `G_real/G_fake` | 至少 4 step，保存 ckpt/state |
| C2 | 加 frozen `G_real` no-grad probe；默认用 Stage1 `v5.3.5 step-10000`，`dense_full` attention | 不把 `G_real` 放进 optimizer；不让 `G_real` 反传 | 日志出现 `real_probe=...`，且 trainable 参数量不增加 |
| C3 | 加 frozen `G_fake` no-grad probe，并计算 DMD 方向的 logging-only loss | 不更新 `G_fake`；不把 logging-only DMD 加进 student loss | 日志能同时看到 `real_probe/fake_probe/dmd_probe` |
| C4 | 给 `G_fake` 独立 optimizer，训练 fake distribution model | 不和 student optimizer 混在一起；必须保存独立 fake state | fake grad norm 非 0，`flashvsr_stage3c_extra.pt` 包含真实 fake optimizer |
| C5 | 打开 student DMD loss + fake FM loss + pixel/LPIPS | 不再使用 placeholder scalar fake | 20 step smoke 可保存完整 state，loss 分项稳定打印 |

## 6.2 DMD2 对应关系

参考代码：

```text
/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_guidance.py
/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py
```

DMD2 的关键实现不是一个普通 loss，而是 two time-scale runner：

```text
generator/student step:
  1. student 生成 fake latent/image
  2. G_real 和 G_fake 都 no-grad 估计 score / x0
  3. 用 (p_real - p_fake) / norm(p_real) 构造 distribution matching gradient
  4. 只更新 student

fake/guidance step:
  1. fake latent detach
  2. 加噪
  3. G_fake 预测噪声 / x0
  4. 用 diffusion/FM loss 更新 G_fake
```

对应到 FlashVSR：

| DMD2 名称 | FlashVSR Stage3 名称 |
|---|---|
| generator / feedforward model | `G_one` / student sparse-causal DiT |
| real_unet | Stage1 `v5.3.5` full-attention teacher `G_real` |
| fake_unet | Stage1 teacher copy `G_fake` |
| fake image / latent | `z_pred = scheduler.step(student_noise_pred, ..., to_final=True)` |
| real/fake x0 prediction | 用 `G_real/G_fake` 在 noisy `z_pred` 上预测 final latent |
| loss_fake_mean | `G_fake` 的 fake distribution FM / denoising loss |

当前 C2 只做 `G_real` no-grad probe，还没有把 DMD gradient 加进 student。

## 7. 当前不确定项

- 真实 `G_fake` 是否需要 DeepSpeed / DDP 包装，还是先单机 replica + 手动同步。
- `G_real` 的具体 checkpoint 路径需要固定为 Stage1 full-attention teacher。
- DMD timestep / noise range 需要参考 DMD2 和 OSEDiff 后单独 sweep。
- C0 placeholder fake optimizer 只验证工程路径，不代表 DMD 数学已经接入。

## 8. C0 smoke 结果

时间：2026-05-14

远端机器：`6ai5mpi47f`

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c0_lora_89f_videoonly_dualopt_641data_20260514_v7c0_smoke_r3
```

使用 Stage2 teacher / student 初始化：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors
```

验收结果：

| 项 | 结果 |
|---|---|
| Stage2 checkpoint 加载 | 通过，`lq_proj_in keys=8`，LoRA `keys=480` |
| Student forward/backward | 通过，已出 `step=1/2/3` loss |
| Fake optimizer skeleton | 通过，`fake_scale` 从 `0.000003` 更新到 `0.000010` |
| Student checkpoint | 通过，已保存 `step-1.safetensors`、`step-2.safetensors`、`step-4.safetensors` |
| Extra fake state | 通过，已保存 `training_state/step-1/flashvsr_stage3c_extra.pt`、`step-2`、`step-4` |
| 结论 | C0 dual-optimizer runner 工程路径成立，但还没有接入真实 `G_fake / G_real / DMD loss` |

关键日志：

```text
[stage3c_runner] C0 dual optimizer skeleton student_lr=1e-05 fake_lr=1e-05 fake_skeleton_weight=1e-06 fake_update_ratio=5
[stage3c_train] epoch=0 step=1 loss=0.890148 student=0.890147 fake_skeleton=0.00000100 fake_scale=0.000003
[stage3c_train] epoch=0 step=2 loss=0.177220 student=0.177219 fake_skeleton=0.00000100 fake_scale=0.000007
[stage3c_train] epoch=0 step=3 loss=0.454809 student=0.454808 fake_skeleton=0.00000100 fake_scale=0.000010
[stage3c_train] epoch=0 step=4 loss=0.594092 student=0.594091 fake_skeleton=0.00000100 fake_scale=0.000013
```

说明：

- 第一次 smoke OOM 是因为 GPU 0/1 上残留旧 full-memory 占卡进程，不是 C0 runner 本身 OOM。
- 清理旧占卡后，GPU 0/1 正常进入训练；GPU 2-7 保持低显存占卡。
- 当前 C0 的 fake branch 只是 optimizer/state 骨架验证，下一步才能进入 C1/C2，把真实 `G_real/G_fake` 逐步接入。

## 9. C2/C3 smoke 结果

时间：2026-05-14

远端机器：`6ai5mpi47f`

### 9.1 `v7-C2`：frozen `G_real` probe

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c2_lora_89f_videoonly_realprobe_641data_20260514_v7c2_smoke_r1
```

配置与脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c2_lora_89f_videoonly_realprobe_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C2-Lora-89f-VideoOnly-RealProbe-641Data.sh
```

关键设置：

| 项 | 设置 |
|---|---|
| Student | Stage2 `v6.4.1 step-6000` |
| `G_real` | Stage1 `v5.3.5 step-10000` |
| `G_real` 更新 | frozen / no-grad / 不进 optimizer |
| `G_real` attention | `dense_full` |
| 数据 | 89f, 1280x768, Takano/Yubari 0.5/0.5 |

验收结果：

```text
[stage3c_runner] C2 frozen G_real probe enabled ...
[stage3_v7_b_loss] loss=1.146446 ... compute_z_pred=True need_reconstruction=True
[stage3_v7_b_loss] loss=0.136480 ... compute_z_pred=False need_reconstruction=False
```

已保存：

```text
output/step-1.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
```

结论：

- `G_real` 可以作为独立 frozen probe 加载并执行 no-grad forward；
- `G_real` 不参与 optimizer，未改变 trainable 参数集合；
- 89f dense `G_real` probe 成本很高，后续 C4/C5 不能每步无脑全量跑 dense real/fake，需要串行、降频或改成真正 DMD 所需的最小 forward。

### 9.2 `v7-C3`：frozen `G_fake` probe

新增代码：

```text
--stage3_fake_probe_checkpoint
--stage3_fake_probe_attention_mode
--stage3_fake_probe_every
```

新增配置与脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c3_lora_89f_videoonly_fakeprobe_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C3-Lora-89f-VideoOnly-FakeProbe-641Data.sh
```

最终通过的 smoke 目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c3_lora_89f_videoonly_fakeprobe_641data_20260514_v7c3_smoke_r5_17f_512x256
```

关键设置：

| 项 | 设置 |
|---|---|
| Student | Stage2 `v6.4.1 step-6000` |
| `G_fake` probe | Stage2 `v6.4.1 step-6000` |
| `G_fake` 更新 | frozen / no-grad / 不进 optimizer |
| `G_fake` attention | `block_sparse_chunk_causal` |
| smoke 尺寸 | 17f, 512x256 |
| smoke 数据 | Takano only，`dataset_num_workers=0` |

中间失败：

- `17f, 320x192` 失败，因为 latent grid 为 `(4,12,20)`，不能被 block sparse 窗口 `(2,8,8)` 整除；
- 改为 `512x256` 后 latent grid 满足 `(2,8,8)` 分块要求。

验收结果：

```text
[stage3c_runner] C3 frozen G_fake probe enabled ...
[stage3_v7_b_loss] loss=1.030844 ... compute_z_pred=True need_reconstruction=True
[stage3_v7_b_loss] loss=0.319069 ... compute_z_pred=False need_reconstruction=False
```

已保存：

```text
output/step-1.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
```

结论：

- `G_fake` frozen probe 的模型加载、no-grad forward 和日志路径成立；
- 当前 C3 仍是 logging/probe，不更新 `G_fake`，也没有把 DMD gradient 加入 student；
- 下一步 C4 才进入真正 DMD direction：用 student `z_pred` 构造 noisy fake latent，分别过 frozen `G_real/G_fake` 得到 score/x0，记录 DMD gradient；C5 再打开 `G_fake` optimizer 交替更新。

## 10. 2026-05-15：C4/C5 DMD direction 接入与当前瓶颈

### 10.1 dense full attention 是否仍然走 FlashAttention

本次专门在远端设置：

```text
FLASHVSR_DEBUG_DIR=/mnt/task_wrapper/user_output/artifacts/debug/flashattn_v7c4_20260515_r2
```

验证 `G_real` 的 `dense_full` 分支。日志文件：

```text
/mnt/task_wrapper/user_output/artifacts/debug/flashattn_v7c4_20260515_r2/flash_attention_branches.log
```

关键输出：

```text
[flash_attention] rank=0 local_rank=0 branch=flash_attn_2 ...
[flash_attention] rank=1 local_rank=1 branch=flash_attn_2 ...
```

结论：当前 `dense_full` 不是 torch dense fallback，而是进入 `diffsynth/models/wan_video_dit.py::flash_attention()`，实际使用 `flash_attn_2`。因此当前 C4/C5 的主要慢点不是“完全没用 flash attention”。

### 10.2 C4：logging-only DMD direction probe

新增逻辑：

```text
_stage3c_probe_predict_x0(...)
_maybe_run_stage3c_dmd_probe(...)
--stage3_dmd_probe_every
```

目的：

- student 先得到 one-step `z_pred`；
- frozen `G_real` 和 frozen `G_fake` 分别在同一个 `z_pred` 上预测 `x0`；
- 只记录 `|real_x0 - fake_x0| / |real_x0|`，不回传；
- 用来确认 DMD real/fake direction 路径能串起来。

新增配置与脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c4_lora_89f_videoonly_dmdprobe_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C4-Lora-89f-VideoOnly-DMDProbe-641Data.sh
```

已验证内容：

- student one-step + pixel / LPIPS 能跑；
- frozen `G_real` flow probe 能跑；
- frozen `G_fake` flow probe 能跑；
- `dense_full` 确认走 `flash_attn_2`。

当前问题：

- C4 原始设计重复跑了 `real_probe/fake_probe` 和 DMD real/fake x0，单 step 过重；
- 已把单独 `real_probe_every/fake_probe_every` 调高，避免重复前向；
- 但 DMD x0 probe 本身仍然较慢，说明下一步需要把 probe 条件构造和模型调用进一步串行优化，而不是接受“全模型重复跑很多次”的版本。

### 10.3 C5：DMD2-style student loss 初版

新增参数：

```text
--stage3_dmd_weight
```

新增逻辑：

```text
_maybe_run_stage3c_dmd_student_loss(...)
```

当前 C5 的数学形式对齐 DMD2 的核心梯度写法：

```text
p_real = z - real_x0
p_fake = z - fake_x0
grad = (p_real - p_fake) / mean(abs(p_real))
loss = 0.5 * mse(z, (z - grad).detach())
```

工程约束：

- `G_real` no-grad / frozen / eval；
- `G_fake` 当前仍是 frozen probe，不更新；
- 只有 student `z_pred` 保留 autograd graph；
- C5 还不是完整 DMD2，因为真实 `G_fake` 独立 optimizer / alternating update 还没有打开。

新增配置与脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C5-Lora-89f-VideoOnly-DMDLoss-641Data.sh
```

当前结论：C5 已经把 DMD loss 接到 student graph，但远端 smoke 仍在验证 real/fake x0 prediction 成本。若后续仍过慢，下一步不应改 worker 或数据，而应继续拆解 `G_real/G_fake` x0 路径，确认是否有多余 VAE encode / condition 构造 / model offload 导致重复开销。

### 10.4 C5 smoke 验收结果

通过目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data_20260515_v7c5_smoke_r3_9f_256x256_nogather
```

关键设置：

| 项 | 设置 |
|---|---|
| Student | Stage2 `v6.4.1 step-6000` |
| `G_real` | Stage1 `v5.3.5 step-10000`, `dense_full`, frozen/no-grad |
| `G_fake` | Stage2 `v6.4.1 step-6000`, `block_sparse_chunk_causal`, frozen/no-grad |
| DMD loss | `stage3_dmd_weight=0.1` |
| smoke 尺寸 | 9f, 256x256，最小合法 `num_frames % 8 == 1` |
| 数据 | Takano only，`dataset_num_workers=0` |
| logging | `FLASHVSR_STAGE3C_NO_GATHER_LOG=1`，只跳过分布式日志 gather，不改变训练计算 |

关键日志：

```text
[stage3c_train] epoch=0 step=1 loss=1.703014 student=1.606964 fake_skeleton=0.00000100 fake_scale=0.000003 real_probe=0.304703 fake_probe=0.147766 dmd_student=0.096049 dmd_grad=1.029076
```

保存结果：

```text
output/step-1.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
```

额外定位：

- C5 曾在保存后卡住，定位为分布式日志 gather 阶段，而不是 DMD 前向 / backward / optimizer / checkpoint；
- 加入 `FLASHVSR_STAGE3C_NO_GATHER_LOG=1` 后正常打印并退出；
- 这个开关只影响 smoke 日志聚合，不影响正式训练计算图。

当前 C5 结论：

- DMD2-style student loss 已经真正接入 student graph；
- `G_real/G_fake` 以 no-grad 串行方式提供 real/fake x0 direction；
- `G_fake` 仍未作为 trainable fake score model 交替更新，因此这还不是最终完整 DMD2 runner；
- 下一步进入 C6 / C-final：把 placeholder fake optimizer 替换成真实 `G_fake` optimizer，并实现 alternating fake update。

## 11. 2026-05-15：C6 trainable `G_fake` 接入

### 11.1 本轮代码改动

代码入口仍为：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py
```

新增/修正内容：

- `stage3_fake_checkpoint` 不再只是占位参数；
- 当传入 `stage3_fake_checkpoint` 时，会创建完整 `FlashVSRStage3BTrainingModule` 作为 trainable `G_fake`；
- 新增 `stage3_fake_attention_mode`，正式 DMD2 对齐默认应使用 `dense_full`；
- `G_fake` 使用独立 `fake_optimizer`；
- `G_fake` 不走 Deepspeed prepare，避免破坏 student 现有 ZeRO2 路径；
- fake branch backward 后手动 `dist.all_reduce` 平均 `G_fake` 梯度，保证各 rank 的 standalone `G_fake` 参数同步；
- 保存 `flashvsr_stage3c_extra.pt` 时，如果 `G_fake` 是完整模型，只保存 trainable LoRA / `lq_proj_in` 参数，不保存整套 Wan base；
- `G_fake` 的 fake FM loss 新增为：

```text
z_fake = stopgrad(z_pred)
z_noisy = add_noise(z_fake, noise, t)
target = flow_training_target(z_fake, noise, t)
loss_fake = mse(G_fake(z_noisy, LR, t), target) * stage3_fake_fm_weight
```

这对应 DMD2 中 `compute_loss_fake(...)` 的核心思想：`G_fake` 追随当前 student 生成的 fake latent distribution。

### 11.2 C6 smoke 配置

新增配置和脚本：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh
```

关键设置：

| 项 | 设置 |
|---|---|
| Student | Stage2 `v6.4.1 step-6000` |
| `G_real` | Stage1 `v5.3.5 step-10000`, `dense_full`, frozen/no-grad |
| `G_fake` | Stage1 `v5.3.5 step-10000`, `dense_full`, trainable |
| `stage3_dmd_weight` | `0.1` |
| `stage3_fake_fm_weight` | `0.1` |
| `stage3_fake_update_ratio` | `1` |
| smoke 尺寸 | 9f, 256x256 |
| 数据 | Takano only |
| logging | `FLASHVSR_STAGE3C_NO_GATHER_LOG=1` |

### 11.3 C6 smoke 结果

通过目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r2_9f_256x256
```

关键日志：

```text
Stage3 v7-C6 trainable G_fake loaded trainable_params=570961408 checkpoint=.../step-10000.safetensors attention_mode=dense_full
Stage3 v7-C6 uses trainable G_fake also as no-grad DMD fake probe.
[stage3c_runner] ... fake_fm_weight=0.1 fake_update_ratio=1 fake_trainable_params=570961408
[stage3c_train] epoch=0 step=1 loss=1.483968 student=1.389879 fake_loss=0.00162535 fake_scale=0.000000 real_probe=0.160513 fake_probe=0.438550 dmd_student=0.092463 dmd_grad=1.008987
```

保存结果：

```text
output/step-1.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
```

`flashvsr_stage3c_extra.pt` 内容核查：

```text
fake_model_is_full_stage3=True
fake_model trainable keys=488
stage3_fake_fm_weight=0.1
stage3_fake_update_ratio=1
```

Resume 验证：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_resume_r2_9f_256x256
```

从上一条 smoke 的：

```text
.../output/training_state/step-1
```

恢复后继续跑到 `step=2`，日志确认：

```text
[stage3c_resume] loaded trainable G_fake state keys=488 missing=1019 unexpected=0
[stage3c_resume] loaded extra fake state: .../flashvsr_stage3c_extra.pt
[stage3c_train] epoch=0 step=2 loss=1.412708 student=1.309235 fake_loss=0.00766978 fake_scale=0.000000 dmd_student=0.095803 dmd_grad=1.038058
```

这说明 C6 的 fake trainable state 和 fake optimizer state 至少能被重新加载并继续训练一个 step。

结论：

- C6 已经不再是 scalar placeholder；
- trainable `G_fake` 已经真正进入 fake FM loss；
- `fake_optimizer` 和 fake trainable state 已经保存；
- `G_real` 仍 frozen/no-grad；
- DMD student loss 和 fake FM loss 可以同时运行一个 smoke step。

### 11.4 当前仍不等价完整 DMD2 的地方

| 项 | 当前状态 | 后续处理 |
|---|---|---|
| fake update ratio | 当前是每个 student step 做 1 次 fake update，`stage3_fake_update_ratio` 先作为步频控制 | DMD2 更严格是多次 fake update / generator update，需要后续加 fake substep loop |
| fake optimizer Deepspeed | 当前 standalone fake + 手动 all-reduce grad | 先保持，避免多模型 Deepspeed 破坏 student；正式长训前再评估是否需要 fake ZeRO |
| fake checkpoint | 当前只保存 trainable state，不保存 base model | 这是期望行为；恢复时需要同一个 base checkpoint + extra fake trainable state |
| 89f 正式显存 | C6 smoke 是 9f/256 | 下一步必须做 89f smoke 或短训 |

因此 C6 可以称为：

```text
trainable G_fake + fake FM smoke
```

但仍不能称为完全最终版 DMD2，直到 fake substep / long-run save-resume / 89f 正式尺寸全部验证。

### 11.5 Full-temporal C6 smoke

在 9f/256x256 最小 smoke 之后，继续验证 `89f` 的 Stage2 / Stage3 时间语义。

#### 89f / 256x256

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r3_89f_256x256
```

关键设置：

```text
--num_frames 89
--height 256
--width 256
--max_train_steps 1
```

关键日志：

```text
[stage3_v7_b_loss] loss=1.106393 flow=0.004727 mse=0.020885 lpips=0.540391 latent_window=[20,22) frame_window=[77,85) decode_latents=[0,22) recon_latents=2 decoded_frames=8 detached_context_latents=20
[stage3c_train] epoch=0 step=1 loss=1.205961 student=1.106393 fake_loss=0.00660759 real_probe=0.147064 fake_probe=0.308552 dmd_student=0.092960 dmd_grad=0.999603
```

结论：

- `89f -> 22` latent 的 Stage2/Stage3 路径可运行；
- random recon window 抽到 `[20,22)` 时，decoder 会 no-grad 推 prefix `[0,20)`，只对尾部 selected window 反传；
- fake FM、DMD student loss、real/fake probe 在 89f 时间长度下都能跑。

#### 89f / 512x256

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r4_89f_512x256
```

关键设置：

```text
--num_frames 89
--height 256
--width 512
--max_train_steps 1
```

关键日志：

```text
[stage3_v7_b_loss] loss=1.151675 flow=0.017730 mse=0.015953 lpips=0.558996 latent_window=[20,22) frame_window=[77,85) decode_latents=[0,22) recon_latents=2 decoded_frames=8 detached_context_latents=20
[stage3c_train] epoch=0 step=1 loss=1.477752 student=1.151675 fake_loss=0.00033964 real_probe=0.178705 fake_probe=0.086649 dmd_student=0.325738 dmd_grad=1.902128
```

结论：

- C6 在更高空间尺寸 `512x256` 下仍能完成 one-step + pixel/LPIPS + DMD + trainable fake FM；
- 单 step 约 58 秒，主要瓶颈是三套模型串行前向和 selected-window decoder/LPIPS；
- 更新：后续改为先完成 `2GPU` 正式尺寸完整验收，再进入 `48GPU`。顺序固定为：
  `2GPU 正式尺寸首 loss smoke -> 2GPU validation smoke -> 2GPU 短训 -> 2GPU resume smoke -> 48GPU 正式训练`。

### 11.6 正式尺寸 C6 smoke 与 validation 修复

#### 89f / 1280x768 / no validation

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r5_89f_1280x768_noval
```

关键设置：

```text
--num_frames 89
--height 768
--width 1280
--max_train_steps 1
--validation_num_samples 0
```

关键日志：

```text
[stage3c_train] epoch=0 step=1 loss=1.231092 student=1.192990 fake_loss=0.00091958 real_probe=0.016255 fake_probe=0.306179 dmd_student=0.037182 dmd_grad=0.671462
```

结论：

- C6 在正式尺寸 `89f / 1280x768` 下可以完成 forward / backward / fake update / DMD student loss / checkpoint；
- 单 step 约 2 分 43 秒；
- 峰值显存没有 OOM，具备进入 validation smoke 的基础。

#### validation OOM 定位与修复

第一次正式尺寸 validation 失败点不是训练，而是保存点 validation：

```text
output/step-1.safetensors 已保存
output/validation/step-1/sample_000/hr.mp4 已保存
output/validation/step-1/sample_000/lq.mp4 已保存
sr.mp4 未保存，rank0 validation forward OOM
```

根因：

- C6 训练时常驻 student、`G_real`、trainable `G_fake`；
- `ModelLogger.save_model()` 原本保存后立即调用 validation；
- validation callback 里没有 `torch.no_grad()` / `torch.inference_mode()`，导致 validation DiT forward 构建完整计算图；
- rank0 在保存点同时保留训练模型、辅助模型和 validation 图，触发 OOM。

修复：

- 在 `launch_stage3c_dual_optimizer_task()` 中拦截保存点 validation；
- 保存 checkpoint 时临时禁用 `ModelLogger.validation_callback`；
- 保存完 checkpoint 和 training state 后，把 `G_real` / `G_fake` / fake optimizer state 临时 offload 到 CPU；
- 手动调用 validation callback；
- validation callback 的推理主体改为 `torch.inference_mode()`；
- validation 后恢复辅助模型并加全 rank barrier。

对应代码位置：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py
_stage3c_run_validation_with_aux_offload(...)
FlashVSRStage3BValidationCallback.__call__(...)
```

#### 89f / 1280x768 / validation smoke

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r9_89f_1280x768_val_infermode
```

关键输出：

```text
output/step-1.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
output/validation/step-1/sample_000/hr.mp4
output/validation/step-1/sample_000/lq.mp4
output/validation/step-1/sample_000/sr.mp4
output/validation/step-1/sample_000/meta.json
```

结论：

- 正式尺寸 2GPU validation smoke 已通过；
- validation 当前是 one-step student validation；
- 这一步验证的是训练保存点 validation callback 可用，不等同于长训效果评估。

### 11.7 48GPU 正式训练配置预备

新增 48GPU release 配置和启动脚本，但必须等 2GPU 短训与 resume smoke 通过后再启动。

文件：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh
```

关键设置：

```text
num_frames=89
height=768
width=1280
batch_size=1
dataset_num_workers=2
takano_video_prob=0.5
yubari_video_prob=0.5
learning_rate=1e-5
stage2_attention_mode=block_sparse_chunk_causal
stage3_fake_attention_mode=dense_full
stage3_fake_fm_weight=0.1
stage3_dmd_weight=0.1
stage3_recon_num_latents=2
validation_num_samples=1
validation_num_inference_steps=1
```

启动脚本特点：

- 需要 `MASTER_ADDR` / `MASTER_PORT` / `MACHINE_RANK`；
- 使用 `accelerate_zero2_flashvsr_6node48gpu_nooffload.template.yaml`；
- 自动检查 / 下载 Takano manifest、VGG16 LPIPS 权重、Stage2 `641 step-6000` checkpoint、Stage1 `535 step-10000` checkpoint；
- snapshot 记录 train py、config、启动 sh、accelerate yaml 和 Stage2 attention 文件。

### 11.8 正式尺寸短训与 resume 验收

#### 89f / 1280x768 / short10

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_short10_89f_1280x768_noval
```

关键日志：

```text
step=1  loss=1.231092 student=1.192990 fake_loss=0.00091958 dmd_student=0.037182 dmd_grad=0.671462
step=5  loss=0.942800 student=0.877925 fake_loss=0.00111603 dmd_student=0.063760 dmd_grad=0.892547
step=10 loss=0.508211 student=0.470638 fake_loss=0.01109592 dmd_student=0.026477 dmd_grad=0.547276
```

保存结果：

```text
output/step-10.safetensors
output/training_state/step-10/flashvsr_stage3c_extra.pt
output/training_state/step-10/pytorch_model/*optim_states.pt
output/training_state/step-10/scheduler.bin
output/training_state/step-10/random_states_*.pkl
```

验证点：

- 正式尺寸下连续 10 step 可以训练；
- random latent window 在不同 step 中变化；
- `decode_latents=[0,end)` 和 `detached_context_latents` 正常出现；
- 保存点包含 student state 和 trainable `G_fake` extra state。

#### resume step10 -> step12

目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_resume12_from_short10_89f_1280x768_noval
```

resume 来源：

```text
...v7c6_short10_89f_1280x768_noval/output/training_state/step-10
```

关键日志：

```text
[stage3c_resume] loaded trainable G_fake state keys=488 missing=1019 unexpected=0
[stage3c_resume] loaded extra fake state: .../flashvsr_stage3c_extra.pt
[stage3c_resume] loaded student/fake state step=10 epoch_id=0
step=11 loss=1.082293 student=1.009620 fake_loss=0.00114987 dmd_student=0.071523 dmd_grad=0.943789
step=12 loss=0.921560 student=0.900640 fake_loss=0.00789572 dmd_student=0.013024 dmd_grad=0.396666
```

结论：

- student Deepspeed optimizer / scheduler / random state 可以恢复；
- trainable `G_fake` 参数和 fake optimizer / scheduler 可以恢复；
- C6 已通过 2GPU 全尺寸 first-loss、validation、short run、resume 四项验收；
- 下一步是 48GPU 正式训练，但必须启动后继续确认 6 节点全部进入训练循环并看到首条 loss。

### 11.9 48GPU 正式训练启动结果

实验目录：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_043300_v7c6_48gpu
```

机器：

```text
t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk
MASTER_ADDR=240.12.138.137
MASTER_PORT=29571
```

启动文件：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml
```

关键结果：

```text
step=1 loss=1.137422 student=1.073758 fake_loss=0.00815974 real_probe=0.193740 fake_probe=0.075488 dmd_student=0.055505 dmd_grad=0.798404
step=2 loss=1.373095 student=1.304030 fake_loss=0.00247871 dmd_student=0.066586 dmd_grad=0.853061
```

输出确认：

```text
output/step-1.safetensors
output/step-2.safetensors
output/training_state/step-1/flashvsr_stage3c_extra.pt
output/training_state/step-2/flashvsr_stage3c_extra.pt
output/validation/step-1/sample_000/{hr,lq,sr,meta}.*
output/validation/step-2/sample_000/{hr,lq,sr,meta}.*
```

结论：

- 48GPU 路径已经看到首两步 loss；
- checkpoint、Deepspeed state、C6 fake extra state 和 one-step validation 都能在正式多机路径上写出；
- 当前版本已经达到“真正第三阶段训练可运行”的最低验收标准；
- 后续要继续观察长训稳定性，而不是再反复停训重启。
