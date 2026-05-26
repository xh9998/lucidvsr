# FlashVSR Stage3 v7-D 计划：作者权重、DMD Spike Guard、Streaming One-Step Validation

日期：2026-05-15

## 1. 背景

`v7-C6` 已经可以 48 卡启动并输出 loss，但有两个问题：

- `stage3_fake_fm_weight=0.1` 和 `stage3_dmd_weight=0.1` 是工程降权，不是论文对齐设定。
- `run.log` 中出现过 DMD 分支极端尖峰，例如 step 177：
  - `loss=171.024`
  - `student=0.5686`
  - `fake_loss=0.0090`
  - `dmd_student=170.4464`
  - `dmd_grad=41.5662`

这个尖峰主要来自 DMD direction，不是 flow / pixel / LPIPS 主分支整体崩掉。

## 2. 论文和参考代码对齐结论

FlashVSR Stage 3 公式写为：

`L = L_DMD + L_FM + ||x_pred - x_gt||^2 + lambda LPIPS`

其中明确给出的权重是 `lambda=2`，没有写 DMD/FM 乘 `0.1`。

DMD2 参考实现中默认值也更接近等权：

- `dm_loss_weight = 1.0`
- `denoising_loss_weight = 1.0`
- `max_grad_norm = 10.0`

OSEDiff 参考实现中也使用：

- `lambda_l2 = 1`
- `lambda_lpips = 2`
- `lambda_vsd = 1`
- `lambda_vsd_lora = 1`

因此 `v7-D` 采用更作者对齐的主设定：

- `stage3_fake_fm_weight = 1.0`
- `stage3_dmd_weight = 1.0`
- `stage3_lpips_weight = 2.0`
- `stage3_first_frame_pixel_weight = 4.0`
- `stage3_first_frame_lpips_weight = 4.0`

## 3. v7-D 相对 v7-C6 的改动

| 项目 | v7-C6 | v7-D |
|---|---:|---:|
| `stage3_fake_fm_weight` | 0.1 | 1.0 |
| `stage3_dmd_weight` | 0.1 | 1.0 |
| DMD spike guard | 无 | `skip` |
| `stage3_dmd_grad_norm_max` | 无 | 5.0 |
| validation | one-step full-sequence recon | Stage2-style streaming/KV-cache one-step |
| validation videos | 1 | 3 |
| inference | 复用旧脚本时容易混淆 | `infer_flashvsr_stage3_v7_d.py` |

## 4. DMD Spike Guard 的含义

`DMD grad clipping / norm clamp` 的本质是对 DMD direction 做异常值保护。

当前实现提供两个策略：

- `skip`：如果 `mean(abs(dmd_grad)) > stage3_dmd_grad_norm_max`，这一 step 的 DMD 项置零，不回传 DMD 梯度。
- `clamp`：如果超阈值，把 DMD direction 缩放到阈值以内，仍然回传一个被限制的 DMD 梯度。

本轮 `v7-D` 选择 `skip`，原因是目标更接近“尖峰烂数据不回传梯度”，避免极端 DMD direction 直接污染 student。

当前阈值设为 `5.0`：

- 正常 step 的 `dmd_grad` 多数在 `0.x - 3.x`。
- 已观察到的异常尖峰为 `41.5662`。
- `5.0` 可以保留正常 DMD 训练，同时跳过明显异常。

## 5. Validation / Inference 对齐

`v7-D` 的 validation 改回 Stage2-style streaming / KV-cache 路径，只是 denoising step 变成 1：

- LQ projector：沿用 Stage2 streaming 语义。
- DiT inference：使用 streaming / KV-cache。
- denoising：`num_inference_steps=1`。
- color fix：推理脚本默认仍开启。
- 输出：沿用 Stage2 的 89 -> 85 有效帧语义。

对应代码：

- 训练 validation：`FlashVSRStage3BValidationCallback`
- 推理脚本：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d.py`

## 6. 仍未完全对齐的部分

`v7-D` 不是最终完整 DMD2 复刻，还有一个重要差异：

- DMD2 论文/代码中 fake model 通常使用 TTUR 或多次 fake update。
- 当前 DiffSynth runner 只支持当前这套双 optimizer 的单步更新，`stage3_fake_update_ratio=1`。
- 后续如果要完全复刻 DMD2，需要继续扩展 runner，让 `G_fake` 可以每个 student step 更新多次。

这个问题不影响当前 `v7-D` 做“作者权重 + spike guard + validation 对齐”的验证。

## 7. 验收标准

`v7-D` smoke / 正式启动前后需要检查：

- resolved args 中 `stage3_fake_fm_weight=1.0`。
- resolved args 中 `stage3_dmd_weight=1.0`。
- resolved args 中 `stage3_dmd_spike_policy=skip`。
- resolved args 中 `stage3_dmd_grad_norm_max=5.0`。
- 训练日志中出现 `dmd_skip=0.0/1.0`。
- validation meta 中出现 `validation_mode=stage3_v7_d_streaming_kvcache_one_step`。
- validation 每次输出 3 个视频。
- C6 step100 / step200 可以用新的 `v7-D` inference 脚本重新测试。

