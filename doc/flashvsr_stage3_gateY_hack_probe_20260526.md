# FlashVSR Stage3 GateY Hack Probe

目标不是写主线正确代码，而是用最小魔改缩小 DMD-only 黄绿/灰屏问题范围。所有实验固定 `dfake=5`，不把 fake 更新频率作为变量。

## 边界

- 不修改正式 `train_flashvsr_stage3_v7_d4_4_lora.py`。
- 不修改生产 YAML/SH。
- 只新增 `hack_probe` / `gateY` 文件。
- 不在本文件完成阶段远程启动实验；后续启动时按 TeaForTwo 卡位计划分配 4GPU block。

## 新增入口

- 训练 wrapper：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_hack_probe_lora.py`
- 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GateY-HackProbe-4GPU-v7-D4-4.sh`
- YAML：
  - `stage3_gateY_hack_probe_fake_x0_equal_real_4gpu_dmdonly_dfake5.yaml`
  - `stage3_gateY_hack_probe_dmd_grad_scale0p1_clipnear_4gpu_dmdonly_dfake5.yaml`
  - `stage3_gateY_hack_probe_color_match_fake_x0_to_real_4gpu_dmdonly_dfake5.yaml`

## Variants

| Variant | 改动变量 | 如果现象缓解，说明什么 | 如果仍然失败，说明什么 |
|---|---|---|---|
| `fake_x0_equal_real` | DMD student loss 内强制 `fake_x0 = real_x0` | 黄绿/灰屏来自 DMD real/fake score 差异路径，而不是 validation 或 flow/recon 旁路 | 即使 DMD grad 归零仍坏，问题在 fake FM 更新、数据、val 或非 DMD 路径 |
| `dmd_grad_scale0p1_clipnear` | 保留 real/fake score，但 DMD grad 乘 `0.1` 并 elementwise clip 到 `0.25` | DMD 方向可能大致对，但更新幅度/spike 太危险 | 单纯缩小更新不能救，可能是方向/teacher/fake score 本身错 |
| `color_match_fake_x0_to_real` | DMD 前把 `fake_x0` per-sample mean/std 对齐到 `real_x0` | 黄绿偏色主要来自 real/fake x0 全局颜色/尺度漂移 | 颜色统计不是主因，继续查 condition / timestep / DMD 公式方向 |

每个 variant 会打印：

- 当前 variant 和修改变量。
- `real_fake_mse_before/after`。
- `real_x0 / fake_x0_orig / fake_x0_used` 的 mean/std/min/max。
- `dmd_loss_unweighted`、`dmd_grad_absmean`、`dmd_grad_absmax`。

## 远端运行模板

```bash
cd /mnt/task_runtime/lucidvsr

HACK_PROBE_VARIANT=fake_x0_equal_real \
GPU_IDS=0,1,2,3 \
MASTER_PORT=29561 \
bash wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GateY-HackProbe-4GPU-v7-D4-4.sh
```

可替换：

```bash
HACK_PROBE_VARIANT=dmd_grad_scale0p1_clipnear
HACK_PROBE_VARIANT=color_match_fake_x0_to_real
```

`dmd_grad_scale0p1_clipnear` 可通过环境变量调整：

```bash
FLASHVSR_STAGE3_HACK_PROBE_DMD_GRAD_SCALE=0.1
FLASHVSR_STAGE3_HACK_PROBE_DMD_GRAD_ABSMAX=0.25
```

## 验收

先跑 20/50/100/220 step，不跑 1000。每个结果进入 `visual-and-metrics-judge`：

- 如果 `fake_x0_equal_real` 不坏，确认 DMD real/fake 差异路径是破坏源。
- 如果 `dmd_grad_scale0p1_clipnear` 明显缓解，优先调 DMD grad 归一化/clip/spike guard。
- 如果 `color_match_fake_x0_to_real` 明显缓解，优先查 fake/real x0 颜色统计、fake_lq_proj_in、condition path。
