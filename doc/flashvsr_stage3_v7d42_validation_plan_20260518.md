# FlashVSR Stage3 v7-D4.2 快速验证计划

日期：2026-05-18

目标：用尽量短的 48 卡实验，把 Stage3 当前不确定性拆开验证，尤其是：

- D4.2 的梯度归属是否正确；
- pixel / DMD / fake FM loss 是否真的有效回传；
- Stage3 teacher 对齐是否符合 Stage2 v6.4 的目标定义；
- 当前推理鬼影是否来自 DMD、pixel loss、teacher 时间错位或 Stage3 训练目标冲突。

## 结论页

### 已确认

1. Stage3 teacher 对齐规则应改为前 22 对齐。
   - Stage2 v6.4 的监督目标不是 `GT 89 -> VAE 23 -> drop z0`。
   - Stage2 v6.4 目标是：`GT 前 85 帧 -> WAN VAE -> 22 latents`。
   - 因此 Stage3 teacher 若使用 Stage1 `nonstreaming_aligned`，会得到 23 个 latent positions；对齐 student 的 22 个 positions 时，应保留 teacher `[0,22)`，丢弃 teacher 最后一个 position `[22,23)`。
   - 旧 D3/D4.2 的 `trim_front_to_match` 是错的：它丢 teacher position 0、保留 `[1,23)`。

2. D4.2 teacher projector 当前使用的是 Stage1 非流式整段输入路径。
   - D4.2 release config：
     - `stage3_real_lq_proj_temporal_mode: nonstreaming_aligned`
     - `stage3_fake_lq_proj_temporal_mode: nonstreaming_aligned`
   - `FlashVSRLQProjIn.forward()` 在 `nonstreaming_aligned` 下调用 `forward_nonstreaming(video)`。
   - `forward_nonstreaming()` 会把整段 LQ video 一次性送入 causal conv，不走 Stage2 的 4 帧 `stream_forward()` cache 推理。
   - 89 帧输入输出 23 个 teacher/LQ positions。

3. 已修正 D4.2 新线的 teacher 对齐。
   - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
     - `nonstreaming_aligned` teacher 现在返回 `trim_tail_to_match`。
   - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
     - 新增 `trim_tail_to_match` 支持。
     - debug 日志会显示：
       - `keep_teacher_positions=[0,22)`
       - `drop_teacher_positions=[22,23)`
       - `note=teacher_front_positions_match_stage2_v64_target`
   - 本地语法检查通过：
     - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`

### 尚未确认

1. D4.2 修正为前 22 对齐后，还没有重新做 2GPU smoke。
2. 完整 Stage1 teacher deterministic forward 数值等价还没有做。
3. 还没有用 48 卡短实验证明：
   - fake-only step 不会改 student；
   - student step 不会意外改 fake；
   - pixel loss / DMD loss / fake FM loss 都确实产生有效梯度；
   - DMD 梯度方向是否和 pixel 梯度冲突。

## “Stage1 teacher deterministic forward 等价”是什么

这个实验不是训练实验，而是数值等价检查。

要验证的问题：

- Stage3 里构造的 `G_real/G_fake` teacher wrapper，是否真的等价于 Stage1 v5.3.5/USMGT 模型本身的 forward。
- 重点不是 checkpoint 是否加载成功，而是同一个 LQ/GT batch、同一个 timestep/noise 下：
  - Stage1 原始训练/推理 wrapper 的 `noise_pred`；
  - Stage3 里用于 DMD 的 teacher wrapper 的 `noise_pred`；
  是否在数值上几乎一致。

它要覆盖的关键点：

1. LQ projector temporal path：
   - Stage1 teacher 应为 `nonstreaming_aligned`。
   - 不能误用 Stage2 streaming cache。

2. attention mode：
   - Stage1 teacher 应为 full/dense attention。
   - 不能误用 Stage2 block-sparse / chunk-causal mask。

3. LQ token 对齐：
   - Stage1 teacher 89 帧产生 23 positions。
   - Stage3 对 student 22 positions 做 DMD 时，应使用 teacher `[0,22)`。

4. checkpoint loading：
   - lq_proj_in 和 LoRA 权重都必须来自同一个 Stage1 checkpoint。
   - 当前 D4.2 使用用户指定的 USMGT step-3000。

通过标准：

- 同一 batch / 同一 noise / 同一 timestep 下，Stage1 wrapper 与 Stage3 teacher wrapper 的输出 shape 一致。
- 在对齐到同一 22 positions 后，`max_abs_diff` 和 `mean_abs_diff` 接近浮点误差范围。
- 若差异明显，必须定位到 projector、attention、position slicing、checkpoint loading 或 dtype/device。

## 实验编号

### E0：D4.2 前 22 对齐 smoke

目的：确认修正后的 D4.2 在 2GPU 上能跑过第二个 generator turn，并打印正确 teacher temporal map。

设置：

- 机器：`6ikhpjzv3z` 或其他 2/4 卡测试机。
- GPU：0,1。
- 使用 D4.2 smoke config。
- 环境：
  - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1`
  - `FLASHVSR_STAGE3_TIMING_DEBUG=1`
  - `FLASHVSR_STAGE3C_NO_GATHER_LOG=1`

判定：

- 日志必须出现：
  - `mode=trim_tail_to_match`
  - `teacher_positions_before=23`
  - `keep_teacher_positions=[0,22)`
  - `drop_teacher_positions=[22,23)`
- runner 序列必须通过：
  - runner 0：generator+fake
  - runner 1-4：fake-only
  - runner 5：generator+fake
- 保存 `step-1.safetensors` 和 `step-2.safetensors`。

当前状态：待执行。

### E1：Stage1 teacher deterministic forward 数值等价

目的：确认 Stage3 teacher wrapper 真的是 Stage1 teacher，而不是“Stage1 权重 + 错 wrapper”。

设置：

- 固定一个 Takano batch。
- 固定随机种子、noise、timestep。
- 分别运行：
  - Stage1 原生 wrapper；
  - Stage3 D4.2 teacher wrapper。
- 对齐规则：teacher `[0,22)`。

输出：

- lq projector output shape。
- DiT input token shape。
- noise_pred shape。
- `max_abs_diff`、`mean_abs_diff`、`relative_l2_diff`。

判定：

- shape 必须一致。
- 差异应接近 dtype 误差。
- 如果不一致，必须先修 teacher wrapper，不能启动 48 卡正式训。

当前状态：待执行。

### E2：梯度归属/互不干扰检查

目的：证明 D4.2 没有把 fake loss 回传到 student，也没有把 student loss 回传到 fake。

设置：

- 48 卡短 smoke，最多 10 runner steps。
- 打开参数 checksum / grad norm debug。
- 分别记录：
  - fake-only step 前后 student trainable params checksum；
  - generator step 前后 fake params checksum；
  - student grad norm；
  - fake grad norm。

判定：

- fake-only step：
  - student checksum 不应变化；
  - fake checksum 应变化；
  - student grad 应为 0 或 None。
- generator step：
  - student checksum 应变化；
  - fake checksum 只应由 fake loss step 改变；
  - fake loss 不应给 student 产生梯度。

当前状态：待执行。

### E3：loss 有效回传检查

目的：确认 pixel loss、DMD student loss、fake FM loss 都不是“只打印数值但没实际更新”。

设置：

- 同一 batch，分别开关 loss，跑极短步数。
- 记录对应参数的 grad norm：
  - `grad_norm_pixel_to_student`
  - `grad_norm_dmd_to_student`
  - `grad_norm_fakefm_to_fake`

判定：

- pixel-only 时 student grad norm > 0。
- dmd-only 时 student grad norm > 0，且 `dmd_skip=0`。
- fake-only 时 fake grad norm > 0。
- 任一 loss grad 长期为 0，说明该 loss 未真正生效。

当前状态：待执行。

### E4：pixel 梯度与 DMD 梯度冲突检查

目的：解释 full DMD 训练是否可能把模型往鬼影方向拉。

设置：

- 固定同一个 batch。
- 分别 backward：
  - pixel/recon loss；
  - DMD student loss。
- 对 student trainable params 计算：
  - `||grad_pixel||`
  - `||grad_dmd||`
  - `cos(grad_pixel, grad_dmd)`
  - `||grad_dmd|| / ||grad_pixel||`

判定：

- `grad_dmd` 极小：DMD 基本没起作用。
- `grad_dmd` 极大：DMD 可能压过重建目标。
- cosine 长期明显为负：DMD 与 pixel/recon 目标冲突，可能制造残影/鬼影。

当前状态：待执行。

### E5：DMD 方向性 sanity check

目的：确认 DMD loss 的梯度符号和公式方向没反。

设置：

- 固定 batch，得到 student `z_pred`。
- 计算 DMD student loss 与 `grad_z`。
- 构造：
  - `z_minus = z_pred - epsilon * normalize(grad_z)`
  - `z_plus = z_pred + epsilon * normalize(grad_z)`
- 重新计算 real/fake probe 与 DMD loss。

判定：

- `z_minus` 的 DMD loss 应低于原始 z。
- `z_plus` 的 DMD loss 应高于原始 z。
- 若相反，DMD 梯度方向或公式符号有问题。

当前状态：待执行。

### E6：四组 48 卡 100-step 快速 ablation

目的：把鬼影来源从 pixel、DMD、fake critic 中拆出来。

共同设置：

- 使用 D4.2 前 22 对齐修正版。
- 同一 Stage1 USMGT step-3000。
- 同一 Stage2 v6.4.1 step-6000 初始化。
- 保存 step 0 / 50 / 100。
- 固定同一套 10 个合成测试集做推理。

实验组：

1. `E6A_pixel_only`
   - `stage3_dmd_weight=0`
   - `stage3_fake_fm_weight=0`
   - 只保留 pixel / LPIPS / flow recon。

2. `E6B_dmd_only`
   - pixel / recon 权重置 0。
   - 开 DMD student loss 和 fake critic。

3. `E6C_fake_only`
   - student 不更新。
   - 只更新 fake critic。
   - student 推理结果应完全不变。

4. `E6D_full_d42`
   - 当前完整 D4.2。

判定：

- `pixel_only` 若无鬼影，说明 Stage3 recon path 本身基本安全。
- `dmd_only` 若明显鬼影，优先查 DMD teacher 对齐和梯度方向。
- `fake_only` 若 student 输出变化，说明梯度/optimizer ownership 串了。
- `full_d42` 若比 `pixel_only` 更鬼影，说明 DMD/fake 目标与 pixel 目标冲突或权重过大。

当前状态：待执行。

### E7：10 个合成测试集的视觉/指标对照

目的：把短训 checkpoint 的视觉变化快速落地。

输入：

- Stage2 baseline。
- D3.2 step-1500。
- D4.2 前 22 对齐修正版短训 checkpoints。
- E6 四组 ablation checkpoints。

输出：

- 每个模型固定 10 个 mp4。
- 记录推理耗时、是否鬼影、是否锐化、是否时间抖动。

判定：

- 如果 Stage2 稳、pixel-only 稳、full/DMD 鬼影，则 DMD 路径优先处理。
- 如果 pixel-only 也鬼影，则 Stage3 recon/decode/window 或 Stage3 target 对齐仍有问题。
- 如果只有 D3.2 鬼影而 D4.2 前 22 不鬼影，旧 `trim_front_to_match` 是主因。

当前状态：待执行。

## 执行顺序

优先级从高到低：

1. E0：先证明 D4.2 前 22 对齐 smoke 能过。
2. E1：做 Stage1 teacher deterministic forward 数值等价。
3. E2：做梯度归属检查。
4. E3/E4/E5：确认 loss 有效回传与梯度方向。
5. E6：48 卡 100-step 快速 ablation。
6. E7：固定 10 个合成集推理对照。

## 风险与注意

- 不要把 E0 smoke 的 Takano-only 数据设置误带到正式 48 卡训练。Takano-only 只是为了避免 2GPU smoke 抽到远端 Yubari conductor 长尾。
- 正式训练前必须重新确认 release config 的数据比例。
- 所有停止远端实验必须先只读确认 PID 或用 tmux Ctrl-C，禁止模糊 `pkill -f`。
- 如果 E1 不通过，不能启动 E6 或正式训练。
