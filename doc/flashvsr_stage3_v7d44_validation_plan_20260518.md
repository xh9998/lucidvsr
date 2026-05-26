# FlashVSR Stage3 v7-D4.4 验证计划与结果

日期：2026-05-18

目标：围绕 D4.4 dual Accelerator + dual DeepSpeedPlugin 版本做最短路径验证。D4.4 的目的不是重新证明 Stage1 teacher deterministic forward 等价；这个已经做过多次，本文不再列为阻塞项。本文重点验证 D4.4 是否能稳定跑 48 卡、fake/student 梯度归属是否正确、各 loss 是否真的生效，以及 fake backward 异常慢到底来自哪里。

## 版本范围

D4.4 代码与启动文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`

D4.4 语义：

- student/generator：一个 Accelerator + DeepSpeedPlugin + optimizer/scheduler。
- trainable `G_fake`：另一个 Accelerator + DeepSpeedPlugin + optimizer/scheduler。
- `G_fake` 每个 runner step 更新。
- student/generator 每 `stage3_dfake_gen_update_ratio=5` 个 runner step 更新一次。
- fake FM loss 使用当前 `z_pred.detach()`，不回传 student。
- Stage1-like teacher/fake 使用 `nonstreaming_aligned` projector 和 `dense_full` attention。
- teacher 23 positions 对齐 student 22 positions 时使用前 22，即 `trim_tail_to_match`。

## 当前已知事实

1. D4.4 fake 没有误训 full WAN body。
   - 2GPU e1 smoke 日志：`fake_trainable_params=570961408`。
   - 参数分组：`lora=283115520`、`lq_proj_in=287845888`。
   - 没有 `dit_base_unexpected`。

2. D4.4 fake dense_full 已走 flash-attn。
   - 2GPU e1 smoke 记录：`branch=flash_attn_2`。
   - shape：`(1, 84480, 12, 128)`。

3. fake backward 仍然很慢。
   - 2GPU e1 runner 0：`fake_backward_sync=172.897s`、`student_backward=6.129s`。
   - 2GPU e1 runner 1 fake-only：`fake_backward_sync=45.945s`。
   - 暂时排除 full model 误训、缺 flash-attn、fake offload 三个原因。

4. D4.4 48 卡正式 run 已启动，用于先出结果。
   - run name：`train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
   - 主机 tmux：`v7d44_48gpu`
   - 自动 wandb 离线打包 tmux：`wandb_package_v7d44_dfake5`
   - wandb S3：`s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1.tar.gz`
   - 初始 package loop 曾因训练进程尚未出现而退出；已在 05:35 重启，interval 为 900s，S3 包已刷新。

## 结论页

### 已确认

- D44-0 通过：48 卡 D4.4 已完成初始化，跑出至少两个 generator turn 和多个 fake-only turn。
  - run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - 日志确认：`D4.4 dual Accelerate DeepSpeed engine`。
  - fake 参数分组仍为 `lora=283115520`、`lq_proj_in=287845888`，没有 full WAN body 误训。
  - fake DeepSpeed config 为 Stage1 同款 no-offload：`deepspeed_zero2_flashvsr_nooffload.json`，日志确认 `fake_ds_zero_stage=2`、`fake_ds_offload=0`。
  - runner 0：`loss=0.857881`、`student=0.800874`、`fake_loss=0.05700695`、`real_probe=0.054306`、`fake_probe=0.167886`。
  - timing：`data=97.220s`、`student=15.177s`、`probe=18.110s`、`dmd=18.187s`、`fake=9.069s`、`fake_backward_sync=68.595s`、`student_backward=15.511s`、`save_sched=106.844s`。
  - runner 5，也就是第二个 generator turn：`loss=1.197141`、`student=0.923289`、`fake_loss=0.03292919`、`dmd_student=0.240923`、`dmd_grad=0.531402`。
  - runner 5 timing：`data=0.502s`、`student=15.049s`、`dmd=17.810s`、`fake=9.153s`、`fake_backward_sync=63.379s`、`student_backward=15.372s`、`save_sched=105.583s`。
  - 6 个节点 GPU 均稳定高显存、高利用率；抽查从节点每台约 11 个匹配进程，8 张 GPU 当前 100% 利用率。
  - 初步判断：48 卡下 fake backward/sync 明显低于 2GPU smoke 的 172s，但仍是主要耗时之一；generator turn 还会叠加 DMD/student backward 与保存开销。

- D44-1 第一轮速度观察通过，但速度仍不算理想。
  - fake-only runner 在 48 卡上通常 `fake_backward_sync=15-55s`，明显优于 2GPU 的 `45-64s` 和首个 generator turn 的 60s+。
  - step 保存约 `105-107s`，会周期性压低整体吞吐；用户要求不减少保存点，所以这里只记录，不作为当前修复项。
  - 当前瓶颈已经不再像 D4.1/D4.2 那样表现为 rank 长时间互相等待到完全低利用；至少当前抽查时 6 节点均为高利用。

- 2026-05-18 20:30 更新：根据用户反馈，D44 验证重点从“继续解释慢点/残影”收窄到“代码语义是否正确、梯度是否串线、实现是否只是功能跑通但不严谨”。
  - D44-6 残影/梯度冲突验证暂时取消：用户反馈 D6 残影已经消失，不再优先排查。
  - D44-4/D44-5 暂不优先：48 卡 D4.4 当前步速已到约 60s 量级，fake backward/sync 虽仍波动，但不是当前最需要用卡验证的问题。
  - 已在本地和 4 卡验证机 `6ikhpjzv3z` 做代码级 correctness 审查，不释放 GPU、不影响占卡。
  - 审查确认：
    - fake FM loss 内部使用 `clean_latents.detach()`，fake loss 不应回传 student；
    - DMD real/fake probe 在 `torch.no_grad()` 下运行，且 DMD target 使用 `.detach()`，DMD student loss 只应把梯度送到 student `clean_latents`；
    - fake 使用 `fake_accelerator.backward(fake_loss)`，student 使用 `accelerator.backward(student_total_loss)`，不是合成一个总 loss 一次 backward；
    - optimizer step 分离为 `optimizer.step()` 与 `fake_optimizer.step()`；
    - `fake_probe_model` 在 fake DeepSpeed prepare 后重新绑定为 prepared `fake_model`，DMD 用的是当前 trainable fake critic；
    - dfake 调度为 `runner_step % stage3_dfake_gen_update_ratio == 0`，即 `0,5,10...` 更新 student/generator，每步更新 fake；
    - 远端正式代码存在 `trim_tail_to_match`，即 Stage1 teacher 前 22 对齐 student 22。

### 风险与未确认

- fake backward 慢点尚未定位到具体 kernel/reduction；目前只能说不是 full WAN 误训、不是 flash-attn 缺失。
- D4.4 虽然更贴近 dual optimizer / dual model 语义，但未证明训练结果一定优于 D3.2/D4.2。
- 16 卡 ownership/grad 细查尚未完成；当前在 `bfs6vaz4d6` GPU0/1 跑的是 D4.4 2GPU timing/runner smoke。
- 2026-05-18 05:49 更新：`bfs6vaz4d6` 的 2GPU timing smoke 已停止并恢复占卡，因为 GPU1 长时间 0% 等待，不适合继续占用做低效验证。D44-2/D44-3 改为待用户醒后决策是否专门排卡。
- 2026-05-18 05:50 更新：48 卡已推进到 runner 15 / step 4；主机 GPU 持续高利用。runner 15 出现 `fake_loss=0.00000000`，但同条记录中 `fake_update=1`、`dmd_student=0.213833`、`dmd_grad=0.502822` 正常。该点暂不判错，列为需要后续观察的 loss 异常候选。
- 2026-05-18 05:51 更新：wandb offline package loop 已在 14:50:50 第二次成功上传，S3 包大小约 `107535` bytes，自动 offload 确认不是一次性上传。
- 2026-05-18 05:52 更新：runner 16/17 的 fake loss 已恢复非零，分别为 `0.00718069`、`0.09564392`，因此 runner 15 的 `fake_loss=0.00000000` 更像单 batch 偶发/格式化极小值，仍保留观察但不作为立即阻塞项。
- 2026-05-18 20:30 更新：runner 0 ownership run 中 `dmd_student=0` / `dmd_grad=0` 的解释更新：
  - 这不是立即异常。
  - runner 0 计算 DMD student loss 时，`G_real` 和 `G_fake` 都刚从同一个 Stage1 USMGT checkpoint 初始化，且 fake 还没有完成第一次 optimizer step；
  - DMD student loss 使用 `real_x0 - fake_x0` 的 score difference。如果 real/fake 初值完全相同或非常接近，`p_real - p_fake` 就会为 0 或接近 0；
  - 正式 48 卡日志中 runner 5 以后已经出现非零 DMD，例如 runner 5 `dmd_student=0.240923`、runner 10 `dmd_student=0.249116`，说明 fake 更新后 DMD 项开始生效；
  - 因此“首个 generator turn DMD 为 0”更符合 DMD2-style 初始化逻辑，不应当作 DMD 被 detach 错误截断的证据。

## 实验 D44-0：48 卡启动健康检查

目的：确认 48 卡 D4.4 正式 run 真的进入训练，而不是只完成初始化或卡在 checkpoint/data。

设置：

- 6 节点 48 卡。
- run：`20260518_v7d44_48gpu_fresh1`。
- `FLASHVSR_STAGE3_TIMING_DEBUG=1`。
- `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0`。
- `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0`。

检查项：

- 6 个节点各 8 个训练进程存在。
- GPU 显存稳定，不出现单节点空跑。
- run.log 出现：
  - `D4.4 dual Accelerate DeepSpeed engine`
  - `fake_trainable_groups`
  - `fake_ds_zero_stage=2`
  - `fake_ds_offload=0`
  - 至少一条 `stage3c_train`
  - 至少一条 `stage3_timing`

判定：

- 通过：至少完成 runner 0 和 runner 1，且没有 NCCL/CUDA/DeepSpeed error。
- 失败：卡初始化超过合理时间、任一节点 OOM、rank 数不完整、或 loss/timing 不出现。

当前状态：进行中。

2026-05-18 05:30 更新：

- 已完成 runner 0。
- GPU 显存稳定，rank0 节点 GPU 利用率约 97-100%。
- 2026-05-18 05:35 更新：已继续完成 runner 1-4 fake-only，无 NCCL/CUDA/DeepSpeed error。
  - runner 1：`fake_loss=0.00931268`、`fake_backward_sync=19.763s`
  - runner 2：`fake_loss=0.01540674`、`fake_backward_sync=15.244s`
  - runner 3：`fake_loss=0.01425055`、`fake_backward_sync=45.479s`
  - runner 4：`fake_loss=0.05309864`、`fake_backward_sync=55.245s`
  - 非首步 `data` 约 `0.4-0.6s`，说明 dataloader 没有持续卡死。

2026-05-18 05:42 更新：

- 已完成 runner 5-8。
- runner 5 是第二个 generator turn：
  - `loss=1.197141`
  - `student=0.923289`
  - `fake_loss=0.03292919`
  - `dmd_student=0.240923`
  - `dmd_grad=0.531402`
  - `fake_backward_sync=63.379s`
  - `student_backward=15.372s`
  - `save_sched=105.583s`
- runner 6-8 fake-only：
  - runner 6：`fake_loss=0.07938955`、`fake_backward_sync=16.102s`
  - runner 7：`fake_loss=0.09516689`、`fake_backward_sync=21.146s`
  - runner 8：`fake_loss=0.02226195`、`fake_backward_sync=34.984s`
- 跨节点抽查：
  - `t5qdtykjsw`、`a9suya6gxe`、`67dxkwcb7m`、`ui9n6p293s`、`g48bd6x4h7`、`gx2intv5rk` 均为高显存、高利用率。
  - 从节点每台抽查到 8 张 GPU 约 100% 利用率。
- 判定：D44-0 通过。

## 实验 D44-1：48 卡速度与利用率观察

目的：确认 D4.4 在 48 卡上是否比 2GPU smoke 更可接受，尤其 fake backward/sync 是否仍然拖垮整体吞吐。

设置：

- 使用 D44-0 同一正式 run。
- 统计前 10 个 runner step 的 timing。

记录：

- `data`
- `student`
- `probe`
- `dmd`
- `fake`
- `fake_backward_sync`
- `student_backward`
- `save_sched`
- step wall time
- 6 节点 GPU 利用率快照

判定：

- 如果 fake_backward_sync 在 48 卡显著下降，说明 2GPU 慢主要是小规模 ZeRO/通信形态问题。
- 如果 fake_backward_sync 仍然极慢，继续做 D44-4 的 Stage1 对照和 D44-5 的 fake 参数/同步缩减实验。

当前状态：待执行。

2026-05-18 05:30 更新：

- 已有首步 timing：
  - `fake_backward_sync=68.595s`
  - `student_backward=15.511s`
  - `save_sched=106.844s`
- runner 1-4 fake-only timing：
  - runner 1：`student=10.799s`、`fake=8.795s`、`fake_backward_sync=19.763s`
  - runner 2：`student=15.392s`、`fake=8.789s`、`fake_backward_sync=15.244s`
  - runner 3：`student=9.594s`、`fake=8.795s`、`fake_backward_sync=45.479s`
  - runner 4：`student=8.044s`、`fake=8.787s`、`fake_backward_sync=55.245s`
- 初步结论：
  - 48 卡下 fake-only step 明显快于 2GPU smoke；
  - fake backward/sync 仍有波动，runner 3/4 会升到 45-55s；
  - 首步最慢主要叠加了 probe/DMD 与 `step-1.safetensors` 保存。

2026-05-18 05:42 更新：

- 第二个 generator turn runner 5：
  - `data=0.502s`
  - `student=15.049s`
  - `dmd=17.810s`
  - `fake=9.153s`
  - `fake_backward_sync=63.379s`
  - `student_backward=15.372s`
  - `save_sched=105.583s`
- 后续 fake-only runner 6-8：
  - `fake_backward_sync=16.102s / 21.146s / 34.984s`
  - `data=0.447-0.577s`
- 当前结论：
  - 48 卡 D4.4 的 `data` 不再是主要瓶颈；
  - fake forward 约 9s，DMD 约 18s，student backward 约 15s；
  - fake backward/sync 和 checkpoint save 是当前最大可见耗时；
  - D4.4 48 卡缩短了 2GPU 下最离谱的 fake sync，但没有消除 fake critic 训练本身的重计算/同步成本。

2026-05-18 05:50 更新：

- 新增 runner 9-15：
  - runner 9 fake-only：`fake_loss=0.00695447`、`fake_backward_sync=94.174s`
  - runner 10 generator：`loss=0.789458`、`student=0.525690`、`fake_loss=0.01465193`、`dmd_student=0.249116`、`dmd_grad=0.542519`、`fake_backward_sync=41.757s`、`student_backward=15.470s`
  - runner 11 fake-only：`fake_loss=0.02635277`、`fake_backward_sync=99.130s`
  - runner 12 fake-only：`fake_loss=0.05087015`、`fake_backward_sync=20.903s`
  - runner 13 fake-only：`fake_loss=0.03671652`、`fake_backward_sync=42.620s`
  - runner 14 fake-only：`fake_loss=0.01504827`、`fake_backward_sync=19.093s`
  - runner 15 generator：`loss=0.860282`、`student=0.646449`、`fake_loss=0.00000000`、`dmd_student=0.213833`、`dmd_grad=0.502822`、`fake_backward_sync=17.647s`、`student_backward=15.101s`
- 观察：
  - 48 卡主机连续 6 次 10 秒采样均为 97-100% 利用率，没有复现长期 0%。
  - `fake_backward_sync` 仍有大波动，runner 9/11 到约 94-99s，runner 15 则降到 17.647s。
  - runner 15 的 `fake_loss=0.00000000` 需要继续观察，看是否只是偶发极小值/格式化，还是 fake FM loss 计算在某些 batch 被截断。

## 实验 D44-2：梯度归属检查

目的：确认 fake-only step 不改 student，student/generator step 不被 fake FM loss 污染。

设置：

- 16 卡机器上跑 D4.4 短验证，不占用正式 48 卡。
- 打开精简 checksum/grad debug，只跑少量 runner steps。
- 不开长训用的 noisy debug。

检查项：

- fake-only step：
  - student trainable checksum 不变。
  - fake trainable checksum 变化。
  - student grad norm 为 0 或 None。
- generator step：
  - student checksum 变化。
  - fake checksum 的变化来自 fake phase，不来自 student backward。
  - fake FM loss 使用 `z_pred.detach()`。

判定：

- 通过：checksum/grad ownership 与上述一致。
- 失败：fake-only 改 student，或 fake loss 给 student 留梯度。

当前状态：待执行。

2026-05-18 05:30 更新：

- 已在 16 卡机器中的 `bfs6vaz4d6` 上启动 D4.4 2GPU timing/runner smoke：
  - tmux：`d44_validate_2gpu`
  - run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_validate_bfs_2gpu`
  - GPU0/1 用于验证；GPU2-7 已挂占卡。
- 该验证主要用于对照 D44-0/D44-1，不替代完整 16 卡 ownership check。

2026-05-18 05:42 更新：

- 2GPU smoke 已完成 runner 0-2，仍在运行。
- 当前 timing：
  - runner 0：`fake_backward_sync=173.689s`、`student_backward=6.132s`
  - runner 1 fake-only：`fake_backward_sync=45.155s`
  - runner 2 fake-only：`fake_backward_sync=64.202s`
- 解释：
  - 2GPU 上 fake sync 仍复现慢点；
  - 48 卡上同一版本明显好一些，说明小规模 ZeRO2/通信形态会放大 fake backward/sync；
  - 但 fake 训练 570M 参数加 dense_full backward 仍然是客观重负担。

2026-05-18 05:49 更新：

- 该 2GPU smoke 已停止。
- 停止原因：
  - 已经得到 D44-1 所需的 timing 证据；
  - 后续采样显示 GPU0 仍 100%，GPU1 长时间 0%，继续跑会让 16 卡利用率不好看；
  - 用户要求不用卡时必须占卡。
- 停止方式：
  - 只对 `d44_validate_2gpu` tmux session 发 Ctrl-C；
  - 没有使用 `pkill` 或模糊 kill。
- 停止后 `bfs6vaz4d6` GPU0/1 已启动 `occupy01` 占卡，GPU0-7 当前均为 100% 利用率。
- D44-2 当前结论：
  - 完整 ownership checksum/grad 检查尚未执行；
  - 后续若要做，建议单独开短窗口，避免 2GPU dual-DeepSpeed smoke 的 rank 等待把 16 卡利用率拉低。

2026-05-18 20:03 更新：

- 已新增独立 D4.4 gradcheck 线，不改正式训练代码：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh`
- 启动 16 卡 Ownership 验证：
  - 机器：`bfs6vaz4d6` rank0 + `i6hf4scd4y` rank1。
  - run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradcheck_16gpu_v7_d4_4_ownership_20260518_d44_ownership_16gpu`
  - 启动参数：`GRAD_CASE=Ownership`、`stage3_flow/mse/lpips/dmd/fake_fm` 全开、`max_train_steps=2`。
  - 使用 Stage2 `v6.4.1 step-6000` 与 Stage1 USMGT `v5.3.5 step-3000`。
- runner 0 generator turn 结果：
  - `loss=0.847793`、`student=0.804051`、`fake_loss=0.04374290`、`real_probe=0.041056`、`fake_probe=0.073525`。
  - `dmd_student=0.000000`、`dmd_grad=0.000000`、`dmd_skip=0`；该 batch 没有证明 DMD student loss 有效，只能说明链路执行到 DMD 位置且未 skip。
  - 参数 ownership：
    - `student_delta=-9.776723e+00`
    - `fake_delta=1.448021e+00`
    - `real_delta=0.000000e+00`
  - 解释：generator turn 中 student 和 fake 都按 D4.4 设计更新；G_real 保持冻结不变。
  - timing：`data=72.372s`、`student=9.507s`、`probe=18.040s`、`dmd=18.044s`、`fake=9.029s`、`fake_backward_sync=503.511s`、`student_backward=6.999s`。
- grad norm 采集边界：
  - `student_grad_norm/fake_grad_norm/real_grad_norm` 直接从 `.grad` 读到 `0.000000e+00`。
  - 这与参数实际 delta 不矛盾，原因是模型经过 Accelerate/DeepSpeed ZeRO2 包裹后，普通 `.grad` 读取点不能作为可靠的“loss 是否回传”证据。
  - 因此 D44-2/D44-3 后续应优先用参数 delta、DeepSpeed 内部 grad 访问或单卡/非 ZeRO 专用 probe，而不是直接读 `.grad`。
- runner 1 fake-only 没有等到完成：
  - 进程长时间卡在 `deepspeed/runtime/zero/stage_1_and_2.py` 的 `all_gather_dp_groups` / optimizer step 路径。
  - 为避免 16 卡长时间低效占用，已停止 `d44_grad_ownership` 验证。
  - 停止方式：先对 `d44_grad_ownership` tmux 发 Ctrl-C；残留进程按完整脚本名只读 `pgrep -af 'train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py'` 确认后，仅 kill 明确 PID；没有使用空变量或模糊 `pkill`。
  - 停止后两台机器均重新启动 `occupy_all` 占卡，最后检查两台 8 张 GPU 均约 `125132 MiB`、`100%`。
- D44-2 当前判定：
  - 部分通过：runner 0 已证明 G_real 冻结、generator turn 会更新 student 和 fake。
  - 未完成：fake-only turn 是否只改 fake 尚未拿到参数 delta，因为 runner 1 卡在 ZeRO all-gather/optimizer step。
  - 新发现：16 卡 dual DeepSpeed debug 形态下 fake optimizer/sync 明显比 48 卡正式 run 更差，不适合作为长时间验证环境。

2026-05-18 20:30 更新：

- 根据用户要求，后续不再优先用 16 卡或 4 卡长跑 D44-2 慢速 ownership debug；重点改为代码语义审查与短验证。
- 代码级结论：
  - fake-only 语义上不会回传 student：`_stage3c_fake_fm_loss()` 对 `clean_latents` 做 `detach()` 后才构造 fake clean/noisy latents；
  - DMD student 语义上不会更新 fake/real：DMD probe 包在 `torch.no_grad()` 里，`target=(clean_latents - dmd_grad).detach()`，因此梯度只从 `F.mse_loss(clean_latents, target)` 回到 student 的 `clean_latents`；
  - fake/student backward 和 optimizer step 是分离的，不是一个 total backward；
  - formal D4.4 仍保留 dual Accelerator + dual DeepSpeedPlugin，不是手写 fake grad sync 的旧实现。
- 剩余未完全动态证明的点：
  - 还没有拿到 fake-only runner 的参数 delta 证明“只 fake 变、student 不变”；
  - 但从代码图和 detach/optimizer 分离看，目前没有发现 fake-only 改 student 的路径。

## 实验 D44-3：loss 有效回传检查

目的：确认 pixel/recon、DMD student、fake FM 都不是只打印不更新。

设置：

- 16 卡短验证。
- 固定少量 batch。
- 分别打开或记录各 loss 对应的 grad norm。

检查项：

- pixel/recon 对 student grad norm > 0。
- DMD student 对 student grad norm > 0，且 `dmd_skip=0`。
- fake FM 对 fake grad norm > 0。

判定：

- 通过：三类 loss 都有有效梯度。
- 失败：任一 loss 长期 grad 为 0 或被 detach 错误截断。

当前状态：待执行。

2026-05-18 20:03 更新：

- Ownership 用例 runner 0 间接覆盖了部分 loss 有效性：
  - `student_delta != 0`，说明 student 侧总损失确实更新了 student 参数。
  - `fake_delta != 0`，说明 fake FM loss 确实更新了 G_fake 参数。
  - `real_delta = 0`，说明 frozen G_real 没有被误更新。
- 但该 batch 的 `dmd_student=0.000000`、`dmd_grad=0.000000`，所以尚不能宣称 DMD student loss 已在该 run 中有效回传。
- 直接 `.grad` norm 在 DeepSpeed ZeRO2 下读到 0，不作为有效性证据。
- 后续若要完成 D44-3，建议改成更轻的单卡/非 ZeRO probe 或直接读取 DeepSpeed partitioned grad，而不是继续用 16 卡 D4.4 dual-engine 方式硬跑。

2026-05-18 20:30 更新：

- DMD 首步为 0 的原因已重新解释：
  - runner 0 时 `G_real == G_fake` 初始化，DMD score difference 可以为 0；
  - 后续 fake 更新后，正式 48 卡日志已出现非零 DMD student loss 和 DMD grad norm；
  - 因此首步 `dmd_student=0` 不是 DMD loss 实现错误的证据。
- 当前 D44-3 结论：
  - pixel/recon 有效性：由 student 参数 delta 和正式训练 student loss 持续变化间接支持；
  - fake FM 有效性：由 fake 参数 delta、fake_loss 非零、fake optimizer step 支持；
  - DMD student 有效性：由正式 48 卡 runner 5/10/15 等非零 `dmd_student/dmd_grad` 支持；
  - 但还没有做“单 loss 单独 backward 的精确 grad norm”动态验证。若后续必须做，建议写单卡/非 ZeRO probe，而不是完整 D4.4 dual-engine debug。

2026-05-18 21:45 更新：补充 D4.4-DMDOnly 16 卡对照。

- 目的：用户要求验证“如果去掉 pixel/recon，只保留 DMD student + fake FM，DMD 分支是否真的能单独驱动 student”，且明确不要修改正在跑的 D4.4 正式代码。
- 实现方式：
  - 不动 `train_flashvsr_stage3_v7_d4_4_lora.py`；
  - 只新增独立 config/launch：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5.yaml`
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-16GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-DMDOnly-Dfake5.sh`
  - loss 权重为 `flow/mse/lpips=0`、`dmd=1`、`fake_fm=1`、`dfake_gen_update_ratio=5`。
  - 保留 fake FM 是必要的；否则 `G_fake` 不更新，DMD 的 real/fake score difference 后续无法形成。
- 运行：
  - 机器：`bfs6vaz4d6` + `i6hf4scd4y`
  - tmux：`d44_dmdonly16`
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5_20260518_v7d44_dmdonly_16gpu`
- 关键日志：
  - runner 0 generator：`student=0.000000`、`fake_loss=0.00697420`、`dmd_student=0.000000`、`dmd_grad=0.000000`。首个 generator turn 时 `G_real/G_fake` 初始相同，DMD 为 0 正常。
  - runner 1-4 fake-only：`student=0.000000`，`mse/lpips` 打印保持 0，`need_reconstruction=False`。
  - runner 5 第二个 generator turn：`loss=0.161390`、`student=0.000000`、`fake_loss=0.01611994`、`dmd_student=0.145271`、`dmd_grad=0.411943`、`dmd_skip=0`。
  - runner 5 timing：`dmd=18.121s`、`fake_backward_sync=100.645s`、`student_backward=4.834s`。
- 结论：
  - DMD-only 对照证明 D4.4 的 DMD student branch 在 fake critic 更新后会产生非零 student loss 和非零 DMD gradient；
  - 因为 `student=0.000000` 且 pixel/recon 权重为 0，这个非零 student 目标来自 DMD，不是 MSE/LPIPS/flow；
  - 这支持“DMD 不是完全没生效”的判断。
- 需要保留的边界：
  - 这个对照只验证 DMD branch 能单独产生 loss/grad，不证明 DMD-only 视觉效果一定好；
  - fake-only 的 `fake_backward_sync` 波动仍明显，例如 runner 2/3 到 `196.618s`/`174.086s`，说明该 16 卡对照不适合作为速度基准；
  - 当前没有修改 D4.4 正式代码。

2026-05-18 22:11 更新：D4.4-DMDOnly 对照停止并释放为占卡。

- 用户询问是否继续长训，以及 validation / wandb 情况。
- 配置确认：
  - `validation_num_samples=0`，没有 validation；
  - `use_wandb=false`，不会自动同步 W&B；
  - `max_train_steps=20`，但该对照目的不是正式候选训练。
- 补充观察：
  - runner 10：`student=0.000000`、`fake_loss=0.01019344`、`dmd_student=0.160736`、`dmd_grad=0.434257`、`dmd_skip=0`。
  - 已保存：`step-1.safetensors`、`step-2.safetensors`。
  - 尝试等 `step-5.safetensors`，但训练到 runner 15 后长时间停在 generator turn 内部，未完成 runner 15 日志，也未生成 step-5。
- 处理：
  - 因 DMD-only 语义证据已充分，继续占 16 卡收益低，已停止该对照；
  - 停止方式为对明确 tmux session `d44_dmdonly16` 发送 Ctrl-C；
  - 未使用 `pkill` 或模糊 kill；
  - 两台 8 卡机器 `bfs6vaz4d6` / `i6hf4scd4y` 已重新启动 `occupy_all`，最后检查 16 张卡均约 `166604 MiB`、`100%`。

## 实验 D44-4：Stage1 v5.3.5 同机同卡 backward timing 对照

目的：解释 fake backward 是否“本来就像 Stage1 一样慢”。

设置：

- 在同一 16 卡机器上运行 Stage1 v5.3.5/USMGT 或最接近的 Stage1 smoke timing。
- 使用同样 89f、dense/full-attention、LoRA rank 384、lq_proj_in trainable。
- 记录单步 backward 与 optimizer timing。

判定：

- 若 Stage1 backward 同量级很慢：D4.4 fake 慢主要是 Stage1-like full-attention fake backward 成本。
- 若 Stage1 明显快：D4.4 的 dual Accelerator/dual DeepSpeed 或 fake loss graph 有额外等待，应继续 profile。

当前状态：待执行。

2026-05-18 20:30 更新：暂缓。

- 原因：用户当前更关心代码正确性，而不是进一步解释 fake 慢点；
- 48 卡正式 D4.4 当前速度已经进入约 60s/step 量级，fake backward/sync 虽然仍波动，但不再是必须立刻用卡定位的阻塞项。

## 实验 D44-5：fake 参数/同步缩减消融

目的：定位 570M fake trainable 参数中，LoRA 与 `lq_proj_in` 哪部分主导 ZeRO2 sync/显存/速度。

设置：

- 仅用于诊断，不作为正式训练目标。
- 16 卡短跑两组：
  - A：fake 只训 LoRA，冻结 fake `lq_proj_in`。
  - B：fake 只训 `lq_proj_in`，冻结 fake LoRA。
- 每组只跑 1-2 个 runner step。

记录：

- fake trainable params 分组。
- `fake_backward_sync`。
- fake grad norm。
- 显存。

判定：

- 如果冻结 `lq_proj_in` 后 fake_backward_sync 大幅下降，说明 projector 参数/梯度同步是主要负担。
- 如果冻结 LoRA 后大幅下降，说明 LoRA rank 384 的 q/k/v/o 参数同步是主要负担。
- 如果两者都仍慢，主要瓶颈更可能是 dense_full fake backward compute/activation。

当前状态：待执行。

2026-05-18 20:30 更新：暂缓。

- 原因同 D44-4：当前优先级从性能 profile 转为代码语义正确性。
- 保留该实验作为后续如果 fake backward/sync 再次成为阻塞时的定位手段。

## 实验 D44-6：DMD 与 pixel 梯度冲突检查

目的：解释 Stage3 结果出现残影时，是否 DMD 方向与 pixel/recon 方向冲突。

设置：

- 固定同一 batch。
- 分别 backward pixel/recon 与 DMD student。
- 计算 student trainable params 上的：
  - `||grad_pixel||`
  - `||grad_dmd||`
  - `cos(grad_pixel, grad_dmd)`
  - `||grad_dmd|| / ||grad_pixel||`

判定：

- `grad_dmd` 极小：DMD 基本没起作用。
- `grad_dmd` 极大：DMD 可能压过重建目标。
- cosine 长期明显为负：DMD 和 pixel 目标冲突，可能对应残影/鬼影。

当前状态：待执行。

2026-05-18 20:30 更新：取消当前优先级。

- 用户反馈 D6 残影已经消失，因此不再优先做 DMD/pixel 梯度冲突实验。
- 若后续新版本再次出现 ghost/残影，再恢复该实验。

## 运行备忘：DMD 尖峰处理

2026-05-19 18:05 记录：

- 当前 D4.4 clean2 resume 使用配置中的 `stage3_dmd_grad_norm_max: 5.0` 与 `stage3_dmd_spike_policy: "skip"`。
- 线上 W&B 截图中 `train/dmd_student_loss` 早期出现尖峰，但对应 `dmd_grad_norm` 仍低于 5，因此日志中 `dmd_skip=0`，没有触发 skip。
- 下次若重启/新开 D4.4 对照，可考虑把 DMD 尖峰策略改为更平滑的 `stage3_dmd_spike_policy: "clamp"`，并将 `stage3_dmd_grad_norm_max` 试为 `3.0`。这只是候选设置，当前训练先不改。

## 2026-05-20 补充：D44-5 变成阻塞项，中间方案曾考虑冻结 fake lq_proj_in

clean2 run 在生成 validation 之前再次退出：

- run：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260519_v7d44_resume_clean2_from_step100`
- 最后一条训练日志：
  `runner_step=743, step=149`
- 失败位置：
  `fake_accelerator.backward(fake_loss)` 里的 DeepSpeed/ZeRO fake-side all-reduce。
- 关键报错：
  `OpType=ALLREDUCE, NumelIn=287838720, Timeout(ms)=600000`。
- 本次没有发现 `SlowDown/OSError/DataLoader worker`，和 2026-05-18 的 conductor 数据读取失败不同。

新结论：

- D44-5 原本是性能 profile 实验，现在已经成为稳定性阻塞项。
- D4.4 原始 fake trainable 组为：
  - `lora`: `283115520`
  - `lq_proj_in`: `287845888`
- timeout 的 all-reduce 大小与 fake `lq_proj_in` 组基本一致。

曾考虑的中间修复：

- `train_flashvsr_stage3_v7_d4_4_lora.py` 新增 `stage3_fake_train_lq_proj_in`；
- 中间方案是把 `G_fake` 改成 LoRA-only critic：
  - student/generator 仍训练 LoRA + `lq_proj_in`；
  - `G_fake` 仍每个 runner step 更新；
  - `G_fake` FM loss 仍用当前 `z_pred.detach()`；
  - 但 `G_fake` 只训练 LoRA，不再同步 fake 侧 `lq_proj_in`。

最终更正：

- 用户不接受把 fake 侧 `lq_proj_in` 永久关掉作为稳定性修复。
- 当前最终方案不是 LoRA-only fake，而是：
  - `stage3_fake_train_lq_proj_in: true`
  - `stage3_fake_lq_proj_update_every_n_runner_steps: 5`
  - LoRA 每个 fake step 更新；
  - fake `lq_proj_in` 每 5 个 runner step 更新一次；
  - fake FM loss 仍使用当前 `z_pred.detach()`，不回传 student。
- 同时保留 fake 侧 DeepSpeed bucket 下调与 activation checkpointing：
  `wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload_fake_stable.json`
- 48 卡由于 a9 GPU3 不可用、bfs 不在同一卡群，改为 5 节点 40 卡 fresh：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`

40 卡启动验证结果：

- 已确认启动日志：
  - `distributed_shape=5node40gpu`
  - `fake_lq_proj_trainable=True`
  - `fake_lq_proj_update_every_runner_steps=5`
  - `fake_ds_reduce_bucket=50000000`
  - `fake_ds_allgather_bucket=50000000`
- 已确认分频语义：
  - `runner_step=0`：`generator_update=1` 且 `fake_lq_proj_update=1`
  - `runner_step=1`：`generator_update=0` 且 `fake_lq_proj_update=0`
  - `runner_step=2`：`generator_update=0` 且 `fake_lq_proj_update=0`
- W&B：
  - run id：`v7d44_40g_freq5_1`
  - t5 远端同步 tmux：`wandb_sync_v7d44_40g_freq5`
  - 已确认第一次 `wandb sync --include-offline` 成功。

下一步验证优先级：

1. 40 卡 fresh 启动后先确认：
   - `distributed_shape=5node40gpu`
   - `fake_lq_proj_trainable=True`
   - `fake_lq_proj_update_every_runner_steps=5`
   - 至少跑过一个 fake update 和一个 generator update。
2. 如果 40 卡仍在 fake backward 超时，再继续 D44-5 profile LoRA/projector 同步成本。
3. 如果 40 卡稳定，则把 clean2 问题归因到 48 卡下 fake 侧 projector 大同步过重与 a9 节点异常共同放大风险。
