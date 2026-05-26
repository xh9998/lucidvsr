# FlashVSR Stage3 Loss 验证计划 2026-05-17

## 1. 目标与边界

这个文档只记录 Stage3 `v7-D4.x` 的 loss 正确性验证。目标不是先找最好看的 checkpoint，而是先证明：

- 每个 loss 更新的参数对象是对的。
- fake / generator 的交替更新没有把梯度混在一起。
- DMD、pixel、LPIPS、flow 单独打开时行为可解释。
- 现在看到的残影/拖影到底更可能来自哪一类训练信号。

稳定训练主线不允许塞临时 debug 逻辑。所有 GradCheck / ablation / probe 都从稳定源码复制出专用文件运行。

稳定基线：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
- `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_turnisolated_dfake5_offlinewandb.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-TurnIsolated-Dfake5-OfflineWandb.sh`

验证专用线：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_gradcheck_lora.py`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-1.sh`

从 `v7-D4.2` 开始，Stage1 teacher / G_real / G_fake 默认使用新的 USMGT Takano 微调模型：

- BFS 本地路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- S3 中转：`s3://lxh/tmp/usmgt_stage1_takano20250205_step3000_20260517/step-3000.safetensors`

## 2. D4.1 / D4.2 的训练语义

`v7-D4.1` 把 `v7-D3.2` 里 “student loss + fake loss 同一轮合起来 backward” 改成 turn-isolated：

| turn | runner step | forward | backward | optimizer |
|---|---:|---|---|---|
| generator/student | `runner_step % 6 == 0` | student 正常 forward | `student_loss + dmd_student_loss` | 只更新 student |
| fake-only | 其他 step | student `torch.no_grad()` 生成当前 fake 样本，`G_fake` 吃 `z_pred.detach()` | `fake_loss.backward()` | 只更新 `G_fake` |

`stage3_dfake_gen_update_ratio: 5` 表示：

`generator, fake, fake, fake, fake, fake, generator, ...`

`v7-D4.2` 不改训练结构，只把 Stage1 teacher / G_real / G_fake 初始化换成新的 USMGT Takano `step-3000`。

## 3. 梯度归属验证

DeepSpeed ZeRO2 下直接读 wrapped student 的 `param.grad` 可能不可靠，所以验证以 optimizer step 后的参数 checksum delta 为主，grad norm 只作为辅助。

验证专用代码通过环境变量打开日志：

`FLASHVSR_STAGE3_GRAD_OWNERSHIP_DEBUG=1`

期望日志：

- `[stage3_grad_ownership]`：辅助看 grad norm。
- `[stage3_param_ownership]`：核心，看 `student_delta / fake_delta / real_delta`。

期望参数归属：

| loss | 应该更新 | 不应该更新 |
|---|---|---|
| flow / MSE / LPIPS | student LoRA + student LQ projector | G_real, G_fake, VAE, LPIPS |
| DMD student | student one-step 输出路径 | G_real, G_fake |
| fake FM | trainable G_fake | student, G_real |

## 4. Grad-A 到 Grad-E

短跑 16 卡验证，每组只跑到能覆盖对应 turn。

| ID | 打开的 loss | 覆盖目的 |
|---|---|---|
| Grad-A | flow only | 证明 flow 只更新 student |
| Grad-B | pixel + LPIPS only | 证明 decode 重建 loss 只更新 student，首帧 4x 生效 |
| Grad-C | DMD student only | 证明 DMD student loss 只更新 student，不更新 teacher/fake |
| Grad-D | fake FM only | 证明 fake turn 只更新 G_fake，student no-grad 只负责造 fake 样本 |
| Grad-E | full D4.2 | 证明 full objective 下 generator/fake turn 分离正确 |

CLI 覆盖：

| ID | overrides |
|---|---|
| Grad-A | `--stage3_flow_weight 1 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0` |
| Grad-B | `--stage3_flow_weight 0 --stage3_mse_weight 1 --stage3_lpips_weight 2 --stage3_dmd_weight 0 --stage3_fake_fm_weight 0` |
| Grad-C | `--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 1 --stage3_fake_fm_weight 0` |
| Grad-D | `--stage3_flow_weight 0 --stage3_mse_weight 0 --stage3_lpips_weight 0 --stage3_dmd_weight 0 --stage3_fake_fm_weight 1` |
| Grad-E | `--stage3_flow_weight 1 --stage3_mse_weight 1 --stage3_lpips_weight 2 --stage3_dmd_weight 1 --stage3_fake_fm_weight 1` |

## 5. Loss Ablation 视觉验证

Grad-A 到 Grad-E 通过后，再做短训视觉验证。测试集固定用同一组 10 个 synthetic 视频，输出统一下载到桌面，拆帧检查。

| ID | 目的 |
|---|---|
| Loss-1 Flow only | one-step FM baseline，检查不用 DMD 时是否已经有残影 |
| Loss-2 Pixel only | 检查 GT 对齐、首帧 4x、尾帧监督是否合理 |
| Loss-3 Pixel + LPIPS | 检查 LPIPS 是否增加细节，也是否引入时序不稳 |
| Loss-4 DMD only | 单独观察 DMD direction 是否导致鬼影/拖影 |
| Loss-5 Fake only | fake critic 不应该直接改变 student 输出，用于确认调度 |
| Loss-6 DMD + Fake | 核心 DMD loop，不含重建 loss |
| Loss-7 Full D4.2 | 当前正式目标，对比所有 ablation |

## 6. Ghost / 残影专项计划

现在用户反馈 Stage3 输出动起来有残影。这个问题不能只靠看 full run，需要拆开定位。

固定对比规则：

- 同一组 10 个视频。
- 同一套 89f Stage1/Stage2/Stage3 推理接口。
- 输出拆帧，重点看运动边缘、手/头发/字幕/纹理边界。
- 对 `takano04` 等重点视频额外做相邻帧差分热图。

专项实验：

| ID | 设置 | 要回答的问题 |
|---|---|---|
| Ghost-0 | Stage2 6.4.1 teacher/student baseline | 二阶段本身有没有残影 |
| Ghost-1 | Full D4.2 | 当前完整 Stage3 的残影程度 |
| Ghost-2 | Flow + Pixel + LPIPS，不开 DMD/Fake | 残影是否来自重建分支或 one-step 本身 |
| Ghost-3 | DMD + Fake，不开 Pixel/LPIPS | 残影是否主要来自 DMD 对抗方向 |
| Ghost-4 | DMD only | `G_real - G_fake` 方向单独是否稳定 |
| Ghost-5 | Pixel/LPIPS only | 强监督能否压住残影 |
| Ghost-6 | full D4.2 但 dense attention probe | 排除 block-sparse / chunk mask 带来的时序断裂 |
| Ghost-7 | 内部 validation vs 外部 batch inference | 排除 validation/inference 路径不一致 |

判断标准：

- 如果 Ghost-0 干净、Ghost-2 干净、Ghost-3/4 明显残影，则优先怀疑 DMD/fake 训练方向或权重。
- 如果 Ghost-2 也残影，优先查 one-step student、pixel GT 对齐、LPIPS 对时序的影响。
- 如果只有外部 inference 残影，优先查推理路径是否仍和 Stage2 streaming / cache 语义一致。

## 7. 已知结果

### Grad-A Attempt 1

目录：

`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradcheck_16gpu_v7_d4_1_Grad-A_20260517_gradA_003`

结论：

- loss 设置正确：`flow=0.107508`，`mse=0`，`lpips=0`，`fake_fm_weight=0`，`dmd_weight=0`。
- `need_reconstruction=False`，说明 pixel/LPIPS decode 没有进入。
- `stage3c_train` 显示 `turn=generator`，`fake_update=0`，`dmd_probe=0`。
- 但 ZeRO2 下直接读到的 `student_grad_norm=0` 不足以证明没有梯度，因此这个结果不作为最终归属结论。

后续修正：

- 稳定 `v7-D4.1` 源码保持不动。
- 只在 `train_flashvsr_stage3_v7_d4_1_gradcheck_lora.py` 中加入 checksum ownership 日志。
- 重新跑 Grad-A 到 Grad-E 时必须使用 `v7-D4.2` 默认的 USMGT Stage1 `step-3000`。

### Ghost Probe 1: Stage3 D3.2 step-1000 vs Stage2 6.4.1

时间：2026-05-17

输出目录：

- 本地：`/Users/lixiaohui/Desktop/stage3_ghost_probe_20260517/takano04`
- S3：`s3://lxh/artifacts/inference/stage3_ghost_probe_20260517/takano04`

测试视频：

- `takano_04_lq.mp4`

对比项：

| 输出 | 目的 | 结论 |
|---|---|---|
| `stage2_v641_step6000_sparse_adain.mp4` | 二阶段 6.4.1 baseline | 作为正常流式 one-step 前的参考 |
| `stage3_d32_step1000_sparse_adain.mp4` | 当前可取得的 Stage3 D3.2 step-1000 | 用于复现残影/ghost |
| `stage3_d32_step1000_sparse_nocolor.mp4` | 去掉 color fix | 和 adain 版本只有轻微差异，color fix 不是主因 |
| `stage3_d32_step1000_dense_adain.mp4` | 试图测试 dense attention | 无效对照：结果和 sparse 字节完全相同 |
| `stage3_d32_step1000_officialmask_adain.mp4` | 试图测试 official mask | 无效对照：结果和 sparse 字节完全相同 |

关键发现：

- `sparse_adain`、`dense_adain`、`officialmask_adain` 三个 mp4 和关键帧 hash 完全一致。
- 原因不是模型真的对 attention/mask 不敏感，而是当前 `infer_from_lq_streaming()` 固定调用 `stage2_streaming_block_forward()`；该 streaming cache 路径不读取 `stage2_attention_mode`，因此 `dense_full` / `block_sparse_official_mask` 参数没有生效。
- `sparse_nocolor` 和 `sparse_adain` 的相邻帧变化指标接近，说明 color fix 不是这次 ghost 的主要来源。
- 相邻帧最大变化出现在 `21->22`、`29->30`、`53->54` 等位置，和用户观察到的 chunk 边界跳变位置一致；但因为 dense/officialmask 对照无效，目前还不能把原因归结为 block sparse/mask。

下一步：

- 若要验证 attention/mask 是否导致 ghost，必须写一个真正不走 streaming cache 的 full-DiT / dense probe，或在 streaming 函数内部显式实现可切换 attention 路径。
- 更优先的验证仍是 loss ablation：先区分 ghost 是否来自 Stage3 权重/训练目标（DMD/Fake、Pixel/LPIPS、one-step distillation），再查 inference mask。
