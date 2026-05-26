# FlashVSR Stage3 显存与作者实现对齐计划

日期：2026-05-14

本文档记录 Stage3 `v7-B` 后续排查和实现顺序。目标不是先把代码“勉强跑通”，而是先确认训练语义与 FlashVSR 作者方案一致，再定位为什么当前实现显存明显高于作者描述的 80GB 级别。

## 1. 当前判断

当前问题不是单一的 `pixel / LPIPS loss` 显存过大，而是需要同时确认两件事：

1. Stage1 / Stage2 训练底座是否已经比作者实现更重。
2. Stage3 的 Wan decoder reconstruction 分支是否按作者语义实现，而不是用自定义 tile-level decode 绕开问题。

如果 Stage2 flow-only 底座已经接近 140GB，Stage3 再叠加 Wan decoder backward 后 OOM 是必然结果。此时继续堆 offload 或 tile 只能临时绕过，不能说明实现对齐。

## 2. 必须保留的 Stage3 语义

Stage3 `v7-B` 暂时不接入完整 DMD dual optimizer，也不接入 `G_real / G_fake` 交替更新。当前只做 one-step student + random latent reconstruction 分支。

必须保留：

- CPU 在线退化。
- `stage3_recon_num_latents=2`。
- 每个 step 随机选择连续 2 个 latent。
- prefix no-grad / selected grad。
- pixel / LPIPS 只监督选中的 2 个 latent 对应帧。
- 如果选中全局首帧，pixel 和 LPIPS 首帧权重都乘 4。
- validation 使用 one-step student，不再沿用 Stage1/2 的 50-step validation。
- 当前不引入 `G_real / G_fake`，避免在单 optimizer runner 里硬塞错误 DMD。

作者语义固定为：

```text
随机选中 [recon_start, recon_end)

Wan decoder forward 语义:
  decode [0, recon_end)

梯度语义:
  [0, recon_start) 只 forward，detach / no-grad
  [recon_start, recon_end) 带梯度

loss:
  只对 [recon_start, recon_end) 对应的 decoded frames 算 pixel / LPIPS
```

## 3. 暂时撤掉 tile-level training decode 主线

当前决策：

- 不把自定义 spatial tile-level training decode 作为 Stage3 主线。
- 不用 tile 先逃避显存问题。
- 保留 Wan decoder causal cache 串行 decode 语义。

原因：

- FlashVSR Stage3 的关键是 causal decoder prefix + selected latent backward。
- spatial tile-level training decode 会引入额外不确定性：tile 边界、空间融合、causal decoder cache 与完整帧是否严格等价。
- 如果 full-frame selected decode OOM，应优先定位是 DiT 底座、VAE decoder activation、LPIPS activation、还是 graph 生命周期问题，而不是先用 tile 把问题藏起来。

主线实现应为：

```text
clear Wan decoder cache

if recon_start > 0:
  no_grad:
    decode prefix [0, recon_start)
    只推进 causal decoder cache

grad:
  decode selected [recon_start, recon_end)
  full-frame output

pixel / LPIPS:
  对 selected decoded frames 串行计算
```

必要时只给 selected decode 分支加：

```text
torch.autograd.graph.save_on_cpu()
torch.utils.checkpoint.checkpoint(..., use_reentrant=False)
```

## 4. 先做 Stage1/2 底座显存排查

作者描述里 Stage3 在 H100 80GB 级别可以训练，而当前实现 Stage1/2 峰值已经明显高。需要先确认底座是否对齐。

### 4.1 block sparse attention

要查：

- Stage2 / Stage3 是否真的走 block sparse CUDA kernel。
- 是否有条件不满足时静默走 dense fallback。
- 当前 top-k / mask / chunk 逻辑是否导致实际 attention 图过大。
- 是否仍有 debug / fallback path 在训练里启用。

验收：

- 在训练日志里打印一次 attention backend。
- 明确写出每个 DiT block self-attn 实际走的函数。
- 如果走 fallback，必须修正或显式标记该实验不是作者对齐版本。

### 4.2 gradient checkpointing

要查：

- DiT block 是否真正被 gradient checkpoint 覆盖。
- `use_gradient_checkpointing=true` 是否只设置了 flag，但部分 block 没有生效。
- DeepSpeed activation checkpointing 是否没有开启，但模型级 checkpoint 是否已足够。
- `use_gradient_checkpointing_offload` 是否只是 smoke 临时方案，不应默认进入正式设置。

验收：

- 训练启动时打印一次每个关键模块的 checkpoint 状态。
- 只保留低频 summary，不刷屏。

### 4.3 trainable 参数

要查：

- LoRA / `lq_proj_in` 以外是否还有不该训练的参数 `requires_grad=True`。
- Wan VAE / text encoder / frozen DiT base 是否被误挂入 optimizer。
- LPIPS 是否被注册进 DeepSpeed model tree。

验收：

- 打印 trainable parameter summary。
- 打印 optimizer parameter group 数量和参数量。
- LPIPS 必须作为 loss-only cache，不进入 DeepSpeed checkpoint state。

### 4.4 validation / debug / extra branch

要查：

- validation 是否在训练前或训练中持有不必要的模型 graph。
- debug tensor / snapshot 是否保留 GPU tensor 引用。
- Stage3 loss metadata 是否只存 float / int，不存 tensor。

验收：

- loss metadata 只允许 CPU scalar。
- validation 不参与训练图。

### 4.5 数据退化

必须保持：

- 在线退化仍然启用。
- 退化在 CPU 上执行，不在 DataLoader worker 中创建 CUDA context。

目的：

- worker 数量不再和 GPU 显存峰值耦合。
- 避免 GPU0 或 local rank 相关显存异常跳高。

验收：

- DataLoader worker 中不调用 CUDA。
- `dataset_num_workers=2` 时不因为退化导致显存飙升。

## 5. Stage3 显存分解实验

在作者语义固定、无 tile 主线固定后，逐步加回 Stage3 loss，量化每一项显存增量。

### Probe 1: Stage2 flow only

目的：

- 测 DiT / scheduler / block sparse / LoRA / projector 的底座显存。

设置：

- 用 Stage2 v6.4.1 同样输入规则。
- 不 decode。
- 不算 MSE / LPIPS。

结论用途：

- 如果这个已经远高于 80GB，优先修 attention / checkpoint / trainable 参数。

### Probe 2: one-step z_pred only

目的：

- 确认 one-step student `scheduler.step(..., to_final=True)` 本身是否额外留图过多。

设置：

- 有 `z_pred`。
- 不进 Wan decoder。

### Probe 3: MSE decode only

目的：

- 测 Wan decoder full-frame selected decode 的显存增量。

设置：

- prefix no-grad cache。
- selected 2 latent full-frame decode。
- 只算 MSE。
- 不算 LPIPS。

### Probe 4: MSE + LPIPS

目的：

- 测 LPIPS / VGG activation 增量。

设置：

- LPIPS 逐帧串行。
- LPIPS 首帧权重跟 pixel 一致，抽到全局首帧时乘 4。

### Probe 5: 小分辨率语义 smoke

目的：

- 避开显存问题，验证 random latent decode 语义正确。

设置：

- 低分辨率。
- full-frame selected decode。
- `stage3_recon_num_latents=2` 不降级。

验收：

- `latent_window` 多 step 变化。
- `decoded_frames` 正确。
- `first_frame_pixel_weight / first_frame_lpips_weight` 只在抽到全局首帧时为 4。
- forward / backward / checkpoint / training_state 都正常。

## 6. 当前代码方向

当前本地代码已经开始朝这个方向调整：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py
```

主线 decode 应使用：

```text
_stage3_decode_selected_window_full_frame(...)
_stage3_decode_selected_with_checkpoint(...)
```

不再使用：

```text
_stage3_tiled_decode_selected_window(...)
_stage3_decode_one_tile_selected(...)
```

如果后续为了显存重新引入 tile，必须作为单独实验分支记录，不能混入作者对齐主线。

## 7. 执行顺序

推荐顺序：

1. 固定无 tile 的 Stage3 full-frame selected decode 代码。
2. 做小分辨率 correctness smoke，确认语义。
3. 做 Stage2 flow-only 显存基线。
4. 做 one-step z_pred only 显存基线。
5. 做 MSE decode only。
6. 做 MSE + LPIPS。
7. 如果 MSE decode only 已 OOM，优先查 Wan decoder checkpoint / offload / graph 生命周期。
8. 如果 Stage2 flow-only 已过高，优先查 block sparse kernel / GC / trainable 参数。
9. 所有结论补回本文档和 `FLASHVSR_WORKLOG.md`。

## 7.1 CPU 退化后的 DataLoader worker sweep

当前 `v7-B 641data smoke` 配置：

```yaml
dataset_num_workers: 2
dataloader_prefetch_factor: 1
dataloader_persistent_workers: true
dataloader_in_order: false
dataloader_multiprocessing_context: spawn
```

这里的 `dataset_num_workers=2` 是每个 rank / 每张卡 2 个 worker。2GPU smoke 等于总计 4 个 worker；48GPU 正式训练如果也设 2，则总计约 96 个 worker。

CPU 退化后，需要重新测试 worker。原因：

- 过去 worker 开大容易让退化在 GPU 上创建 CUDA context，导致显存爆；
- 当前退化改到 CPU 后，worker 理论上可以开大；
- 但 worker 太多也可能造成 CPU / conductor / tar 解码争用，导致 step time 反而变慢。

worker sweep 不应该只看能不能跑，要同时看：

| 指标 | 目的 |
|---|---|
| first batch latency | 判断数据初始化是否变慢 |
| step time | 判断训练真实吞吐 |
| GPU utilization | 判断是否仍然数据饥饿 |
| CPU load / worker 数 | 判断 CPU 是否过载 |
| GPU memory peak | 确认 CPU 退化后 worker 不再推高 GPU 显存 |
| conductor/cache QPS | 判断是否打爆远端 IO |

计划测试：

```text
dataset_num_workers = 1 / 2 / 4 / 8
```

先在 2GPU smoke 上测每个配置至少 5 step：

```text
记录平均 step time
记录 0/1 卡 GPU util
记录显存峰值
记录是否出现 DataLoader worker error
```

如果 2GPU 上 `4/8` 稳定，再把候选值带到多卡训练；如果 2GPU 上已经 CPU/IO 变慢，就不要盲目放大到 48GPU。

当前结论见第 10.2 节。实际扫描后更新为：

- `worker=2` 是当前稳妥默认；
- `worker=4` 没有明确速度收益；
- `worker=8` 的 OOM 根因已经定位：DataLoader worker init 里调用了 CUDA，导致每个 worker 创建 CUDA context；
- 修复 worker CUDA context 后，`worker=8` 可以跑通，但在 2GPU smoke 上仍没有稳定快于 `worker=2`；
- 因此正式默认暂时仍建议 `worker=2`，后续要做更长吞吐测试后再决定是否提高。

## 7.2 Offload 范围核对

当前 smoke 并不是“只有 decoder offload”。

当前配置实际包含三类 offload：

| offload | 当前状态 | 说明 |
|---|---|---|
| DeepSpeed optimizer offload | 开启 | ZeRO2 config 中 optimizer 在 CPU |
| DeepSpeed param offload | 开启 | ZeRO2 config 中 param 在 CPU |
| model gradient checkpoint offload | 当前 smoke 开启 | `use_gradient_checkpointing_offload: true`，这是为了先验证链路，不应默认视为作者最终设置 |
| Stage3 decoder activation offload | 开启 | `stage3_decoder_cpu_offload: true`，这是作者建议优先保留的 decode offload |

作者口径是“Stage3 只需要给 decode 开 offload”。因此后续需要做一个 offload ablation：

```text
A. 当前 smoke 设置：
   use_gradient_checkpointing_offload=true
   stage3_decoder_cpu_offload=true

B. 作者对齐设置：
   use_gradient_checkpointing_offload=false
   stage3_decoder_cpu_offload=true

C. 极限对照：
   use_gradient_checkpointing_offload=false
   stage3_decoder_cpu_offload=false
```

判断标准：

- 如果 B 能稳定跑，正式 Stage3 优先用 B；
- 如果 B OOM，但 A 能跑，说明当前 DiT/decoder 底座仍比作者重，需要继续查 attention / GC / trainable 参数；
- C 只是判断 decoder offload 是否必要，不作为正式配置。

## 8. 暂不做的事

暂不做：

- 不接入 `G_real / G_fake`。
- 不写 dual optimizer runner。
- 不把 tile-level decode 当主线。
- 不用降低 `stage3_recon_num_latents` 到 1 作为正式验收。
- 不用关 pixel / LPIPS 来宣称 Stage3 成功。

## 9. Stage3 后续大模块的串行执行原则

Stage3 后续还要加入 `G_real / G_fake / DMD`，不能把所有模型和 loss 都塞进同一个大计算图。核心原则是：

```text
只有 student one-step 生成路径需要保留到 backward。
所有 teacher / reference / frozen scorer / prefix cache 都必须 no-grad 或 detach。
需要更新的模型必须有自己的 optimizer 和更新阶段。
```

### 9.1 模块职责表

| 模块 | 是否训练 | 是否需要梯度回传到 student | 推荐执行方式 |
|---|---:|---:|---|
| Student DiT + LoRA + projector | 是 | 是 | 正常训练图 |
| Wan decoder pixel / LPIPS 分支 | decoder 不训 | 需要对 `z_pred` 回传 | prefix no-grad cache；selected 2 latent full-frame decode |
| LPIPS / VGG | 不训 | 需要对 decoded frame 回传 | freeze；逐帧串行；checkpoint / offload |
| `G_real` | 不训 | 可能需要给 student 梯度 | `eval()` + `requires_grad_(False)`；不更新自身参数 |
| `G_fake` | 是 | 单独训练 | 独立 optimizer；alternating update |
| Teacher / reference model | 不训 | 通常不需要 | `torch.no_grad()` 推理；输出 detach |
| 数据退化 | 不训 | 不需要 | CPU worker 在线做；不创建 CUDA graph |

### 9.2 为什么不能一次性全塞进一个 loss

如果把 student、Wan decoder、LPIPS、`G_real`、`G_fake` 全部放进同一个 forward / backward：

- frozen 模块也可能保留 activation；
- teacher / scorer 的中间 tensor 会和 student graph 叠加；
- `G_fake` 如果也更新，会和 student optimizer 混在一起；
- DeepSpeed 当前 runner 只有一个 optimizer，不能正确表达 DMD2 的 alternating update；
- 显存峰值会随模块数线性或更差地叠加。

所以完整 DMD 不能硬塞进当前 `launch_training_task` 的单 optimizer runner。需要新写 `v7-C` 或独立 Stage3 runner。

### 9.3 推荐的完整 Stage3 执行顺序

后续完整 Stage3 应拆成下面几个阶段，而不是一个 monolithic loss：

```text
1. student one-step forward
   - 生成 z_pred
   - 只保留 student graph

2. flow / FM loss
   - 可直接对 student backward
   - 或与后续 student loss 合并，但不能保留无关 teacher graph

3. selected reconstruction loss
   - Wan decoder frozen
   - prefix [0:recon_start) no-grad 推进 causal cache
   - selected [recon_start:recon_end) 带 grad full-frame decode
   - pixel / LPIPS 逐帧串行算
   - loss 只回传 student

4. DMD student loss
   - G_real / G_fake 参数 freeze
   - scorer 只提供对 student output 的梯度
   - 不更新 G_real / G_fake

5. G_fake update
   - freeze student
   - freeze G_real
   - 只更新 G_fake
   - 使用独立 optimizer
```

### 9.4 当前 `v7-B` smoke 的范围

当前 smoke 只验证下面部分：

- CPU degradation；
- student one-step；
- random 2 latent；
- prefix no-grad cache；
- selected 2 latent full-frame decode；
- pixel / LPIPS；
- first-frame weight；
- checkpoint / training_state。

当前 smoke 不验证：

- `G_real`；
- `G_fake`；
- DMD dual optimizer；
- alternating update。

如果当前 `v7-B` 无 tile full-frame selected decode 仍 OOM，下一步应优先按第 5 节拆显存，而不是马上恢复 tile。

## 10. 2026-05-14 smoke 结果和 worker/offload 结论

### 10.1 `v7-B` 无 tile full-frame selected decode smoke

远端机器：`6ai5mpi47f`

代码：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py`

输入 checkpoint：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-3300.safetensors`

关键设置：

- `num_frames=89`
- `batch_size=1`
- `stage3_recon_num_latents=2`
- `stage3_decoder_cpu_offload=true`
- `stage3_first_frame_pixel_weight=4.0`
- `stage3_first_frame_lpips_weight=4.0`
- CPU 在线退化
- 不使用自定义 spatial tile training decode

通过的 smoke：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_v7b_fullframe_notile_contigfix`
- 结果：跑到 `step=20`，保存 `step-1/2/5/10/20.safetensors`。

结论：

- `prefix no-grad cache + selected 2 latent full-frame decode + pixel/LPIPS` 这条主线可以跑通。
- 显存不再立即 180GB OOM，2 卡 smoke 中主进程显存约 120GB 量级。
- 日志中 `first_frame_*_weight=1.0` 是因为随机 window 没抽到全局首帧；只有 `frame_start==0` 时才会变成 4.0。

### 10.2 worker 扫描

同一套 `v7-B` smoke，固定 `batch_size=1`，只改 `dataset_num_workers`。

| workers | 结果 | 典型耗时 | 结论 |
|---:|---|---|---|
| 1 | 跑了 2 step 后停止 | `step1≈168s`, `step2≈120s` | 明显慢，不继续作为候选 |
| 2 | 正常跑完 4 step | `step1≈135s`，后续有效 step 约 `31-48s` | 当前默认候选 |
| 4 | 正常保存到 `step-4`，结束时有非关键 noisy terminate | `step1≈148s`，后续与 worker2 接近或略慢 | 没有收益，不建议默认 |
| 8 | 初次扫描 OOM；修复 worker CUDA context 后可跑通 | 修复后 `step1≈164s`，后续有效 step 约 `29-42s` | 可用但不稳定快于 worker2 |

worker CUDA context 根因：

- `diffsynth/diffusion/runner.py` 里 DataLoader `worker_init_fn` 原本会调用 CUDA 相关逻辑；
- 多 worker 时，每个 worker 进程都会在 GPU 上创建约 610MB CUDA context；
- `dataset_num_workers=8` 时，每个 rank 会额外出现 8 个 worker CUDA context；
- `nvidia-smi --query-compute-apps` 能直接看到这些 614MB 左右的 worker 进程；
- 这解释了“worker 开大以后显存随 worker 数波动”的现象，确实是数据侧有人碰了 GPU。

已修复：

- `runner.py` 新增 `_init_data_worker_no_cuda(...)`；
- DataLoader worker init 只设置 CPU 随机种子，不再调用 `torch.cuda.is_available()` / `torch.cuda.set_device()`；
- `launch_training_task(...)` 和 `launch_data_process_task(...)` 两条路径都改成使用 `_init_data_worker_no_cuda(...)`；
- `parquet_tar_dataset_v2.py` 的旧退化 wrapper 也改成强制 CPU degradation；
- `aliyun_video_degradation.py` / `realesrgan_kernels.py` 的 CUDA seed 只在 model device 真正是 CUDA 时才执行。

修复后验证：

- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_v7b_worker8_after_workerinit_fix`
- `worker=8` 跑到 `step=4`，没有 worker CUDA context OOM；
- `nvidia-smi --query-compute-apps` 中不再出现一串 614MB worker CUDA 进程；
- 但 `worker=8` 速度没有稳定优于 `worker=2`，当前默认仍保守用 `worker=2`。

当前结论：

- “worker 开大会导致显存波动”这个问题已经排查清楚，根因是 DataLoader worker CUDA context，不是退化算法本身必须占 GPU；
- 修复后 worker 数不再直接推高 GPU context 显存；
- Stage3 正式训练默认仍建议 `dataset_num_workers=2`，因为 2GPU smoke 下 `worker=8` 没有明确吞吐优势。

### 10.3 和 `v6.4.1` 的一步耗时对比

`v6.4.1` 正式训练日志来自：

- `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/run.log`

`v6.4.1` 设置：

- `40GPU`
- `batch_size=1`
- `dataset_num_workers=2`
- Stage2 flow loss
- block sparse attention
- 无 Stage3 pixel/LPIPS decode 分支

日志后段稳定速度：

- 大约 `14.2-15.0s/step`。

当前 `v7-B` 2GPU smoke：

- `batch_size=1`
- `dataset_num_workers=2`
- one-step student + selected 2 latent full-frame decode + pixel MSE + LPIPS
- `step1≈135s`
- 后续有效 step 约 `31-48s`

注意：

- 这不是严格 apples-to-apples：`v6.4.1` 是 40GPU 正式训练，`v7-B` 当前是 2GPU smoke。
- 但差距方向明确：Stage3 的 Wan decoder + LPIPS reconstruction 分支是新增大头。

### 10.4 GC offload 扫描

测试项：

- `dataset_num_workers=2`
- `stage3_decoder_cpu_offload=true`
- 对比 `use_gradient_checkpointing_offload=true/false`

结果：

| 设置 | 结果 | 耗时 | 显存 |
|---|---|---|---|
| GC offload on | 4 step 正常 | `step1≈135s`，后续有效 step 约 `31-48s` | 约 120GB |
| GC offload off | 4 step 正常 | `step1≈129s`，后续有效 step 约 `31-45s` | 约 140GB |

结论：

- 关 GC offload 只带来小幅速度收益，但确实是当前已完成 smoke 里最快的组合。
- 代价是显存从约 120GB 上升到约 140GB，余量明显变差。
- 如果目标是速度优先，后续优先试 `use_gradient_checkpointing_offload=false`；如果正式多卡 OOM 或启动不稳定，再回退 `use_gradient_checkpointing_offload=true`。

### 10.5 Stage3 显存分解 probe

为避免把所有 Stage3 分支混在一起误判，新增了 `stage3_compute_z_pred` 开关：

- `--no-stage3_compute_z_pred`：只算 flow，不执行 one-step `z_pred`，不进 Wan decoder；
- `--stage3_compute_z_pred --stage3_mse_weight 0 --stage3_lpips_weight 0`：执行 one-step `z_pred`，但不进 Wan decoder；
- `--stage3_compute_z_pred --stage3_mse_weight 1 --stage3_lpips_weight 0`：执行 one-step + Wan decoder selected-window + MSE；
- `--stage3_compute_z_pred --stage3_mse_weight 1 --stage3_lpips_weight 2`：完整 MSE + LPIPS。

Probe 结果：

| Probe | 实验目录 | 关键设置 | 观测显存 | 结论 |
|---|---|---|---|---|
| Probe1 flow-only | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe1_flow_only` | no z_pred, no decoder | 约 40/43GB | Stage2/flow 底座不是 120GB 主因 |
| Probe2 z_pred-only | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe2_zpred_only` | z_pred, no decoder | 约 41/43GB | one-step scheduler / z_pred 本身不是主因 |
| Probe3 MSE-only | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe3_mse_only` | z_pred + decoder + MSE | 约 119/121GB | Wan decoder selected-window backward 是显存大头 |
| Probe4 MSE+LPIPS | `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe4_mse_lpips_full` | z_pred + decoder + MSE + LPIPS | 约 119/121GB | LPIPS 不是额外显存大头，主要增加耗时和 loss |

关键结论：

- 当前 120GB 级峰值不是 DataLoader，也不是 one-step student；
- 主要来自 Wan decoder 对 selected latent window 的反传 activation；
- LPIPS 逐帧串行 + checkpoint 后，显存增量不明显；
- 后续若要接入 `G_real/G_fake`，必须继续保持串行执行和 no-grad/freeze，不应把所有分支同时挂在一个大 graph 里。

### 10.6 已完成 / 未完成项

已完成：

- CPU degradation 和 DataLoader worker 不再创建 CUDA context；
- `worker=8` CUDA context OOM 根因已定位并修复；
- 无 tile 的 full-frame selected decode 可以跑通；
- `stage3_first_frame_pixel_weight=4.0` 和 `stage3_first_frame_lpips_weight=4.0` 已在抽到首帧 window 时生效；
- 显存分解 probe 已明确 decoder backward 是 Stage3 峰值主因；
- LPIPS 已作为 loss-only cache，通过 `object.__setattr__` 避免注册进 DeepSpeed model tree。
- block sparse attention 已核对：
  - `train_flashvsr_stage3_v7_b_lora.py` 复用 `train_flashvsr_stage2_v6_4_lora.py`；
  - `train_flashvsr_stage2_v6_4_lora.py` 调用 `enable_stage2_causal_attention(...)`；
  - attention 实现来自 `diffsynth/models/wan_video_dit_stage2_v6_1.py`；
  - `block_sparse_chunk_causal_attention(...)` 内部调用 `block_sparse_attn_func(...)`；
  - 如果 CUDA extension 不可用，会直接 `RuntimeError("block_sparse_attn is unavailable...")`，不会静默 fallback 到 dense attention；
  - 远端 `flashvsr` 环境中先 import `torch` 后可正常 import `block_sparse_attn_func`。
- trainable 参数检查已核对：
  - Stage3 smoke 日志中 `Trainable parameter tensors: 488`；
  - `Trainable parameter count: 570961408`；
  - 当前训练对象仍是 LoRA + `lq_proj_in`，LPIPS 不进入 DeepSpeed model tree。

### 10.7 `v6.4.1 / 641` 的 89 帧规则重新确认

用户反馈 `641` 的测试效果较好，因此后续 Stage3 先对齐 `641`，不再强行把测试集预处理成 `85 + repeat4`。

代码事实如下：

```text
输入 LQ: 89 frames
LQ projector: 89 -> 22 latent chunks
GT / noise target: target_frames = 22 * 4 - 3 = 85 frames
Wan VAE target: 85 frames -> 22 latents
loss / inference output: 22 latents -> 85 effective frames
```

对应代码锚点：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
  - `target_frames = int(noise.shape[2]) * 4 - 3`
  - `input_video = input_video[:, :, :target_frames]`
- `infer_from_lq_streaming(...)`
  - `latent_frames = max(1, (num_frames - 1) // 4)`
  - `num_frames=89` 时 `latent_frames=22`

所以结论要精确表述为：

- `641` 接受正常 `89` 帧输入；
- 但它实际监督 / 输出的是 `85` 帧有效结果；
- 后续 `v7-B / v7-C` 先沿用这个规则；
- 不再默认要求外部测试集先构造成 `85 + repeat4`，除非是专门做 tail-padding ablation。

### 10.8 worker / offload 的当前速度结论

用户关心的是最快可用设置，而不是最保守设置。当前 smoke 结论如下：

| 设置 | 结果 | 结论 |
|---|---|---|
| `worker=2`, `GC offload on` | 稳定，约 120GB，后续有效 step 约 `31-48s` | 最稳 fallback |
| `worker=2`, `GC offload off` | 稳定，约 140GB，后续有效 step 约 `31-45s` | 当前最快已完成配置 |
| `worker=8`, `GC offload on` | 修复 worker CUDA context 后能跑，但 2GPU smoke 未稳定快于 `worker=2` | 不作为默认 |
| `worker=4/8`, `GC offload off` | 尝试后长时间停在 `0it`，未进入有效训练 | 不作为正式设置 |

因此当前推荐：

- 如果显存允许，Stage3 `v7-B` 后续优先试 `dataset_num_workers=2 + use_gradient_checkpointing_offload=false + stage3_decoder_cpu_offload=true`；
- 如果正式多卡出现 OOM 或启动不稳定，立即回退到 `dataset_num_workers=2 + use_gradient_checkpointing_offload=true`；
- 不再把 `worker=8` 当作默认加速方案，除非后续多卡长时间吞吐证明它确实更快。

### 10.9 B 阶段收口结论

`v7-B` 的目标是确认 one-step student + random latent reconstruction 是否能在不接 DMD dual optimizer 的情况下跑通，并分解显存来源。当前可以收口：

- 数据侧 GPU 显存波动已定位并修复；
- CPU 在线退化已接入，DataLoader worker 不再创建 CUDA context；
- full-frame selected decode 主线可跑通，不再依赖自定义 spatial tile；
- 首帧 pixel / LPIPS 权重已统一为 `4.0`；
- 显存大头明确是 Wan decoder selected-window backward；
- block sparse CUDA kernel 可用，训练路径没有 silent dense fallback；
- 641 规则确认：正常 89 帧输入，内部得到 22 latents，监督 / 输出 85 有效帧。

因此 B 阶段已经完成“可运行、可解释、可继续”的验收。下一步进入 `v7-C`：完整 DMD / dual-optimizer runner。

### 10.10 C 阶段前补充事项

进入 `v7-C` 前新增以下约束：

- `641` 效果较好，后续 Stage3 先按 `641` 路径：
  - 正常输入 `89` 帧；
  - 不默认外部构造 `85 + repeat4`；
  - 内部仍是 `89 -> 22 latent -> 85` 有效监督 / 输出。
- Stage2 pretrain 固定使用 `641 step-6000` checkpoint，不再沿用早期 `3k`，也不默认使用更高 step。
- 如果论文明确 `G_real` 是 Stage1/full-attention teacher，`v7-C` 就固定用 Stage1 teacher，不再做 Stage1/Stage2 teacher 摇摆。
- CPU degradation 迁移需要做固定参数 ablation，确认 CPU/GPU apply 的视觉结果没有不可接受差异。
- Wan decoder selected-window backward 是当前显存增量主因，但优化必须保持语义：
  - 不加 tile；
  - 不改变 prefix no-grad / selected grad；
  - 优先查 graph 生命周期、frozen model 是否误入 autograd、teacher/fake 是否能串行释放。

固定参数 CPU/GPU 退化 ablation 已执行：

- 脚本：`wanvideo/data/flashvsr/tests/compare_degradation_cpu_gpu.py`
- 远端输出：`/mnt/task_wrapper/user_output/artifacts/inference/degradation_cpu_gpu_ablation_20260514`
- 本地下载：`/Users/lixiaohui/Desktop/degradation_cpu_gpu_ablation_20260514`
- 设置：
  - 5 个 `320x192 / 17f / 8fps` 视频；
  - 每个视频先在 CPU 采样一份 params；
  - 同一份 params 分别在 CPU/GPU apply；
  - 输出 `gt.mp4 / lq_cpu.mp4 / lq_gpu.mp4 / absdiff_x8.mp4 / params.json`。
- 数值结果：
  - mean abs diff 范围约 `0.076-0.197`；
  - 差异明显，不应在未肉眼确认前默认认为 CPU/GPU 退化等价。
- 当前保守结论：
  - CPU 退化能解决 worker/GPU context 问题；
  - 但 CPU/GPU 退化输出并非 bitwise/视觉近似完全一致；
  - 正式切换前需要用户看本地结果决定是否接受。

仍需继续：

- 进一步确认正式多卡下 `worker=2/4/8` 的吞吐，不只看 2GPU smoke；
- 如果要低于 120GB，需要继续优化 Wan decoder selected-window backward，而不是继续调 DataLoader；
- `G_real/G_fake` / DMD dual optimizer 仍未接入，必须后续单独写 `v7-C` runner；
- 正式长训前需要做一次多卡 `v7-B` smoke，确认当前本地 runner 修复已同步到所有节点。

### 10.11 训练入口 dump 替代外部退化 ablation

用户检查后发现旧 CPU/GPU ablation 里的 `lq_gpu` 基本只有灰色轮廓。进一步核对后，该 ablation 不能作为训练输入质量的最终证据：

- 旧脚本默认退化配置比轻量测试集更重；
- 旧脚本对小尺寸视频继续做 x4 LQ，视觉退化会被放大；
- 退化内部包含噪声和视频编码随机性，固定外层 params 不能保证 CPU/GPU 严格等价；
- 外部脚本即使修正，也仍不等价于训练 DataLoader 真实输出。

因此已经改为直接在 Stage3 `v7-B` 训练代码里 dump 输入模型前的 batch：

- 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py`
- 参数：`--debug_dump_training_batch_dir`
- dump 点：`forward()` 最开始、`get_pipeline_inputs()` 之前
- 内容：DataLoader 输出的 `video` / `lq_video`
- 用途：判断 CPU 退化、采样、resize/crop 后真正送入模型前的 LQ 是否已经异常

已完成一次 17 帧 CPU 退化真实训练输入 dump：

- 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7b_training_input_dump_cpu_degra_17f_20260514/before_model`
- 本地：`/Users/lixiaohui/Desktop/stage3_v7b_training_input_dump_cpu_degra_17f_20260514/before_model`
- 重点文件：
  - `sample_000/gt_before_model.mp4`
  - `sample_000/lq_before_model.mp4`
  - `sample_000/meta.json`
- 该检查比外部 ablation 更可信，因为它直接来自当前训练代码中进入模型前的真实 batch。
