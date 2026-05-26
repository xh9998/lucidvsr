# FlashVSR Stage2 v6 跳变定位记录（2026-05-09）

## 1. 问题

`v6` 二阶段 89 帧推理出现周期性跳变，肉眼上基本按 chunk 边界发生。需要判断跳变来自哪里：

- KV cache / streaming chunk 推理；
- stage2 chunk causal mask；
- block sparse top-k 太窄；
- LQ projector 4 帧一组输出边界；
- colorfix 或多步 diffusion 放大。

## 2. 固定测试条件

固定 checkpoint：

`/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors`

固定测试输入：

`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`

固定基础参数：

- 输入：`89` 帧；
- 推理尺寸：`1280x768`；
- 输入先做 `bicubic x4`：`--input_bicubic_upscale 4.0`；
- 推理步数：默认 `50`；
- `lq_proj_scale=1.0`；
- colorfix：`adain`；
- base model：`/mnt/models/Wan2.1-T2V-1.3B`；
- prompt tensor：`/mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth`。

最终整理目录：

`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509`

目录内容：

- `videos_10_fullset/`：A-D、历史 `v6.1/v6.2 step-10000` 的 10-video 输出；
- `videos_takano04_only/`：A-F、H、历史 `v6.1/v6.2` 的 `takano_04` 单视频；
- `frames_takano04/`：A-F、H、历史 `v6.1/v6.2`、FlashVSR official 的逐帧拆解；
- `diagnostics/`：G projector chunk stats；
- `contact_sheets/takano04_boundary_frames_A_to_F_v61_v62_flashvsr.png`：A-F 初版关键边界帧总览图；
- `contact_sheets/takano04_boundary_frames_A_to_H_v61_v62.png`：加入 H 后的关键边界帧总览图。

## 3. Probe 设计和结果

| Probe | 设置 | 目的 | 结果 |
|---|---|---|---|
| A | `v6.2` full-DiT mask no KV cache，`block_sparse_chunk_causal`，`local_num=11`，`topk_ratio=2` | 排除 KV cache streaming 是否是唯一原因 | 跳 |
| B | 同 A，但 `topk_ratio=4` | 判断 top-k 太小是否是主因 | 跳 |
| C | `dense_full` | 移除 sparse/chunk causal 路径 | 不跳 |
| D | 同 A，但 `topk_ratio=8` | 继续扩大 top-k | 跳 |
| E | 同 A，但 `num_inference_steps=1` | 判断 50-step 是否放大边界 | 只能作辅助，1-step 本身不代表画质 |
| F | no colorfix / AdaIN offset / wavelet | 判断首帧和颜色是否由 colorfix 主导 | 不改变“按 chunk 跳”的主结论 |
| G | dump LQ projector chunk stats | 判断 projector 输出是否数值爆掉 | projector 全局统计稳定 |
| H | `v6.2` full-DiT mask no KV cache，`block_sparse_official_mask`，`local_num=11`，`topk_ratio=2` | 诊断推理侧 FlashVSR 官方 spatial local mask 与按 temporal chunk 分组的 top-k block pair 选择 | 仍更接近 sparse/chunk 路径，未变成 C 的稳定状态 |
| 历史 v6.1 | cache/streaming 推理 | 与旧推理路径对照 | 跳 |
| 历史 v6.2 | full-sequence mask no-cache 推理 | 与旧 v6.2 对照 | 跳 |
| FlashVSR official | 官方推理 baseline | 观察官方是否也有边界变化 | 有轻微边界变化，但幅度小很多 |

## 4. 关键观察

逐帧看 `takano_04`：

- A/B/D 都有明显 chunk 周期跳变；
- 历史 `v6.1/v6.2` 也有类似跳变；
- 只有 C `dense_full` 明显稳定；
- H 使用官方式 sparse mask/top-k 后，仍没有变成 C 的稳定状态；
- FlashVSR official 也不是完全没有边界变化，但跳变幅度明显小得多。

用户观察到的 A/B/D 跳变段大致为：

- `1-21`
- `22-29`
- `30-37`
- `38-46`
- ...
- `62-70`
- `70-85`

这个分段和 Stage2 chunk 很吻合：

- 第一段 `1-21` 约等于开头 `6 latent-time = 3 chunks` 对应的原始帧范围；
- 后续每段约 `8` 帧，对应 `2 latent-time = 1 chunk`。

因此，“chunk sparse/causal 路径导致边界不连续”这个结论已经比较稳定。

## 5. G 的 projector chunk stats 说明了什么

G 的 CSV：

`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509/diagnostics/takano04_projector_chunk_stats.csv`

关键数值：

- 89 帧输入经过 Stage2 projector 后输出 `22` 个 chunk；
- 每个输出 chunk 是 `3840` tokens；
- `std` 从 `0.0844` 到 `0.0857`，相对变化约 `1.5%`；
- `l2` 从 `205.0` 到 `208.2`，相对变化约 `1.5%`；
- 相邻 chunk 最大 L2 差约 `0.77`，占整体 L2 约 `0.37%`。

结论：

- LQ projector 输出尺度稳定；
- 没看到某个 chunk 数值突然爆掉；
- projector 仍然提供了 4 帧一组的边界，但不像跳变的直接数值根因；
- 结合 C 不跳，问题更像是 sparse/chunk-causal attention 把边界放大成了视觉不连续。

## 6. 当前结论

当前结论稳定为：

- 跳变不是单纯由 KV cache 引起，因为 no-cache 的 A/B/D 和历史 v6.2 也跳；
- 跳变不是单纯由 top-k 太小引起，因为 `topk_ratio=2/4/8` 都跳；
- 跳变不是单纯由 projector 数值爆掉引起，因为 G 的 projector chunk stats 稳定；
- 跳变也不是“top-k 完全没有按官方式 chunk 分组选”这一点单独造成，因为 H 在推理诊断里加入官方式 spatial local mask + chunk-grouped top-k 后仍未恢复到 C 的稳定状态；后续训练主路径也已统一为 chunk-grouped top-k，但不加入 spatial local mask。
- 跳变不是普通模型质量问题，因为 C dense full 不跳；
- 更合理的根因是 sparse/chunk causal 路径仍有更深的训练-推理对齐问题，或当前 mask / top-k block pair 构造与 FlashVSR 官方调用仍有细节差异。当前代码已经调用 `Block-Sparse-Attention` 的 `block_sparse_attn_func` CUDA kernel，缺失 kernel 时会直接报错，不是静默退回 Python fallback。

一句话：

`v6` 当前不是“模型学不会”，而是 sparse/chunk causal 推理路径本身在 chunk 边界上不连续；dense full attention 可以绕开这个问题。

## 7. 是否应该立刻重训

不建议现在直接重训同一套 `v6`。

理由：

- 现有证据指向结构/推理路径问题，不是训练步数不够；
- 如果继续用当前 sparse/chunk causal 实现重训，大概率仍会学到同样的边界 artifact；
- 第三阶段 pixel loss 可能压平跳变，但如果 teacher 或 sparse path 本身仍跳，会把问题蒸馏进模型，风险较高。

更合理的顺序：

1. 不建议继续用当前 sparse/chunk 路径盲目重训 Stage2。
2. 如果要继续 Stage2 sparse，需要优先核对官方 block-sparse kernel 的真实执行路径、训练时 sparse mask 与推理时 block pair 选择是否完全一致。
3. 同时保留 C `dense_full` 作为 teacher / upper-bound，对第三阶段 pixel distillation 做准备。
4. 第三阶段 pixel loss 可以尝试压平跳变，但不要用会明显跳变的 sparse 输出直接当 teacher。

## 8. 下一步

已完成 `Probe H`：

- 远端输出：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeH_takano04_20260510`
- 本地下载：`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeH_takano04_20260510`
- 合并视频：`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509/videos_takano04_only/H_official_mask_topk2_local11.mp4`
- 拆帧目录：`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509/frames_takano04/H_official_mask_topk2_local11`
- 总览图：`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509/contact_sheets/takano04_boundary_frames_A_to_H_v61_v62.png`

后续如果继续排查，优先级应改为：

- 核对训练侧 `generate_causal_block_mask` / temporal `local_num` 与作者训练口径是否完全一致；
- top-k block pair 训练主路径已统一为官方式 chunk-grouped selection；
- `local spatial mask` 只作为官方高分辨率推理侧诊断，不应混入当前 Stage2 训练；
- 对比官方训练是否真的使用同一个 sparse kernel，而不是 dense/masked 近似；
- 检查训练时 `topk` 的 block pair 是否 stop-gradient、是否每个 denoise step 动态重算；
- 用 C dense full 输出作为上界，判断第三阶段 pixel distillation 是否能压平 sparse 边界。
