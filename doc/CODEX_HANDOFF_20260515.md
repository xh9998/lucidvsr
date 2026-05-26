# Codex Handoff - FlashVSR / LucidVR

日期：2026-05-15

这份文档用于新 Codex 对话接手当前工作。先读这份，再按需读下面列出的 skill 和 doc。不要依赖聊天记忆。

## 1. 必读 skill

必须先读：

```text
/Users/lixiaohui/.codex/skills/用远程集群工作流/SKILL.md
```

skill 名称：

```text
用远程集群工作流
```

核心规则：

- 本地项目根目录：`/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr`
- 远端项目根目录：`/mnt/task_runtime/lucidvsr`
- 项目代码只在本地 `code/lucidvsr` 改，依赖本机 `sync` tmux 同步到远端。
- 不要直接在远端 `/mnt/task_runtime/lucidvsr` 手工改代码，否则会被本地 sync 覆盖。
- 运行 FlashVSR 相关脚本时显式使用：

```bash
export PYTHON_BIN=/mnt/conda_envs/flashvsr/bin/python
```

- 远端训练/测试结束或机器空闲时要恢复占卡：

```bash
bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh
```

- 启动关键实验前要检查本机 `sync` tmux 是否健康。
- 远端长训练必须放在远端 tmux 里，最好用持久 shell，不要 one-shot tmux。

## 2. 当前机器

当前主要 6 节点母机：

```text
t5qdtykjsw
a9suya6gxe
67dxkwcb7m
ui9n6p293s
g48bd6x4h7
gx2intv5rk
```

当前 2 节点 / 测试相关机器：

```text
8nh48ucn8b
6ai5mpi47f
```

本机固定 tmux：

```text
sync   # 代码同步，必须健康
watch  # 远程 nvidia-smi 监控
lxh    # 用户自己的本机窗口，不要随便杀
```

近期用于 48 卡训练的本机 tmux：

```text
codex_v7d1_48
```

注意：如果看到 Codex 提示 `maximum number of unified exec processes`，这是 Codex 工具侧保留了太多 exec 句柄，不是远端训练问题，也不是 W&B 问题。当前工具没有 list/close 这些句柄的接口。应避免再开长时间 `exec` / `sleep` / 轮询，改用短命令抓远端 tmux/log。

## 3. 文档体系

最重要的总账：

```text
doc/FLASHVSR_WORKLOG.md
```

用途：行为记录，按时间记实验目录、脚本、测试、结果、用户反馈。新对话必须优先读最近几节。

Stage3 主计划：

```text
doc/flashvsr_stage3_dmd_plan_20260511.md
```

用途：Stage3 / v7-A/B/C/D/D1 的技术计划、论文对齐、自查表、分支差异。

Stage3 显存 / 对齐计划：

```text
doc/flashvsr_stage3_memory_alignment_plan_20260514.md
```

用途：Stage3 显存拆解、CPU 退化、decoder prefix decode、offload、worker 测试等。

Stage3 v7-D 作者对齐计划：

```text
doc/flashvsr_stage3_v7d_author_aligned_plan_20260515.md
```

用途：`v7-D` 作者权重、DMD spike guard、W&B offline、validation 取舍。

稳定实验登记：

```text
doc/flashvsr_stable_experiment_registry.md
```

用途：长期要保留的 Stage1/Stage2 母本代码、yaml、sh、实验目录。

组会文档：

```text
doc/flashvsr_training_group_meeting_20260511.md
doc/flashvsr_training_group_meeting_20260514.md
```

用途：给 leader 汇报用，不是完整技术日志。

Stage2 / causal / mask 相关图和文档：

```text
doc/flashvsr_stage2_v6_design.md
doc/flashvsr_three_stage_author_aligned_20260429.md
doc/flashvsr_stage2_chunk_causal_mask.svg
doc/flashvsr_official_streaming_frame_rule.svg
doc/flashvsr_stage2_dit_kvcache_inference.svg
doc/flashvsr_stage2_v6_jump_probe_20260509.md
```

用途：理解 Stage2 89->85 / chunk causal / KV-cache / jump probe。

历史 v4/v5 文档：

```text
doc/flashvsr_v4_iteration.md
doc/flashvsr_v4_loss_and_data_investigation.md
doc/flashvsr_v5_iteration.md
doc/flashvsr_v53_flashinit_restart_20260423.md
```

用途：只在需要追溯早期数据/退化/采样设计时读取。

## 4. 当前主线状态

当前主线不是 `v7-D1`，而是回退后的 `v7-D stable`。

当前正式 48GPU 实验：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return
```

主节点：

```text
t5qdtykjsw
```

代码：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d_stable.py
```

配置：

```text
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_stable_authorweights_offline.yaml
```

启动脚本：

```text
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh
```

当前确认：

- 48 rank 已连接。
- W&B offline 写入实验目录。
- 已出 loss：

```text
[stage3_v7_b_loss] loss=1.073758 flow=0.019114 mse=0.020162 lpips=0.517241 ...
loss=0.193740 flow=0.193740
loss=0.075488 flow=0.075488
```

- rank0 节点观察到 GPU 大多能到 `90-100%`，但会有瞬时波动。
- step1 大约 `288s`，Stage3 明显慢。
- DataLoader worker 是开的：rank0 看到 `16` 个 worker，即 `8 rank * worker=2`。

## 5. W&B 当前策略

当前使用 W&B offline，离线文件必须写到 artifacts 实验目录，不要写到 `/mnt/task_runtime/lucidvsr/wandb` 后丢失。

实际同步策略已经改成 relay：

- 训练主节点 `t5qdtykjsw` 只负责把 offline run 打包上传到 S3。
- `6ai5mpi47f` 负责从 S3 拉包并执行 `wandb sync`。
- 原因：48 卡训练节点访问 `https://api.wandb.ai` 超时；6a 可以同步成功。

`v7-D stable` 已经在启动脚本里设置：

```bash
WANDB_MODE=offline
WANDB_DIR="${RUN_DIR}"
```

当前 `v7-D stable` 的同步 tmux：

```text
t5qdtykjsw: wandb_package_v7d_stable
6ai5mpi47f: wandb_sync_from_s3_v7d_stable
```

同步脚本：

```text
wanvideo/model_training/flashvsr/scripts/package_wandb_offline_to_s3_loop.sh
wanvideo/model_training/flashvsr/scripts/sync_wandb_offline_from_s3_loop.sh
```

当前 S3 中转包：

```text
s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return.tar.gz
```

说明：

- t5 侧每小时打包一次，并把 `/mnt/task_runtime/lucidvsr/wandb/offline-run-*` mirror 到 `${RUN_DIR}/wandb` 后再打包。
- 6a 侧每小时下载一次 S3 包并对其中的 `offline-run-*` 执行 `wandb sync`；`wandb sync` 是幂等的，重复同步可接受。
- 新脚本内部显式设置 `PATH`、`NOTARY_CONFIG_FILE`、`AWS_CA_BUNDLE`，避免新 tmux 丢环境。
- 训练本身不依赖 W&B 在线同步。

## 6. `v7-D1` 为什么暂停

`v7-D1` 文件：

```text
wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d1_lora.py
wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d1_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb.yaml
wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D1-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-641Data-OfflineWandb.sh
wanvideo/model_inference/flashvsr/history/run_stage3_v7_d1_scan89_step500_incremental.sh
```

相对 `v7-D stable` 改动：

- 内部 validation 从 1 个样本变成 2 个样本。
- validation 来源改为训练 batch cache。
- validation 走 one-step streaming KV-cache。
- 日志命名清理为 `v7-D1`。

结果：

- 2GPU smoke 能进入训练循环，step1 出 loss。
- validation gating 生效，step1 只缓存 `1/2` 样本，不会提前 val。
- 但 48GPU 下 step2 计算异常重，rank0 显存约 `156-157GB`，耗时明显劣于 `v7-D stable`。
- 当前结论：不作为主线继续推进，先回到 `v7-D stable`。

## 7. Stage1 / Stage2 稳定母本

Stage1 89f 母本：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors
```

Stage2 `641` 母本：

```text
/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors
```

旧 `b8gkuie2ns` 机器已死，但 artifacts 备份在：

```text
s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts
```

如果远端缺 Stage1/Stage2 ckpt，从这个 S3 路径恢复。

## 8. Stage3 当前核心设定

`v7-D stable` / 当前 Stage3 DMD 方向：

- Student 从 Stage2 `641 step-6000` 初始化。
- `G_real` 使用 Stage1 `535 step-10000`，冻结。
- `G_fake` 可训练，初始化来自 Stage1 `535 step-10000`。
- `fake_fm_weight=1.0`
- `dmd_weight=1.0`
- `pixel_mse_weight=1.0`
- `lpips_weight=2.0`
- 首帧 pixel / LPIPS 权重为 `4.0 / 4.0`，但只有 window 覆盖全局首 latent 时才生效。
- DMD spike guard：

```text
stage3_dmd_grad_norm_max=5.0
stage3_dmd_spike_policy=skip
```

目前训练日志里仍可能显示 `[stage3_v7_b_loss]`，这是历史命名残留，不代表在跑 v7-B 旧逻辑。

## 9. 近期测试与推理

`v7-C6 step100/200` 10 个 synthetic 测试：

远端：

```text
/mnt/task_wrapper/user_output/artifacts/inference/v7d_c6_step100_200_10synthetic_notile_20260515
```

本地桌面：

```text
/Users/lixiaohui/Desktop/v7d_c6_step100_200_10synthetic_notile_20260515
```

结果：

```text
step-100: 10 videos, 162s, about 16.2s/video
step-200: 10 videos, 162s, about 16.2s/video
```

推理语义：

- one-step Stage3 inference
- `tiled=false`
- input bicubic x4
- color fix：`adain`
- Stage2/Stage3 streaming/block sparse 参数按当前脚本设置。

## 10. 常见错误和处理

### 10.1 Codex unified exec 上限

现象：

```text
Warning: The maximum number of unified exec processes you can keep open is 60 ...
```

含义：

- 这是 Codex 工具后端的 exec 句柄上限。
- 不是远端训练问题，不是 W&B 问题，不是本机 tmux 问题。
- 当前工具没有关闭这些句柄的接口。

处理：

- 不要再用长 `exec_command` 轮询。
- 不要本地 `sleep 30` 后抓输出。
- 改用远端 tmux 里启动监控，Codex 只短命令 `capture-pane`。
- 如果必须彻底重置，开新 Codex 对话，并让新对话先读本 handoff。

### 10.2 sync 掉了

先检查：

```bash
tmux list-windows -t sync
```

如果掉了，在本机执行：

```bash
cd ~/Library/CloudStorage/Box-Box/code
bolt task sync <task_id>
```

不要直接 scp/远程写项目代码。

### 10.3 W&B 卡住

当前策略是 offline + relay。若在线卡住，不要强行修训练节点在线网络：

1. 确认 t5 的 `wandb_package_v7d_stable` 还在打包上传 S3。
2. 确认 6a 的 `wandb_sync_from_s3_v7d_stable` 还在下载并 `wandb sync`。
3. 若需要手动补同步，在 6a 下载 S3 包后执行 `/mnt/conda_envs/flashvsr/bin/wandb sync <offline-run-dir>`。

### 10.4 validation 卡住

历史上 validation 单独采样容易卡。`v7-D1` 尝试从训练 batch cache 做 validation，但 48GPU 太重，当前暂停。`v7-D stable` 使用 C6-stable validation。

### 10.5 机器空卡

测试/训练结束后要启动占卡：

```bash
bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh
```

不要用低显存占卡版本，用户明确不希望 GPU 利用率记录难看。

## 11. 新对话建议开场

新 Codex 对话建议第一条发：

```text
请先阅读：
/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/CODEX_HANDOFF_20260515.md
以及 skill：用远程集群工作流
然后接手当前 FlashVSR/LucidVR Stage3 工作。
```

然后让新对话先执行：

```bash
tail -120 doc/FLASHVSR_WORKLOG.md
tail -120 doc/flashvsr_stage3_dmd_plan_20260511.md
```

再继续远程操作。

### 10.8 恢复本机 `watch` tmux 的正确方法

用途：本机 `watch` 会话用于同时连接当前 8 个机器，并在每个远端窗口中运行 `watch -n 1 nvidia-smi`。

当前机器顺序固定为：

```text
t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk 8nh48ucn8b 6ai5mpi47f
```

推荐直接用 `bash -lc`，不要在 zsh 里用普通字符串拆词，也不要用 zsh 的 0-based 数组假设；zsh 默认数组是 1-based，字符串也不会自动按空格拆成多个词，容易建错窗口。

```bash
bash -lc '
set -e
machines=(t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk 8nh48ucn8b 6ai5mpi47f)
tmux kill-session -t watch 2>/dev/null || true
tmux new-session -d -s watch -n "${machines[0]}"
for i in 1 2 3 4 5 6 7; do tmux new-window -t watch -n "${machines[$i]}"; done
for i in 0 1 2 3 4 5 6 7; do
  m=${machines[$i]}
  tmux send-keys -t "watch:$i" "cd ~/Library/CloudStorage/Box-Box/code && bolt task ssh $m" C-m
  sleep 0.2
done
sleep 5
for i in 0 1 2 3 4 5 6 7; do
  tmux send-keys -t "watch:$i" "watch -n 1 nvidia-smi" C-m
done
tmux list-windows -t watch
'
```

验收：

```bash
tmux list-windows -t watch
tmux capture-pane -pt watch:0 -S -8 | tail -8
tmux capture-pane -pt watch:7 -S -8 | tail -8
```

如果窗口中还停在本机 shell，没有进入远端，通常是 `bolt task ssh <id>` 失败或本机 VPN/代理状态不对；先修 ssh，不要在本机直接跑 `watch -n 1 nvidia-smi`。
