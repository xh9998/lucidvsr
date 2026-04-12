# FlashVSR 云端说明

## 基本约定

- 本地工作目录：`/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr`
- 云端工作目录：`/mnt/task_runtime/lucidvsr`
- 本地改动会同步到云端
- 云端结果统一放到更稳定的目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp`

## 运行环境

远程运行前先激活：

```bash
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
```

## 远程长跑

长时间实验统一建议：

1. 先登录远程
2. 起远程 `tmux`
3. 在 `tmux` 里启动训练
4. 输出继续写到 `run.log`

当前约定的远程 `tmux` 名字：

- `remote`

## 当前主训练

当前 8 卡主训练脚本：

- `wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Streaming-NormalRes-8GPU.sh`

当前主配置：

- `wanvideo/model_training/flashvsr/configs/stage1_normal_8gpu.yaml`
