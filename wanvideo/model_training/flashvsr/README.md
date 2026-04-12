# FlashVSR 训练目录

这里放的是当前这版 FlashVSR Stage 1 训练代码。

当前主入口：

- `train_flashvsr_stage1.py`
- `configs/stage1_normal_8gpu.yaml`
- `lora/FlashVSR-Stage1-Streaming-NormalRes-8GPU.sh`

当前这版训练已经具备：

- `storymotion + takano` 两路数据读取
- `yaml` 驱动训练参数
- `DeepSpeed ZeRO-2`
- `gradient checkpointing`
- `gradient checkpointing offload`
- 固定 prompt 条件
- `lq_proj_in + dit lora`
- 在线 validation
- 可选 wandb

当前主要问题已经不在 dataloader 主链路，而在大分辨率长视频训练时的显存优化。

当前稳定能跑通的结论：

- 小一些的输入规模可以正常 forward / backward / save ckpt
- 正常分辨率配置已经能起训练，但显存边界和长期稳定性还需要继续调

建议直接看的文件：

- `train_flashvsr_stage1.py`
- `configs/stage1_normal_8gpu.yaml`
- `lora/accelerate_zero2_flashvsr_8gpu.yaml`
- `docs/FLASHVSR_REPRO_GAPS.md`
- `docs/FLASHVSR_WORKLOG.md`
