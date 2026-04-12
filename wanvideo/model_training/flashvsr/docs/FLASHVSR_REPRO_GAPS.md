# FlashVSR 当前缺口

## 已完成

- Stage 1 数据读取主链路
- `storymotion` manifest 化
- `takano` shard 读取
- 在线退化
- 固定 prompt
- `lq_proj_in + dit lora`
- `yaml` 配置
- 在线 validation
- `wandb` 参数接入
- `DeepSpeed ZeRO-2` 启动路线

## 还缺的

### 1. 显存优化还不够

当前最大问题是：

- `768x1280`
- 长帧数
- 大 `lora rank`

这几项叠加后，仍然容易把显存打满。

需要继续验证：

- `flash-attn` 装好后的收益
- 更合适的帧数
- 更合适的 `lora target/rank`
- 是否要把退化固定到 CPU

### 2. 论文一致性还不完整

当前还没有完全补齐论文里的视频-图像联合训练细节，例如：

- image/video segment mask
- 图像联合训练的正式版实现
- 更完整的训练策略对齐

### 3. 长跑稳定性还要继续验证

当前已经能做 smoke 和短跑，但还需要继续观察：

- 远程长跑稳定性
- step-0 validation 与长跑共存是否稳定
- wandb 长时间记录是否稳定
