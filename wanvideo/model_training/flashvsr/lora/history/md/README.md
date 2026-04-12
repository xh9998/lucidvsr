# Lora History

这个目录只放实验性或一次性启动脚本的归档副本。

根目录里长期保留的标准脚本：

- `FlashVSR-Stage1-Release-8GPU.sh`
- `FlashVSR-Stage1-Release-8GPU-v1.sh`
- `FlashVSR-Stage1-Release-8GPU-v2.sh`

其它为了排查问题、做特定 batch/frame/debug/wantext/overfit 的脚本，都应归档到这里，并在对应实验目录里保留：

- `launch_command.sh`
- `launch_command.txt`
- `snapshot/`

如需新增实验性启动脚本，优先：

1. 在根目录创建可运行脚本
2. 跑通后复制一份到 `history/`
3. 在实验目录留存当次启动证据
