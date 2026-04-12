# Inference History

这里记录具体实验对应的推理启动脚本。

根目录保留的标准文件：

- `infer_flashvsr_stage1_v2.py`
- `run_test_flashvsr_stage1_v2.sh`
- `run_batch_flashvsr_stage1_v2.sh`

如果某次推理是为某个特定实验、特定步数、特定输入集定制的，就把启动脚本放到 `history/`，并在输出目录里保留：

- `driver.sh`
- `launch_command.sh`
- `launch_command.txt`
- `run.log`
