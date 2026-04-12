# FlashVSR 训练/测试代码地图

这份文档只记录当前仓库里仍然建议使用的训练与测试入口，以及它们各自的设定来源。

## V1 旧逻辑复现

目标：
- 复原 `train_stage1_release_8gpu_20260408_135241` 那条线的核心思路。
- `lq_proj_in` 只做首层注入。
- `lq_proj_in` 在训练/validation 中不走流式 `stream_forward` 拼接，而是整段 `fullclip` 投影。
- validation 使用 `posi_prompt.pth`，`cfg=1`，`50` 步。

训练主文件：
- [train_flashvsr_stage1_v1.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v1.py)

8 卡训练：
- 启动脚本：
  [FlashVSR-Stage1-Release-8GPU-v1.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v1.sh)
- 配置：
  [stage1_release_8gpu_v1.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v1.yaml)

2 卡 smoke：
- 启动脚本：
  [FlashVSR-Stage1-Release-Smoke-2GPU-v1.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU-v1.sh)
- 配置：
  [stage1_release_smoke_2gpu_v1.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu_v1.yaml)

配套推理：
- [infer_flashvsr_stage1_v1.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v1.py)

说明：
- 这套是“训练与推理严格对齐”的旧逻辑版本。
- 适合检查 `posi_prompt.pth + cfg=1` 这条历史 validation 路径到底会输出什么。

## 当前主线训练

目标：
- 继续对齐 FlashVSR release / debug 过的新版逻辑。
- 包含后续加过的 validation baseline、流式首层注入对齐等实验能力。

训练主文件：
- [train_flashvsr_stage1.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1.py)

8 卡训练：
- [FlashVSR-Stage1-Release-8GPU.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU.sh)
- [stage1_release_8gpu.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_8gpu.yaml)

2 卡 smoke：
- [FlashVSR-Stage1-Release-Smoke-2GPU.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU.sh)
- [stage1_release_smoke_2gpu.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu.yaml)

说明：
- 这套包含后面为了 debug 加入的能力。
- 不适合再拿来当“昨天 135241 那版”的严格复现。

## 纯 Wan 基线测试

目标：
- 证明基础 Wan T2V 是否正常。
- 单独测试 prompt / cfg 对输出的影响。

脚本：
- [wan_text_cfg1_t2v.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_inference/wan_text_cfg1_t2v.py)

常用 prompt 文件：
- [prompt.txt](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/prompt.txt)
- [prompt_short.txt](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/prompt_short.txt)
- [prompt_neutral.txt](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/prompt_neutral.txt)

说明：
- 这里只测纯 Wan。
- 不接 LoRA，不接 `lq_proj_in`。

## Stage1 对齐推理

目标：
- 不依赖训练 callback。
- 从纯 Wan 文本推理链出发，再加 stage1 的 LoRA 和 `lq_proj_in`。

脚本：
- [infer_flashvsr_stage1_wan_aligned.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_wan_aligned.py)

说明：
- 这套不是旧的 `posi_prompt + cfg=1` 路径。
- 它用正常文本 prompt 和 Wan 推理主链，方便检查 `lq` 条件是否真正起作用。

## 旧脚本和现状

- `infer_flashvsr_stage1.py`
  - 当前仍可用，但它绑定的是主线 `train_flashvsr_stage1.py`。
  - 如果要严格复现旧逻辑，请改用 `infer_flashvsr_stage1_v1.py`。
- `infer_flashvsr_release_aligned.py`
  - 更偏 FlashVSR release/full pipeline 对齐，不是当前 stage1 主诊断入口。
- `infer_flashvsr_validation_style.py`
  - 用于复现训练内 validation 样本路径。
- `infer_flashvsr_external_mp4_tensor_style.py`
  - 用于把外部 mp4 转成 tensor 风格输入做对照。

## Scripts

目标：
- 放训练辅助脚本，避免继续靠临时命令。

文件：
- [plot_flashvsr_loss_from_log.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/scripts/plot_flashvsr_loss_from_log.py)
- [export_flashvsr_eval_samples.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/scripts/export_flashvsr_eval_samples.py)

说明：
- `plot_flashvsr_loss_from_log.py`
  - 从 `run.log` 提取 loss，导出 `csv/png`
- `export_flashvsr_eval_samples.py`
  - 读取训练 yaml
  - 构建同一套 dataset
  - 固定导出 3 组 `hq/lq` 测试视频和 tensor

## V2 对照版

目标：
- 保持 `V1` 的训练主链不变。
- validation / inference 改成以 `Wan` fixed-prompt 基线为起点，再接 stage1 的条件支路。
- validation / inference 使用：
  - `posi_prompt.pth`
  - `cfg=1`
  - `50` 步
  - 首层 `projection`
  - 非流式 `fullclip`

训练主文件：
- [train_flashvsr_stage1_v2.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py)

8 卡训练：
- [FlashVSR-Stage1-Release-8GPU-v2.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v2.sh)
- [stage1_release_8gpu_v2.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v2.yaml)

2 卡 smoke：
- [FlashVSR-Stage1-Release-Smoke-2GPU-v2.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU-v2.sh)
- [stage1_release_smoke_2gpu_v2.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu_v2.yaml)

配套推理：
- [infer_flashvsr_stage1_v2.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py)

说明：
- `V2` 的目的就是和 `V1` 做 validation 对照。
- 可以近似理解为：
  - `V1`：旧 stage1 validation 链
  - `V2`：Wan fixed-prompt 基线 + LoRA + 首层 `projection`

深度说明：
- [FLASHVSR_V2_DEEPDIVE.md](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/docs/FLASHVSR_V2_DEEPDIVE.md)

## V2 Debug Overfit

目标：
- 用极小固定样本集快速过拟合，验证“训练时间不够”还是“代码/控制链有问题”。
- 不再依赖 streaming dataset。

训练主文件：
- [train_flashvsr_stage1_v2_debug.py](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2_debug.py)

8 卡训练：
- [FlashVSR-Stage1-Release-8GPU-v2-Debug.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v2-Debug.sh)
- [stage1_release_8gpu_v2_debug_overfit.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v2_debug_overfit.yaml)

2 卡 smoke：
- [FlashVSR-Stage1-Release-Smoke-2GPU-v2-Debug.sh](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU-v2-Debug.sh)
- [stage1_release_smoke_2gpu_v2_debug_overfit.yaml](/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/stage1_release_smoke_2gpu_v2_debug_overfit.yaml)

数据来源：
- 固定目录中的 `sample_xxx/hr.pt` 和 `sample_xxx/lq.pt`
- 默认指向：
  - `/mnt/task_wrapper/user_output/artifacts/eval_samples/train_stage1_release_8gpu_v2_20260409_121105`

说明：
- 与 `v2` 主线相比，唯一核心变化是 dataset。
- 模型、LoRA、projection、validation callback 仍沿用 `v2`。
- 适合做：
  - batch size 探测
  - 快速过拟合
  - 固定样本结构控制验证
- 8 卡固定 overfit 启动模板：
  - `wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v2_debug_overfit_17f_bs24.yaml`
  - `wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v2-Debug-17f-bs24.sh`
  - 用途：固定 3 个样本、17 帧、`alpha=5`、`bs=24`、wandb 打开，避免再通过命令行临时改 batch / config。
  - 现在这套已归档到 `history/`，不再视为标准入口。

## 17 帧全量训练集实验

目标：
- 保持 `17帧 / alpha=5 / 50步 validation / posi_prompt / cfg=1 / 8卡`
- 数据入口从固定 3 样本 overfit 切回全量训练集
- 用更小 batch 和更保守学习率验证全量训练是否稳定

历史配置：
- `wanvideo/model_training/flashvsr/configs/history/stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5.yaml`

历史启动脚本：
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-8GPU-v2-17f-Full-bs4-lr1e5-alpha5.sh`

说明：
- 使用 `train_flashvsr_stage1_v2.py`
- 不是 overfit 线，不走 `debug_fixed`
- 训练集恢复为正常 streaming 数据源
