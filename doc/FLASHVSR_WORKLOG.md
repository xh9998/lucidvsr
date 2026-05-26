# FlashVSR 工作记录

这份文档记录 FlashVSR 相关训练、推理、数据与调试过程。

记录原则：

- 按时间顺序追加
- 用中文记录
- 保留关键实验目录、配置变化、代码改动、结果与用户反馈
- 版本结构说明另见 `flashvsr_v5_iteration.md` 等版本文档

## 2026-04-07 到 2026-04-08：训练主链恢复与基础打通

- 恢复并整理了 `lucidvsr` 工作目录中的训练主文件、数据层、推理脚本和文档。
- 修复训练脚本对 `prompt_tensor_path` 的处理：
  - 允许从 yaml 读取，不再只能从命令行硬传。
- 修复训练日志默认不打印 `loss` 的问题：
  - `run.log` 会持续打印 `step/loss`。
- 修复长跑任务容易被外部 `SIGHUP` 杀掉的问题：
  - 后续大训练改为远程 `tmux remote` 启动。
- 修复数据集输出 `PIL.Image` 导致多卡拼 batch 失败的问题：
  - 统一改为 tensor 化返回。
- 修复 parquet 路径下 `video_iter=None` 的问题。
- 修复 `storymotion manifest + takano shard` 混合输入时只走一条源的问题。
- 修复 distributed/rank 切分不完整的问题，避免多卡重复读相同样本。

DeepSpeed / FlashAttention / Checkpoint 方面：

- 确认 `DeepSpeed ZeRO-2` 真正生效，不只是配置文件里写了 `zero_stage=2`。
- 确认运行时真实命中 `flash_attn_2` 分支。
- 发现 LoRA patch 后 `pipe.dit` 仍停在 `eval()`，导致 `gradient checkpoint` 实际没有生效。
- 修复后，训练真实命中 `gradient_checkpoint_forward()` 与 deepspeed checkpoint 路径。
- 修复 checkpoint 路径里的输入 / 上下文传递问题，没有改变 DiT 本身数学结构。
- 修复 fixed prompt 在 `batch_size > 1` 时没有按 batch 展开的错误。
- 修复 8 卡 `accelerator.prepare` 卡住的问题：
  - 关键是给 deepspeed 明确补齐 `train_micro_batch_size_per_gpu`。

推理与验证方面：

- 新增 `wanvideo/model_inference/flashvsr/`，用于测试导出的 `step-xxx.safetensors`。
- 已确认训练导出的 stage1 ckpt 只包含：
  - `lq_proj_in.*`
  - `dit` 上的 LoRA 权重
- 已确认推理时能正确加载：
  - base Wan 1.3B
  - `lq_proj_in`
  - LoRA
- 外部推理目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_step400_20260408_013404/`
- 新增两条更严格的推理检查路径：
  - `infer_flashvsr_validation_style.py`
    - 直接复用训练里的 `collect_fixed_validation_samples()` 与 `lq_tensor`
  - `infer_flashvsr_external_mp4_tensor_style.py`
    - 外部 mp4 先转 tensor，再走更接近训练的推理链

原始 FlashVSR 对齐检查：

- 在原始 FlashVSR 推理链中加入默认关闭的 debug dump：
  - `examples/WanVSR/infer_flashvsr_full_cloud.py`
  - `diffsynth/pipelines/flashvsr_full.py`
  - `flashvsr_inference_cloud_full.sh`
- 原仓调试结果目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_original_debug_20260408_051536`
- 关键统计确认：
  - 输入 `LQ_video` 值域为 `[-1, 1]`
  - 初始噪声 std 约为 `0.999`
  - `LQ_proj_in` 第一层输出 std 约为 `0.0837`
  - `patchify` 后模型 token 与对齐后的 `LQ_latents` 维度一致

结论：

- 训练链路、推理链路、validation 链路都已基本打通。
- 问题重心从“能不能跑”转向“训练效果弱”和“训练/验证是否完全对齐”。

## 2026-04-08：release 风格训练对齐

- 基于公开 release 推理结构新增单独训练配置：
  - `stage1_release_smoke_2gpu.yaml`
  - `stage1_release_8gpu.yaml`
- 核心变化：
  - `lq_proj_layer_num: 1`
- 明确 release 风格与早期训练版的差异：
  - release 推理 `LQ_proj_in` 只输出 `1` 组条件特征
  - 早期训练版是 `30` 层逐层注入

2 卡 smoke 实验：

- 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_20260408_122919`
- 已确认：
  - `validation_at_start` 成功执行
  - validation 后训练继续进行
  - 已命中 `gradient_checkpoint`
  - 已出第一条 `loss`：`step=1 loss=1.358621`

同期修复：

- validation 后 scheduler 没有切回训练态，导致训练阶段在 `add_noise()` 中越界。
- 修复方式：
  - 在训练 `forward()` 前显式 `scheduler.set_timesteps(1000, training=True, shift=5.0)`
  - validation 结束后显式恢复训练 scheduler 状态

## 2026-04-09：step-0 初始化、早期 checkpoint 与对齐排查

### step-0 初始化排查

- 重点检查 LoRA 与 `lq_proj_in` 初始化。
- 确认 `PEFT LoraConfig(init_lora_weights=True)` 默认就是零影响初始化。
- 真正问题在 `lq_proj_in`：
  - `FlashVSRLQProjIn` 的最终 `linear_layers` 原本是随机初始化
  - 会在 step-0 就主动扰动底模

修复：

- 在 `DiffusionTrainingModule.add_lora_to_model()` 中显式写死 `init_lora_weights=True`
- 在 `FlashVSRLQProjIn` 中默认将最终输出投影层清零初始化
- 若从 `lq_proj_checkpoint` 恢复，则不覆盖 checkpoint 内容

实验：

- 新 2 卡 smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_zeroinit_20260408_125119`

结论：

- LoRA 初始化不是主问题。
- `lq_proj_in` 非零初始化会污染 step-0 表现。
- 之后的对照都以“LoRA 零影响 + `lq_proj_in` 输出零初始化”为基线。

### early validation checkpoints

- 新增 `extra_save_steps`，在 `10/25/50/100` 这些早期步额外保存并触发 validation。
- 修改文件：
  - `diffsynth/diffusion/parsers.py`
  - `diffsynth/diffusion/logger.py`
  - `diffsynth/diffusion/runner.py`
  - `wanvideo/model_training/flashvsr/configs/stage1_release_8gpu.yaml`

### 论文理解修正与 release 训练诊断

- 修正对论文的理解：
  - 不再把“30 层逐层注入”当成论文直接写出的结论
  - 当前代码里的 30 层注入是实现设计，不是论文文本本身

实验：

- `train_stage1_release_8gpu_20260408_135241` 已生成：
  - `step-0`
  - `step-10`
  - `step-25`
  - `step-50`
  - `step-100`

检查 `step-100.safetensors`：

- `lq_proj_in` 已参与训练，不是完全没学
- LoRA 也已参与训练
- 但 `lq_proj_in.linear_layers.0.weight/bias` 量级仍然很小

结论：

- “loss 低但 validation 弱”不能简单归因于 projection 没参与训练
- 更像是训练 / validation 路径或目标存在不对齐

### stage1 离线推理对齐

- 发现早期 `infer_flashvsr_stage1.py` / `infer_flashvsr_external_mp4_tensor_style.py`
  - 默认按 `len(pipe.dit.blocks)=30` 构建 `lq_proj_in`
  - 与 `stage1_release_*` 训练中的 `lq_proj_layer_num=1` 不一致
- 修复后，离线推理会从 ckpt 自动推断 `lq_proj_layer_num`

### v1 / v2 路线分化开始

- 明确：
  - `v1`：复原早期 stage1 验证链
  - `v2`：训练主链保持接近 `v1`，但 validation / inference 改为更接近 Wan fixed-prompt 基线

## 2026-04-10：v2 主线、辅助脚本与文档化

### v2 validation 对照版

- 新增文件：
  - `train_flashvsr_stage1_v2.py`
  - `stage1_release_8gpu_v2.yaml`
  - `stage1_release_smoke_2gpu_v2.yaml`
  - `FlashVSR-Stage1-Release-8GPU-v2.sh`
  - `FlashVSR-Stage1-Release-Smoke-2GPU-v2.sh`
  - `infer_flashvsr_stage1_v2.py`

v2 的核心定义：

- 训练主链仍接近 `v1`
- validation / inference 不再复用旧 `FlashVSRStage1Pipeline.infer_from_lq()`
- 改为以 Wan fixed-prompt 基线为起点，再叠加：
  - LoRA
  - `lq_proj_in`
  - 首层注入
  - 非流式 `fullclip`
  - `posi_prompt.pth`
  - `cfg=1`
  - `50` 步推理

### 记录、脚本与缓存样本

- 新增 `scripts/` 目录：
  - `plot_flashvsr_loss_from_log.py`
  - `export_flashvsr_eval_samples.py`
- 用途：
  - 从 `run.log` 抽取 `step/loss` 并导出 `csv/png`
  - 按训练 yaml 构建 dataset，固定抽取 `hq/lq` 样本导出到 `artifacts`

### v2 deep dive 文档

- 新增：
  - `docs/FLASHVSR_V2_DEEPDIVE.md`
- 用来说明：
  - `train_flashvsr_stage1_v2.py` 的模块职责
  - 数据流
  - validation 两个分支
  - 与 `v1` 的真实差异

### v2 明确 bug 修复

- `train_stage1_release_smoke_2gpu_v2_wantextdog_20260409_112420` 报错：
  - `WanVideoUnit_PromptEmbedder` 未 import，运行到文本 validation 时 `NameError`
- 已修复 import

### 8 卡长期配置重置

- 收敛为：
  - `batch_size=2`
  - `save_steps=100`
  - `validation_at_start=true`
  - validation：`cfg=1 + posi_prompt.pth + 50步`
  - `wandb` 开启
- 同时移除了默认导出的 `FLASHVSR_DEBUG_DIR`

## 2026-04-12：v2 比例探针、debug overfit、16 卡训练线

### 原始 FlashVSR 比值对比与 v2 alpha 扫描

- 在原始 FlashVSR full 推理链新增默认关闭的统计导出：
  - 文件：`FlashVSR/diffsynth/pipelines/flashvsr_full.py`
  - 开关：`FLASHVSR_DEBUG_STATS=1`
  - 导出：`04_model_token_stats.json`
- 在 `infer_flashvsr_stage1_v2.py` 中新增可控注入强度参数：
  - `--projection_scale`

关键结果：

- `v2 step-500`
  - `ratio_std_lq_to_x ≈ 0.0533`
  - `ratio_absmean_lq_to_x ≈ 0.0516`
- 原始 `FlashVSR full`
  - `ratio_std_lq_to_x ≈ 0.2608`
  - `ratio_absmean_lq_to_x ≈ 0.2640`

结论：

- 当时的 `v2` 首层注入强度只有主干 token 的约 `5%`
- 原始 FlashVSR 约为 `26%`
- 两者有约 `5x` 差距

### v2 debug overfit 线

- 新增独立 debug 训练入口：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2_debug.py`
- 目标：
  - 不改现有 `v2` 主线源码
  - 只把数据入口换成固定 `sample_xxx/hr.pt` 与 `sample_xxx/lq.pt`
  - 无限循环过拟合

新增：

- 配置：
  - `configs/stage1_release_8gpu_v2_debug_overfit.yaml`
  - `configs/stage1_release_smoke_2gpu_v2_debug_overfit.yaml`
- 启动脚本：
  - `FlashVSR-Stage1-Release-8GPU-v2-Debug.sh`
  - `FlashVSR-Stage1-Release-Smoke-2GPU-v2-Debug.sh`
- 脚本 override：
  - `BATCH_SIZE_OVERRIDE`
  - `MAX_TRAIN_STEPS_OVERRIDE`
  - `LQ_PROJ_SCALE_OVERRIDE`

后续固定了 17 帧 / `bs24` / `alpha=5` 的版本，并新增 history 记录。

### 16 卡 / 双机训练线

- 建立两组母机：
  - 母机1：主机 `29rph59q8s`，从机 `w6zjf5bbw4`
  - 母机2：主机 `z9fc8ez972`，从机 `5sar6nb8vh`
- 新增 16 卡启动模板与 accelerate 模板：
  - `FlashVSR-Stage1-Release-16GPU-v2-17f-Full-bs4-lr1e5-alpha5-NoStartVal.sh`
  - `accelerate_zero2_flashvsr_2node16gpu.template.yaml`

### 16 卡 Takano-only 训练

- 由于新机器缺少本地 `storymotion` manifest，先切成 takano-only。
- 新增配置：
  - `stage1_release_16gpu_v2_17f_takano_bs4_lr1e5_alpha5_nostartval.yaml`
  - `stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_nostartval.yaml`

代表性实验：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_nostartval_20260412_042600`

结论：

- `17` 帧 / 16 卡 / takano-only 训练主链稳定可跑。

### 89 帧 16 卡线

- 母机2 上新增 89 帧版本：
  - `stage1_release_16gpu_v2_89f_takano_bs4_lr1e5_alpha5_nostartval.yaml`
- 代表性实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_89f_takano_bs4_lr1e5_alpha5_nostartval_20260412_232300`

结论：

- 89 帧启动成本显著高于 17 帧，但不是逻辑错误。

## 2026-04-13 到 2026-04-16：v3 全量微调、resume 与 dataset v2

### v3 全量微调线

- 新增：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3.py`
- 目标：
  - 与 `v2 17f takano` 对齐
  - 只把训练方式从 `LoRA + lq_proj_in` 改为 `full DIT finetune + lq_proj_in`

结论：

- `NoGC` 可训练
- `DDP + GC` 可训练
- `DeepSpeed ZeRO-2 + GC + no DS activation checkpointing` 可训练
- 一旦打开 `DeepSpeed activation checkpointing`，会出现：
  - `Trying to backward through the graph a second time`
  - 或 attention/checkpoint 重放时输入打坏

为排查这一点，新增过 `v3.1`：

- `train_flashvsr_stage1_v3_1.py`
- checkpoint 调用风格改得更接近官方 Wan closure

结论：

- `v3.1` 没有解决问题
- 问题不在早期 checkpoint 参数打包写法
- `v3.1` 已废弃，不保留为主线

当前稳定的 `v3` 组合：

- `DeepSpeed ZeRO-2`
- `gradient checkpointing`
- 不使用 `DeepSpeed activation checkpointing`

代表性实验：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v3_17f_takano_fullft_bs16_lr1e5_alpha5_nostartval_20260414_021500`

### trainer-state resume

- 在公共 runner 路径中新增完整训练状态 save/load 支持：
  - optimizer state
  - scheduler state
  - RNG state
  - step / epoch 元数据
- 保存路径：
  - `output/training_state/step-<N>/`
- 原来的 `step-<N>.safetensors` 仍保留用于推理 / validation

新增 resume 能力：

- `resume_training_state_dir`
- `resume_reset_rng_with_global_seed`

2 卡 smoke 验证：

- 母机2 主机上完成真实两阶段 smoke：
  1. 先训练到 `step-2` 并保存 full state
  2. 从 `training_state/step-2` 恢复，并换新 `global_seed`

结论：

- trainer-state resume 路径打通
- streaming dataloader 精确 cursor 延续仍不保证

### dataset v2：新 Takano / 图像 / Yubari

新增与拆层：

- `wanvideo/data/flashvsr/datasets/parquet_index.py`
- `wanvideo/data/flashvsr/datasets/source_index_v2.py`
- `wanvideo/data/flashvsr/datasets/media_reader_v2.py`
- `wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py`

新 Takano：

- 新根目录是 parquet metadata root，不是旧 tar shard root
- parquet 行内的 `path` 直接指向 clip mp4
- 已支持 direct clip path schema

图像数据：

- 路径：
  - `s3://takano-assets/20231106/high_resolution/`
- 当前以 `f=1` 单帧视频样本形式输出

Yubari：

- 最终训练入口统一收敛为：
  - `yubari_video_tar_url=conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/`
- 外部接口不再依赖 sidecar metadata root
- 内部仍利用同目录 `part-*.parquet` 做 byte-range 索引，以读取 `part-*.tar`

验证结果：

- Takano、image、Yubari 都完成了 seed reproducibility smoke
- 同 seed 一致，不同 seed 改变样本与退化

## 2026-04-20：v4 三源数据与 Aliyun 退化

### v4 数据主线

- `FlashVSR v4` 切到新的三源数据版本：
  - Takano：direct clip path parquet
  - image：metadata parquet + `TARGET_S3_PATH`
  - Yubari：`video root + part-*.parquet + part-*.tar`
- 主实验比例：
  - Takano `0.40`
  - image `0.40`
  - Yubari `0.20`

`parquet_tar_dataset_v2.py` 的关键 lazy 化：

- 启动阶段只发现 shard
- 不再全量展开所有 parquet 行
- shard 内记录读取改成 batch 流式迭代
- 三源全部按需读取

统一 conductor bridge：

- parquet
- tar/mp4/jpg
- 都不再由 dataset 在各处手工 `tmp cp`

缓存与日志：

- 缓存统一收口到对象缓存层
- 提供：
  - `CONDUCTOR_CACHE_MAX_BYTES`
  - `CONDUCTOR_CACHE_TARGET_PCT`
  - `CONDUCTOR_CACHE_TTL_SECONDS`
  - `CONDUCTOR_VERBOSITY`
- 把周期性 conductor metrics 降噪，避免淹没 `loss`

### Aliyun 退化对比实验

- 新增独立模块：
  - `wanvideo/data/flashvsr/degradation/aliyun_video_degradation.py`
- 新增配置：
  - `params_aliyun_video_compression_v1.yaml`
- degradation builder 可按 config 切换：
  - `realesrgan_with_second`
  - `aliyun_video_compression_v1`

对比实验目的：

- 不改模型结构
- 不改三源比例
- 不改学习率
- 只替换 online degradation 风格

## 2026-04-22 到 2026-04-23：v5 联合训练线

### v5 主版本线

- `v5.3`
- `v5.3.1`
- `v5.3.2`

当前理解：

- `v5.3` 与 `v5.3.1`：同结构、不同退化强度的正式对照
- `v5.3.2`：特殊实验线，不反向污染主线

### 旧实验目录

- 母机1 旧 `v5.3.2`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs16_lr1e5_aliyundegra_20260422_233000`
- 母机2 旧 `v5.3.1`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs16_lr1e5_aliyunhalf_20260422_203200`
- 母机3 旧 `v5.3`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs16_lr1e5_aliyundegra_20260422_203300`

### 围绕 `lq_proj_in` 的排查

- 检查结论：
  - `v5.3` 训练中接入的 `lq_proj_in` 结构与官方 `FlashVSR-v1.1` 基本对齐
  - `alpha=1` 时注入尺度与官方同量纲
  - 问题不在“结构写错”

ratio probe 结果：

- `v5.3 step10`
  - `ratio_std_lq_to_x = 0.0495`
  - `ratio_absmean_lq_to_x = 0.0466`
- `v5.3 step300`
  - `ratio_std_lq_to_x = 0.1053`
  - `ratio_absmean_lq_to_x = 0.0920`
- `v2 step400`
  - `ratio_std_lq_to_x = 0.2133`
  - `ratio_absmean_lq_to_x = 0.1779`
- 官方 `FlashVSR-v1.1`
  - `ratio_std_lq_to_x = 0.2887`
  - `ratio_absmean_lq_to_x = 0.2629`

结论：

- 新实验 `lq_proj_in` 不是完全没学，而是学得慢
- 主要嫌疑是 `lq_proj_in` 从零开始学，前期 LQ 支路太弱

### flash-init 重启线

统一改动：

- `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
- `zero_init_lq_proj_in=false`
- `batch_size` 从 `16` 降到 `12`

新实验目录：

- 母机1 `v5.3.2`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_20260423_150800`
  - `step=1 loss=0.321344`
  - `step=2 loss=0.109715`
  - `step=3 loss=0.193490`
- 母机2 `v5.3.1`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_20260423_154400`
  - `step=1 loss=0.208661`
- 母机3 `v5.3`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_20260423_154500`
  - `step=1 loss=0.092752`

### `v5.3 / v5.3.1` 启动报错根因

报错：

- `RuntimeError: generator raised StopIteration`
- 出现在 `tar_streaming_dataset_v53.py` 的 image 分支取样阶段

根因：

- image source 用的是：
  - `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`
- 这是 txt manifest，一行一个远端 `jpg`
- 旧代码里：
  - 配置层识别到了 `image_manifest_urls`
  - 但 image iterator 没有真正消费它

修复：

- 图像源同时支持：
  - tar 根目录
  - txt manifest 指向散图
- `streaming_dataset.py`
  - `_iterate_direct_images(...)` 现在会消费 `self.image_manifest_urls`
  - `_image_iterator(...)` 在有 `image_file_urls` 或 `image_manifest_urls` 时都会启用 direct image 路径

### fixed validation 原则修复

- validation 实际只需要视频样本
- 因此 `collect_fixed_validation_samples(...)` 改为：
  - 对 `FlashVSRStreamingDataset` 优先走 `dataset._video_iterator(rng=...)`
  - 只有在拿不到视频分支时才回退到 `iter(dataset)`

结论：

- validation 不应该再被 image branch 的 bug 连带拖死

### `v5.3.2` 的正确定位

- `v5.3.2` 是独立特殊实验线
- 正确逻辑：
  - video branch：一条正常 Yubari 视频 clip
  - image branch：从另一条 Yubari 视频里只解码 1 帧
  - 再把这 1 帧扩成 pseudo-video

用户明确要求：

- 这条特殊逻辑不影响 `v5.3 / v5.3.1`

### 文档记录约束

- 所有实验记录统一写入：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/`
- `FLASHVSR_WORKLOG.md`
  - 记录行为过程、实验目录、结果、用户反馈
- `flashvsr_v5_iteration.md`
  - 记录版本结构、技术实现、废弃原因、当前正确版本

## 2026-04-24

### `v5.3 / v5.3.1` 图像源切换到新 4k tar

- 用户提供新图像路径：
  - `s3://lucid-vr/datasets/takano_image/image/takano-image-20231106-train/4k/`
- 本机用 `conductor s3 ls` 统计：
  - 当前共有 `51168` 个 tar
  - 若按 `1 tar = 100 图` 估算，总图像数约 `5,116,800`
- 随后将以下配置中的 `image_tar_url` 统一改到该 4k tar 根目录：
  - `stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit.yaml`
  - `stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit.yaml`
  - `stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj.yaml`
  - `stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj.yaml`
  - `stage1_release_16gpu_v5_3_phase2_unfreeze_template.yaml`
  - `stage1_release_16gpu_v5_3_1_phase2_unfreeze_template.yaml`
- `v5.3.2` 未改该路径，因为它的 image branch 不依赖外部图像源。

### `freeze projector` 两阶段方案落地

- 在 `train_flashvsr_stage1_v5_3_lora.py` 中新增：
  - `--freeze_lq_proj_in`
- 行为：
  - 当该参数为 `true` 时，`lq_proj_in.parameters()` 全部 `requires_grad=False`
  - 并将 `self.pipe.lq_proj_in.eval()`
- 目的：
  - 用 `flashinit` 固定 projector
  - 先单独观察 LoRA 学习行为
  - 后续再以该阶段的 checkpoint 作为 warm-start，打开 `projector + LoRA` 一起训练

### 母机1：`v5.3.2 freezeproj`

- 已有运行中的冻结版实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600`
- 已确认：
  - `freeze_lq_proj_in=True`
  - `step=1 loss=0.321344`

### 停掉的旧实验

- 为切换到 `freeze projector + 新 4k tar image`，停掉了以下旧实验：
  - 母机2旧 `v5.3.1 flashinit`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_20260423_154400`
  - 母机3旧 `v5.3 flashinit`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_20260423_154500`

### 新成功启动的冻结版实验

- 母机2 `v5.3.1 freezeproj`
  - 首次启动（4k tar image，后续废弃）
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_014800`
  - 回退到原 manifest 后的当前有效目录
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_023200`
  - 已确认：
    - `freeze_lq_proj_in=True`
    - `step=1 loss=0.184871`
- 母机3 `v5.3 freezeproj`
  - 首次启动（4k tar image，后续废弃）
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_014900`
  - 回退到原 manifest 后的当前有效目录
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_023300`
  - 已确认：
    - `freeze_lq_proj_in=True`
    - `step=1 loss=0.139753`

### 4k tar image 路径回退

- 用户在切到新图像 tar 根目录后，判断该版本“不太对劲”。
- 因此将 `v5.3 / v5.3.1` 的 `image_tar_url` 回退到：
  - `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`
- 回退后重新停掉了 4k 版本实验，并用原 manifest 重新启动 `freezeproj`。

### 最终决定：图像源固定使用 tar

- 在重新对比后，最终不再纠结 manifest 路线。
- 当前决定：
  - `v5.3 / v5.3.1` 之后统一使用 tar 图像源：
    - `s3://lucid-vr/datasets/takano_image/image/takano-image-20231106-train/4k/`
- 因此又停掉了 manifest 版：
  - `v5.3.1`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_023200`
  - `v5.3`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_023300`
- 重新启动后的当前有效目录：
  - 母机2 `v5.3.1 freezeproj tar`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100`
    - `step=1 loss=0.216261`
  - 母机3 `v5.3 freezeproj tar`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200`
    - `step=1 loss=0.340537`

### phase2 模板准备完成

- 已补好后续两模块一起训练的模板配置：
  - `stage1_release_16gpu_v5_3_phase2_unfreeze_template.yaml`
  - `stage1_release_16gpu_v5_3_1_phase2_unfreeze_template.yaml`
  - `stage1_release_16gpu_v5_3_2_phase2_unfreeze_template.yaml`
- 已补好模板启动脚本：
  - `FlashVSR-Stage1-Release-16GPU-v5-3-Phase2-Unfreeze-Template.sh`
- 模板约定：
  - 不直接恢复 `training_state`
  - 改走新的 warm-start 实验
  - 启动时显式传：
    - `STAGE1_CKPT=<满意的 freezeproj checkpoint>`
  - 让 `projector + LoRA` 同时打开继续训

### 2026-04-26：定位 `v5.3.2 phase2` 初始化不对的问题

- 核对实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426_20260425_150040`
- 直接检查 `run.log` 与 `resolved_args.yaml` 后确认：
  - 该实验确实读取了
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600/output/step-2000.safetensors`
  - 但日志只打印了：
    - `Stage1 resume loaded LoRA from ...step-2000.safetensors`
  - 没有打印 `Stage1 resume loaded lq_proj_in ...`
- 进一步直接检查 `step-2000.safetensors` 的 key：
  - `num_keys = 480`
  - `num_lora = 480`
  - `num_lq = 0`
- 结论：
  - `freezeproj` 阶段导出的 checkpoint 只保存了 LoRA，没有保存 `projector`
  - 因此当前这条 `phase2` 实际上是：
    - `LoRA` 从 `step-2000` 继承
    - `projector` 重新初始化
  - 这也是为什么前 10 step 视觉结果明显不如冻结版 `step-2000`

### 2026-04-26：修复 phase2 初始化与导出逻辑

- 修复 1：`phase2` 显式补回 FlashVSR projector 初始化
  - 修改文件：
    - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426.yaml`
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-2-Phase2-Unfreeze-FromStep2000-Seed20260426.sh`
  - 新行为：
    - `resume_stage1_checkpoint` 继续加载 `LoRA`
    - 同时显式指定：
      - `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
  - 这样 phase2 变成：
    - `LoRA` 继承 phase1
    - `projector` 继承 FlashVSR v1.1

- 修复 2：允许 `resume_stage1_checkpoint + lq_proj_checkpoint` 同时出现
  - 修改文件：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - 原先 parser 会禁止这个组合
  - 现在只继续禁止：
    - `resume_stage1_checkpoint + lora_checkpoint`

- 修复 3：stage1 checkpoint 导出时强制带上 `pipe.lq_proj_in.*`
  - 修改文件：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - 通过重写 `FlashVSRStage1TrainingModule.export_trainable_state_dict(...)`
  - 即使 `freeze_lq_proj_in=true`，导出时也会把 `pipe.lq_proj_in.*` 一起写入
  - 目的：
    - 后续新的 freezeproj checkpoint 不再只剩 LoRA
    - 后面再做 phase2 时，可以真正从同一个 stage1 ckpt 同时恢复 `LoRA + projector`

- 调整打点位置：去掉 `tar_v53` 数据 discovery 阶段的大量打印，改为只在训练主路径首轮打印关键阶段
  - 修改文件：
    - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`
    - `diffsynth/diffusion/runner.py`
  - 去掉内容：
    - `init_begin / super_init_begin / discover_yubari_begin / discover_takano_begin ...`
  - 新增内容：
    - `resume_state_load_begin/end`
    - `epoch_begin`
    - `progress_bar_created`
    - `data_iterator_created`
    - `first_batch_fetch_begin/end`
    - `first_forward_begin/end`
    - `first_backward_end`
    - `first_optimizer_logger_end`
  - 目的：
    - 既然 `manifest` 已经解决 Takano discovery 慢的问题，后续重点转为定位 `validation callback ready` 之后为什么有的实验进入首个 `loss` 很慢

### 2026-04-26：确认 `v5.3 / v5.3.1` 双机 resume 卡住的真实原因并补齐从机状态

- 背景：
  - 旧母机上的 `v5.3.1` 与 `v5.3` 需要在新六机组上继续从 `step-1300` resume
  - 一开始只从 `s3://lxh/artifacts/...` 拉回实验目录，结果新从机缺少 DeepSpeed ZeRO 的 `rank8-15` optimizer / random state
  - 因此训练不是通信坏了，而是在 DeepSpeed load checkpoint 时缺从机分片
- 已确认的旧从机来源：
  - `v5.3.1` 旧从机：
    - `s3://bolt-prod-2320845741/tasks/5sar6nb8vh/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100`
  - `v5.3` 旧从机：
    - `s3://bolt-prod-2320845741/tasks/be46z26b5v/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200`
- 新从机搬运目标：
  - `v5.3.1` 新从机 `hj65iqg9rh`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100`
  - `v5.3` 新从机 `xwk6qjuej5`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200`
- 已验证：
  - 两个新从机的 `output/training_state/step-1300/` 下，`rank8-15` 的 `bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt` 已齐
  - 两个新从机的 `random_states_8.pkl` 到 `random_states_15.pkl` 已齐
  - 两个新从机的 `flashvsr_training_state.json`、`scheduler.bin`、`pytorch_model/mp_rank_00_model_states.pt` 已齐
  - 手动补了 `output/training_state/latest = step-1300`，避免后续脚本误用 latest 时失败
- 结论：
  - 这次 resume 的核心修复不是换机器组合，也不是改 barrier
  - 正确流程是：主机保留 rank0-7 状态，从机必须从旧从机 task bucket 补齐 rank8-15 状态后再启动

### 2026-04-26：`kh5idf7f98 + hj65iqg9rh` 纯通信 smoke

- 目的：
  - 排除 `v5.3.1` resume 卡住是否由这对节点本身的 `NCCL / TCPStore` 通信异常导致
- 测试方式：
  - 不加载模型
  - 不读数据
  - 只运行 2 节点 16 rank `torchrun`
  - 流程为 `init_process_group(nccl)` -> `barrier` -> `all_reduce`
- 日志目录：
  - `/mnt/task_wrapper/user_output/artifacts/tmp/ddp_smoke_kh_hj/kh_rank0_29701.log`
  - `/mnt/task_wrapper/user_output/artifacts/tmp/ddp_smoke_kh_hj/hj_rank1_29701.log`
- 结果：
  - `kh` 侧 rank0-7 全部完成 `barrier`
  - `hj` 侧 rank8-15 全部完成 `barrier`
  - 16 个 rank 的 `all_reduce` 结果均为 `136.0`
  - smoke 结束后两边 GPU 显存均回到 0
- 结论：
  - `kh/hj` 这对节点的基础 `torchrun + NCCL + TCPStore` 通信可用
  - `v5.3.1` 卡住不应继续归因于裸通信链路
  - 后续应继续查训练/DeepSpeed resume 路径里的同步点或状态加载顺序

### 2026-04-26：`v5.3.1` resume 精确定位与修复

- 目标：
  - 判断 `v5.3.1` 从 `step-1300` resume 卡住是否由 checkpoint 文件损坏、从机状态缺失、TCPStore/NCCL 通信异常，或数据初始化死锁导致
- 新增诊断：
  - 在 `diffsynth/diffusion/runner.py` 的 `load_training_state()` 加入 `FLASHVSR_RESUME_DEBUG=1` 控制的文件级打印
  - 打印每个 rank 的 `flashvsr_training_state.json`、`scheduler.bin`、`random_states_{rank}.pkl`、`mp_rank_00_model_states.pt`、`bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt` 是否存在、大小和 mtime
  - 包装 `torch.load()`，定位每个状态文件的 `torch_load_begin/end/error`
  - 在 `train_flashvsr_stage1_v5_3_lora.py` 增加 dataset/model/validation/runner 关键阶段打印
  - 在 `streaming_dataset.py` 与 `tar_streaming_dataset_v53.py` 增加 remote discovery / manifest / broadcast 关键阶段打印
- 发现的问题：
  - `hj65iqg9rh` 上存在旧的 `/tmp/flashvsr_remote_discovery/*.lock`，会导致部分 discovery 等待；已补 stale lock 自动清理
  - 更核心的问题是分布式 discovery 的 cache 分歧：
    - `kh5idf7f98` 上本地 image discovery cache 命中，global rank0 直接返回，没有进入 `dist.broadcast_object_list`
    - `hj65iqg9rh` 上本地 cache 未命中，rank8-15 进入 `discover_dist_broadcast_begin` 等待 rank0 广播
    - rank0 没有广播，导致从机永久等待
  - 这不是 checkpoint 文件损坏，也不是裸 `NCCL/TCPStore` 不通
- 修复：
  - `streaming_dataset._discover_urls()` 在 `torch.distributed` 已初始化且 world size > 1 时，所有 rank 必须进入同一条 rank0 broadcast 路径
  - rank0 可以读本地 discovery cache 或远程 list，但必须把结果广播给所有 rank
  - 非 rank0 不再因为自己的本地 cache 状态不同而绕开或独立进入别的路径
  - 正常非分布式场景仍保留本地 cache 快速返回
  - `gradient_checkpoint_inputs` 打印从 `FLASHVSR_TRAIN_DEBUG` 拆到 `FLASHVSR_GC_DEBUG`，避免查 resume 时被 GC shape 刷屏
- 验证实验：
  - `v5.3.1` resume 目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_resume_step1300_seed2026042601_manifest_20260426_161300`
  - 主从机：
    - `kh5idf7f98 + hj65iqg9rh`
  - 结果：
    - `rank0-15` 均完成 `accelerator_load_state_end`
    - `rank0-15` 均完成 metadata broadcast，读到 `step=1300`
    - 训练进入 `step=1301`
    - 第一条 loss：
      - `step=1301 loss=0.060703`
    - GPU 显存和利用率恢复到训练态，主机约 `123GB/GPU`，利用率接近 `99-100%`
- 结论：
  - 本次卡住的直接原因是多机数据 discovery cache 分歧造成的 broadcast 死锁
  - checkpoint 读写链路通过文件级打印确认是完整的
  - 后续如果再查 resume，优先打开 `FLASHVSR_RESUME_DEBUG=1`；如果要查 GC shape，单独打开 `FLASHVSR_GC_DEBUG=1`

### 2026-04-26：清理 `v5.3.1` resume 定位打印，保留必要修复

- 背景：
  - `v5.3.1` resume 已经确认不是 checkpoint 损坏，而是多机 discovery cache 分歧导致的 broadcast 死锁
  - 定位过程中加入的 `main_debug`、`tar_v53_debug`、GC shape 打印会淹没正常 loss，不适合长期留在默认训练路径
- 清理：
  - 删除 `train_flashvsr_stage1_v5_3_lora.py` 中一次性的 `main_debug(...)` 阶段打印
  - 删除 `tar_streaming_dataset_v53.py` 中一次性的 `_v53_debug(...)` 初始化和 discovery 阶段打印
  - `resume_debug` 只在显式设置 `FLASHVSR_RESUME_DEBUG=1` 时启用；默认不打印、不包裹 `torch.load()`、不做文件 stat 预检查
  - GC tensor shape 打印从 `FLASHVSR_TRAIN_DEBUG` 拆到 `FLASHVSR_GC_DEBUG`，默认关闭
- 保留：
  - `streaming_dataset._discover_urls()` 的分布式 rank0 broadcast 修复
  - stale discovery lock 自动清理
  - 这两个属于 correctness 修复，不是临时打印
- 影响：
  - 正常训练默认不会新增日志
  - rank0 broadcast 只发生在数据源 discovery 初始化阶段，不进入每个 step 的训练循环
  - stale lock 检查只在发现已有 lock 时触发，正常路径几乎没有额外开销

### 2026-04-27：`v5.3.1 / v5.3` 从冻结 projector 阶段切入阶段2

- 背景：
  - 已确认以下两个冻结 `lq_proj_in` 的阶段1实验效果可继续推进：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_resume_step1300_seed2026042601_manifest_20260426_161300`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_resume_step1300_seed2026042602_manifest_20260426_152600`
  - 两个实验均选用最新 `step-1500.safetensors` 作为阶段2初始化
  - 阶段2目标是解冻 `lq_proj_in`，让 LoRA 与 projection 一起训练
- 新增配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427.yaml`
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427.yaml`
- 新增启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-1-Phase2-Unfreeze-FromStep1500-Seed20260427.sh`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Phase2-Unfreeze-FromStep1500-Seed20260427.sh`
- 公共设置：
  - `num_frames=17`
  - `height=768`
  - `width=1280`
  - `per_device_batch_size=12`
  - `learning_rate=1e-5`
  - `max_train_steps=10000`
  - `save_steps=100`
  - `freeze_lq_proj_in=false`
  - `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
  - `resume_stage1_checkpoint` 指向阶段1的 `step-1500.safetensors`
  - 使用 `manifest` 版 Takano video 列表，避免重新扫描大目录
- 主从机权重准备：
  - `v5.3.1` 的 `step-1500.safetensors` 从 `kh5idf7f98` 上传到 `s3://lxh/tmp/phase2_v531_step1500.safetensors`，再下载到 `hj65iqg9rh` 的同一路径
  - `v5.3` 的 `step-1500.safetensors` 从 `zhki5rrddw` 上传到 `s3://lxh/tmp/phase2_v53_step1500.safetensors`，再下载到 `xwk6qjuej5` 的同一路径
- 新实验启动：
  - `v5.3.1 phase2`
    - 主从机：`kh5idf7f98 + hj65iqg9rh`
    - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427_20260427_001300`
    - 第一条 loss：`step=1 loss=0.033236`
    - 第二条 loss：`step=2 loss=0.063895`
    - 主机显存约 `127GB/GPU`
  - `v5.3 phase2`
    - 主从机：`zhki5rrddw + xwk6qjuej5`
    - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427_20260427_001300`
    - 第一条 loss：`step=1 loss=0.032787`
    - 主机显存约 `128GB/GPU`
    - 检查时 GPU 利用率为 `100%`
- 结论：
  - 两个阶段2实验均已进入训练循环，不是数据初始化或通信卡住
  - 当前阶段2启动流程有效：先确认主从机都能读到 `step-1500.safetensors`，再启动双机 16 卡训练

### 2026-04-27：启动 `v5.3.3` 随机 projector 对照实验

- 目的：
  - 对照当前 `v5.3` 的 `FlashVSR projector 初始化 + 冻结阶段 + 解冻阶段2` 路线
  - 新实验不做 phase1/phase2 拆分，从一开始就让随机初始化的 `lq_proj_in` 与 DiT LoRA 一起训练
- 新增配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj.yaml`
- 新增启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-3-Lora-17f-FullSources-bs12-lr1e5-AliyunDegra-RandomProj.sh`
- 关键差异：
  - 不传 `--lq_proj_checkpoint`
  - 不传 `--resume_stage1_checkpoint`
  - 显式传 `--zero_init_lq_proj_in false`
  - 因此 `lq_proj_in` 走随机初始化，而不是 FlashVSR 原版初始化，也不是 zero-init output
  - DiT LoRA 仍按普通新实验方式初始化
- 其他训练设置：
  - `dataset_mode=tar_v53`
  - Takano 使用 manifest：`/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt`
  - Yubari：`conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/`
  - Image：`s3://lucid-vr/datasets/takano_image/image/takano-image-20231106-train/4k/`
  - `num_frames=17`
  - `height=768`
  - `width=1280`
  - `batch_size=12`
  - `learning_rate=1e-5`
  - Aliyun full 退化：`params_aliyun_video_compression_v1.yaml`
  - `use_gradient_checkpointing=true`
  - `use_gradient_checkpointing_offload=false`
- 机器：
  - 主机：`dd79sgu25m`
  - 从机：`myj7ukyewz`
  - 主机地址：`240.12.239.214`
  - 端口：`29733`
- 启动前处理：
  - 两台机器上存在占卡程序，每张卡约 `166GB` 显存；已停止
  - 两台机器均补齐 manifest：
    - 从 `s3://lxh/data/mainfest/takano_video_train_all.txt`
    - 下载到 `/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt`
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_20260427_130300`
- 启动结果：
  - wandb 已开启
  - 进入训练态后显存约 `120GB/GPU`
  - 第一条 loss：
    - `step=1 loss=0.350427`
  - 第二条 loss：
    - `step=2 loss=0.174317`
- 初步观察：
  - 随机 projector 的初始 loss 明显高于 `v5.3 / v5.3.1 phase2` 的约 `0.03`
  - 这符合预期：`v5.3.3` 没有继承 FlashVSR projector，也没有继承冻结阶段得到的 projector/LoRA 状态

### 2026-04-27：在 `fe5q8vne4p` 生成固定规则推理测试集

- 新增 4 卡测试机器：
  - `fe5q8vne4p`
  - 用户已提前完成代码同步与远程 `tmux lxh` / `watch` 窗口准备
- 固定测试集生成规则：
  - 使用 Aliyun 退化逻辑
  - 去掉 Aliyun 退化最后一步 bicubic 上采样恢复
  - 因此 `LQ` 保持为 `GT` 的 1/4 尺寸
  - 当前标准输出为：
    - `GT=1280x768`
    - `LQ=320x192`
    - `num_frames=17`
    - `fps=8`
- 新增生成脚本：
  - `wanvideo/data/flashvsr/tests/export_inference_testset6_aliyun_x4_lq.py`
  - `wanvideo/data/flashvsr/tests/run_export_inference_testset6_aliyun_x4_lq.sh`
- 生成数据：
  - 3 个 Takano 视频
  - 3 个 Yubari 视频
- 输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq_20260427`
- 输出内容：
  - `gt/`
    - `takano_00_gt.mp4`
    - `takano_01_gt.mp4`
    - `takano_02_gt.mp4`
    - `yubari_00_gt.mp4`
    - `yubari_01_gt.mp4`
    - `yubari_02_gt.mp4`
  - `lq/`
    - `takano_00_lq.mp4`
    - `takano_01_lq.mp4`
    - `takano_02_lq.mp4`
    - `yubari_00_lq.mp4`
    - `yubari_01_lq.mp4`
    - `yubari_02_lq.mp4`
  - `summary.json`
- 校验结果：
  - 6 个 `GT` 视频均为 `1280x768, 17 frames, 8 fps`
  - 6 个 `LQ` 视频均为 `320x192, 17 frames, 8 fps`
- 过程中修复：
  - Takano 使用本地 `.txt` manifest 时，`FlashVSRStreamingDataset` 的 manifest-only 路径不会自动进入 `_video_iterator`
  - 生成脚本中增加 manifest 展开逻辑，将本地 manifest 抽样展开为逗号分隔的 tar URL 列表后再交给 dataset

### 2026-04-27：同步并处理真实 challenging 测试集

- 本机原始目录：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/challenging_test_lxh`
- 原始视频数量：
  - 11 个
- 原始视频云端备份：
  - `s3://lxh/artifacts/inference/challenging_test_lxh`
- 远端处理机器：
  - `fe5q8vne4p`
- 远端原始视频目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/challenging_test_lxh`
- 处理规则：
  - 保持比例缩放
  - 缩放到至少覆盖 `320x192`
  - 长边中心裁剪到 `320x192`
  - 只保留前 17 帧
  - 不做拉伸变形
  - 统一重编码为 `8 fps`
- 远端处理后目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/challenging_test_lxh_17f_320x192`
- 处理后云端目录：
  - `s3://lxh/artifacts/inference/challenging_test_lxh_17f_320x192`
- 校验结果：
  - 11 个处理后视频均为 `320x192`
  - 11 个处理后视频均为 `17` 帧
  - 11 个处理后视频均为 `8 fps`

### 2026-04-27：修复真实测试集 fps 后重跑 v5 / FlashVSR / SeedVR3B 对比

- 修复点：
  - 真实 challenging 测试集初版误保留原视频 fps
  - 已重新覆盖生成并同步为 `8 fps`
  - 云端前缀 `s3://lxh/artifacts/inference/challenging_test_lxh_17f_320x192` 曾被错误同步进部分推理结果，已清理后重新同步纯测试集文件
- 重测机器：
  - `fe5q8vne4p`
- 重测输入：
  - 合成测试集：
    - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset6_17f_aliyun_x4_lq_20260427/lq`
  - 真实测试集：
    - `/mnt/task_wrapper/user_output/artifacts/inference/challenging_test_lxh_17f_320x192`
- 重测模型：
  - `v5.3 step-1300`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200/output/step-1300.safetensors`
  - `v5.3.1 step-1300`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100/output/step-1300.safetensors`
  - `v5.3.2 step-2000`
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600/output/step-2000.safetensors`
  - FlashVSR 官方模型：
    - `/mnt/models/FlashVSR-v1.1`
  - SeedVR3B：
    - `/mnt/models/SeedVR-3B`
- 推理设置：
  - `fps=8`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
  - LoRA checkpoint 只含 LoRA 时，推理侧显式加载 FlashVSR projector：
    - `/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
- 原始输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427`
  - 布局为方法外层、数据集内层
- 整理后输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset`
  - 布局为数据集外层、方法内层：
    - `synthetic/<method>/*.mp4`
    - `real/<method>/*.mp4`
- 完成数量：
  - 总计 `85` 个输出视频
  - 每个 LoRA / FlashVSR / SeedVR3B 都包含：
    - synthetic `6` 个
    - real `11` 个
- 日志整理：
  - 已清理输出视频目录中的逐视频 `.log`
  - 后续脚本改为将日志集中写入顶层 `logs/`
- 遗留问题：
  - 当前 LoRA / FlashVSR wrapper 仍然是“每个视频启动一次推理进程”，因此会重复加载 Wan/VAE/projector
  - 这次重测尚未修复该性能问题
  - 正确修法是新增 batch-directory inference 入口：模型只加载一次，然后在同一进程内循环处理目录内视频

### 2026-04-27：修复 LoRA 对比推理的重复加载与输出目录结构

- 问题：
  - `run_compare_v5_lora_flash_seedvr3b_20260427.sh` 旧版在 LoRA 分支中对每个视频单独调用一次 `infer_flashvsr_stage1_v2.py`
  - 因此每个视频都会重新加载 Wan / VAE / LQ projector / LoRA
  - 对当前 3 个 LoRA、2 个数据集、共 17 个输入视频的对比来说，LoRA 部分会触发 `51` 次模型加载
- 修复：
  - 新增批量推理入口：
    - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2_batch.py`
  - 将单视频推理公共逻辑从 `infer_flashvsr_stage1_v2.py` 拆出：
    - `add_common_args`
    - `build_flashvsr_stage1_pipe`
    - `run_single_video`
  - batch 入口对每个 checkpoint / dataset 只加载一次模型，然后循环处理 `input_dir/*.mp4`
  - LoRA 对比脚本改为调用 batch 入口
- 影响：
  - LoRA 部分模型加载次数从 `51` 次降为 `6` 次：
    - 3 个 LoRA checkpoint
    - 每个 checkpoint 分别跑 synthetic / real 两个数据集
  - 输出目录不再生成旧的“方法外层、数据集内层”布局
  - 默认输出目录改为：
    - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset`
  - 新布局固定为：
    - `synthetic/<method>/*.mp4`
    - `real/<method>/*.mp4`
    - `logs/*.log`
- 日志清理：
  - LoRA 每个方法 / 数据集只写一个集中 log：
    - `logs/<method>_<dataset>.log`
  - 不再在视频输出目录旁边生成逐视频 `.log`
  - SeedVR wrapper 增加 `LOG_FILE` 参数，允许日志写到顶层 `logs/`
- 后续补充修复：
  - 新增 FlashVSR 官方目录级 padded wrapper：
    - `wanvideo/model_inference/flashvsr/infer_flashvsr_full_cloud_padded_dir.py`
  - `run_flashvsr_full_dir_20260421.sh` 改为：
    - 先批量补帧到临时目录
    - 调用官方 `infer_flashvsr_full_cloud.py --input_path <dir>` 一次
    - 再逐个裁回原始帧数并输出 `<sample>_sr.mp4`
  - 因此 FlashVSR 官方分支也从每视频重复加载，改为每个数据集加载一次
- 当前状态：
  - LoRA 分支：每个 checkpoint / dataset 加载一次
  - FlashVSR 官方分支：每个 dataset 加载一次
  - SeedVR3B 分支：原本就是目录级推理，每个 dataset 加载一次
## 2026-04-27

### Stage 2 / v6 训练代码启动

- 阅读并整理 FlashVSR 论文 Stage 2 相关内容，形成文档：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/doc/flashvsr_stage2_v6_design.md`
- 对齐 FlashVSR 官方推理代码中的 causal / block-sparse 设计：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/diffsynth/models/wan_video_dit.py`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/diffsynth/pipelines/flashvsr_full.py`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/examples/WanVSR/utils/utils.py`
- 新增 v6 Stage 2 attention 支持：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/diffsynth/models/wan_video_dit_stage2_v6.py`
  - 当前包含 `dense_time_causal` correctness fallback 和 `block_causal_all_past` 官方 block-sparse kernel 接口。
- 新增 v6 Stage 2 video-only LoRA 训练入口：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py`
- 新增 2 卡 smoke 配置和启动脚本：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_densecausal.yaml`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-DenseCausal.sh`
- 关键判断：
  - Stage 2 不能用普通 `is_causal=True` 代替，因为 WAN token flatten 后同一帧内空间 token 会被错误按序遮挡。
  - v6 第一版用 time-aware dense causal mask 做 correctness baseline。
  - 正式训练应继续切到 FlashVSR 官方 `(2,8,8)` block-sparse kernel，并补齐 top-k block selection。
- 更正 block-sparse 对齐方式：
  - 之前考虑过在 attention 内部为 odd latent-time 补 1 个 latent block 再裁回。
  - 重新核对官方推理后确认，FlashVSR 官方不是这么处理 89 帧的。
  - 官方推理通过 streaming chunk 避开 odd latent-time：首段 `f=6`，后续 `f=2`，每次进入 `(2,8,8)` block partition 的 temporal 维都是偶数。
  - 已把 v6 block-sparse 路线改成 `block_streaming_causal`，不再把 padding 作为正式方案。

### 2026-04-27：二阶段 checkpoint 对比测试重跑

- 测试目标：
  - 在测试机 `fe5q8vne4p` 上重测 3 个二阶段 LoRA checkpoint。
  - 输出目录：
    - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset`
- checkpoint：
  - `v5.3.2 phase2 step-1200`
    - `/mnt/task_wrapper/user_output/artifacts/ckpts/compare_20260427_phase2/v5_3_2_phase2_step1200.safetensors`
  - `v5.3.1 phase2 step-600`
    - `/mnt/task_wrapper/user_output/artifacts/ckpts/compare_20260427_phase2/v5_3_1_phase2_step600.safetensors`
  - `v5.3 phase2 step-600`
    - `/mnt/task_wrapper/user_output/artifacts/ckpts/compare_20260427_phase2/v5_3_phase2_step600.safetensors`
- 重要修正：
  - 第一轮误用了 FlashVSR 官方 `LQ_proj_in.ckpt` 作为 projector。
  - 错误结果保留为 `_flashproj` 后缀，方便对照。
  - 已重新启动正确版本，不再传 `--lq_proj_checkpoint`，由 checkpoint 内部的 `lq_proj_in.*` 参数加载 projector。
  - 日志确认每个 checkpoint 均为：
    - `lq_proj_keys=8`
    - `lora_keys=480`
- 推理设置：
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
  - `fps=8`
  - `num_frames=17`
  - `num_inference_steps=50`
  - `projection_scale=1.0`
- 正确结果：
  - 方法后缀统一为 `_selfproj`
  - synthetic：每个 checkpoint `6` 个视频
  - real：每个 checkpoint `11` 个视频
  - 总计 `_selfproj` 输出 `51` 个视频
- 日志：
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v5_lora_flash_seedvr3b_20260427_by_dataset/logs/phase2_lora_compare_selfproj_settings.log`
  - 完成标记：
    - `[phase2-lora done]`

### 2026-04-27：启动 v5.3.4 89 帧 random projector 对照实验

- 目标：
  - 将原 `v5.3.3 randomproj` 对照线改成 89 帧版本。
  - 保持 image branch 仍为固定 5 帧 pseudo-video，不随 89 帧视频动态变成 23 帧。
- 机器：
  - 主机 `dd79sgu25m`
  - 从机 `myj7ukyewz`
- 停止旧实验：
  - 停掉旧的 `v5.3.3 17f bs12 randomproj`：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_20260427_130300`
- 新增非破坏式代码参数：
  - `image_branch_num_frames`
  - 默认不设时保持旧逻辑：按 `num_frames` 计算 image pseudo-video 长度。
  - 本次 `v5.3.4` 显式设为 `5`。
- 新增配置与脚本：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5.yaml`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-4-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5.sh`
- 核心设置：
  - `num_frames=89`
  - `image_branch_num_frames=5`
  - `batch_size=1`
  - `zero_init_lq_proj_in=false`
  - `lq_proj_checkpoint` 不设置，即 projector 随机初始化，不使用 FlashVSR projector 初始化。
- 新实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_20260427_225834`
- 启动状态：
  - 双机 16 卡启动成功。
  - 已出第一条 loss：
    - `step=1 loss=0.871739`
- 注意：
  - 日志出现 `num_frames % 4 != 1. We round it up to 25.`，训练未中断。
  - 该提示需要后续确认是否只是 downstream shape check 的打印，还是有路径仍把 branch packed 后的长度送入普通 Wan shape checker。

### 2026-04-28：DataLoader worker / prefetch 配置补齐

- 背景：训练 GPU 利用率呈现一阵 `100%` 一阵 `0%`，怀疑数据端只有一个 worker 或没有后台预取。
- 排查结论：当前大量 v5/v6 配置仍为 `dataset_num_workers: 0`，实际含义是每个 rank 的训练主进程同步完成远端读取、视频解码、退化和 batch 组装，没有 DataLoader 后台 worker 队列。
- 代码改动：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/diffsynth/diffusion/parsers.py`
    - 已加入 `dataloader_prefetch_factor`
    - 已加入 `dataloader_persistent_workers`
    - 已加入 `dataloader_pin_memory`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/diffsynth/diffusion/runner.py`
    - DataLoader 现在会在 `dataset_num_workers > 0` 时接入 `prefetch_factor` 和 `persistent_workers`。
    - `pin_memory` 作为可选参数接入。
    - `dataset_num_workers == 0` 时不传 PyTorch 不允许的 worker-only 参数，保持旧行为。
- 验证：
  - `python -m py_compile diffsynth/diffusion/parsers.py diffsynth/diffusion/runner.py` 通过。
- 建议后续实验：
  - 先用 `dataset_num_workers: 2`、`dataloader_prefetch_factor: 2`、`dataloader_persistent_workers: true`。
  - 不建议直接每 rank 开 8 个 worker；16 卡下会变成 128 个 worker，可能反过来打爆 CPU / conductor / cache。

### 2026-04-28：DataLoader worker 无法生效问题定位与修复

- 问题：用户反馈训练中 GPU 利用率呈现 `100% / 0%` 交替，怀疑数据处理只有一个 worker；进一步要求定位为什么 `dataset_num_workers=1` 也接不上。
- 排查机器：测试机 `fe5q8vne4p`。
- 临时测试脚本：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/data/flashvsr/tests/benchmark_v53_dataloader_workers.py`
  - 该脚本只用于 dataset/DataLoader benchmark，不进入 Wan/VAE/DiT 训练。
- 关键发现：
  - 不是 `DataLoader num_workers=1` 本身卡住。
  - 原 smoke 配置在 dataset 构造阶段就卡住，尚未创建 DataLoader。
  - `FLASHVSR_TRAIN_DEBUG=1` 定位到卡点在远端 discovery：
    - `conductor://lucid-vr/datasets/takano_original/video/takano-video-20231214-train/4k/`
  - 该路径存在 stale discovery lock，随后进入远端 list，导致 90s 内无法完成。
- 验证：
  - 将 Takano 输入改为已生成 manifest：
    - `/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_train_all.txt`
  - 构造 dataset 后，`workers=0` 可以正常出 batch：
    - 首 batch `8.8428s`
    - 第二 batch `4.4047s`
  - `workers=1 + prefetch_factor=2 + persistent_workers=True` 可以正常出 batch：
    - 首 batch `7.4420s`
    - measured mean `4.0889s`
    - min `1.4950s`
- 结论：
  - worker 开不起来的根因是 dataset 初始化阶段扫远端目录太慢，不是 PyTorch DataLoader worker 机制不可用。
  - 后续正式训练不应在 config 里直接给四个 Takano 远端目录，优先使用 manifest。
- 已改未来启动用配置，不影响已运行实验目录内的 snapshot：
  - `stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj.yaml`
  - `stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj.yaml`
  - `stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj.yaml`
  - `stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5.yaml`
- 新默认：
  - `dataset_num_workers: 1`
  - `dataloader_prefetch_factor: 2`
  - `dataloader_persistent_workers: true`
  - v5.3 / v5.3.1 的 `takano_video_tar_url` 改为 manifest 路径。
- 测试结束后已在 `fe5q8vne4p` 重新启动占卡程序：
  - `bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`

### 2026-04-28：DataLoader worker 退化集中到 0 卡的问题修复

- 问题：正式训练打开 `dataset_num_workers=2` 后，用户观察到 0 卡显存异常升到约 `181267MiB / 183359MiB`，其他卡只是缓慢增长，怀疑所有 worker 的退化都在 0 卡上执行。
- 根因：
  - 退化模块在 dataset 初始化阶段提前构造 CUDA model。
  - DataLoader worker 进程启动后会复制 dataset 对象；如果不显式设置 CUDA device，worker 中的退化上下文容易落到默认 `cuda:0`。
  - 因此每个 rank 的 worker 都可能在 0 卡上创建 basicsr/diffjpeg 退化上下文，导致 0 卡先爆。
- 代码修复：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/data/flashvsr/degradation/__init__.py`
    - `build_degradation_model(config_path=None, device=None)` 支持显式传入设备。
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/data/flashvsr/datasets/streaming_dataset.py`
    - `ConsistentClipDegradation` 改为 lazy init，每个进程首次退化时才按 `LOCAL_RANK` 构造退化 model。
    - `__getstate__()` 清空 `model/model_pid`，避免 DataLoader worker 继承主进程里的 CUDA model。
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py`
    - 同步加入 lazy init 和 per-process CUDA 绑定。
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/diffsynth/diffusion/runner.py`
    - 新增 `_set_worker_cuda_device()`，DataLoader worker 启动时执行 `torch.cuda.set_device(int(LOCAL_RANK))`。
    - worker 模式使用可 pickle 的 `_first_item_collate`，避免 `spawn` 下 lambda 无法 pickle。
- 本地验证：
  - `python3 -m py_compile diffsynth/diffusion/runner.py wanvideo/data/flashvsr/datasets/streaming_dataset.py wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py wanvideo/data/flashvsr/degradation/__init__.py` 通过。
- 测试机验证：
  - 机器：`fe5q8vne4p`
  - 4 卡 smoke 目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/flashvsr_worker2_rankcuda_4gpu_v532frame_20260427_121849`
  - 设置：
    - `dataset_num_workers=2`
    - `dataloader_prefetch_factor=2`
    - `dataloader_persistent_workers=true`
    - `dataloader_multiprocessing_context=spawn`
    - `dataloader_in_order=false`
  - 结果：
    - `step=1 loss=0.291113`
    - `step=2 loss=0.276561`
    - `step=3 loss=0.199451`
    - 未再出现 0 卡单独 OOM。
- 正式实验重启结果：
  - `v5.3.2`：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426_20260428_032300`
    - 已出 loss：`0.081352`, `0.085281`, `0.061806`
  - `v5.3`：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427_20260428_032300`
    - 已出 loss：`0.077901`, `0.102579`
  - `v5.3.4`：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_20260428_032300`
    - 已出 loss：`1.073209`, `0.430661`, `1.173554`, `1.154215`, `0.834494`
  - `v5.3.1`：
    - 首次重启失败原因不是新逻辑无效，而是 kh 主机没有同步到新代码，日志仍停在旧版 `streaming_dataset.py:188 return self.model.degrade_batch_consistent(...)`。
    - 已通过 `bolt task scp` 手动补齐 kh 主机最小文件集。
    - 新目录：
      - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427_20260428_033500`
    - 已出 loss：
      - `step=1 loss=0.058888`
      - `step=2 loss=0.070909`
      - `step=3 loss=0.094328`
    - 显存分布约 `167-173GB/GPU`，不再是 0 卡单独吃满。
- 当前结论：
  - “worker 全部压到 0 卡”确实是问题之一，已经通过 per-rank/per-worker CUDA 绑定修复。
  - `dataset_num_workers=2` 在当前四个正式实验上已经能启动并出 loss。
  - 仍需注意：worker 会在每张卡上额外持有退化上下文，显存接近满载的配置仍可能需要降到 `workers=1` 或 `workers=0`。

### 2026-04-28：v5.3.4 image branch 帧数参数漏传修复

- 问题：`dd79sgu25m/myj7ukyewz` 上的 `v5.3.4` 89 帧实验持续打印：
  - `num_frames % 4 != 1. We round it up to 25.`
- 排查结论：
  - 该 warning 不是 89 帧 video branch 触发的。
  - 配置里已经写了 `image_branch_num_frames: 5`，目标是 89 帧真实视频 + 5 帧 image pseudo-video。
  - 但 `train_flashvsr_stage1_v5_3_lora.py` 在构造 `FlashVSRTarStreamingDatasetV53` 时漏传了 `args.image_branch_num_frames`。
  - 因此 5.3.4 实际走默认逻辑：`((num_frames - 1) // 4) + 1`，89 帧时得到 23 帧 image pseudo-video。
  - 23 不满足 Wan 的 `4n+1` 检查，于是 shape checker 把它 round 到 25 并打印 warning。
- 影响：
  - 对 `v5.3 / v5.3.1` 的 17 帧实验影响不大，因为默认值正好是 5。
  - 对 `v5.3.4` 不应视作无害 warning；它说明当前运行中的 89 帧对照实验 image branch 不是预期的 5 帧，而是走了 23/25 相关路径。
- 修复：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - `tar_v53` dataset 构造时补传：
    - `image_branch_num_frames=args.image_branch_num_frames`
- 验证：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py` 通过。
- 注意：
  - 已运行中的 `v5.3.4` 进程不会自动吃到这个修复，需要重启后才会变成真正的 `89f video + 5f image pseudo-video`。
- 后续确认：
  - `20260428_035800` debug 版确认旧问题真实表现为 `video_num_frames=89 image_num_frames=23`，因此旧版确实错误。
  - 单独抽 dataset 样本确认正确逻辑应输出：
    - `video (89, 3, 768, 1280)`
    - `image_video (5, 3, 768, 1280)`
  - 修复实际应补在 `tar_v53` 分支，而不是 `tar_v5` 分支；已修正。
  - debug 打印已从代码中移除。
  - 正式干净重启目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_20260428_040800`
  - 该目录已出 loss：
    - `step=1 loss=0.748051`
    - `step=2 loss=0.382171`
    - `step=3 loss=1.202358`
    - `step=4 loss=0.549946`
  - 当前 clean run 日志未再出现 `num_frames % 4 != 1. We round it up to 25.`，可作为正式 `v5.3.4` 训练继续运行。

### 2026-04-28：phase2 resume / worker=2 显存风险复查

- 用户问题：
  - 真正 DeepSpeed optimizer/state resume 时如果换 `batch_size` 是否会出问题。
  - `v5.3.1` 使用 worker=2 的热重启已经 OOM，是否说明必须调小 batch size，或者存在显存泄露。
  - 确认 `v5.3.4` 是否使用 worker=2 且当前是否健康。
- 复查结论：
  - 真 resume 最稳妥是不改 `batch_size/world_size/optimizer/scheduler`，否则即使状态能加载，也会让有效 batch、学习动态和 optimizer state 不再严格连续。
  - 如果必须降低 batch size，更推荐走 warm-start：从已有 safetensors 导入 LoRA + projector，换 seed 新开实验；不要声称这是 optimizer 原地 resume。
  - `v5.3.1` 的 OOM 有两种表现：
    - 早期 OOM：worker 退化上下文仍集中到 `cuda:0`，属于已修复问题。
    - 修复后 OOM：`bs12 + worker=2` 在 backward 阶段仍需额外分配约 `5.38GiB`，但每张卡只剩约 `3GiB`，更像显存余量不足，不像持续增长的显存泄露。
  - `v5.3.2` 当前 warm-start 目录仍在跑：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426_20260428_032300`
    - 最新检查时已到约 `step=82`，loss 在 `0.06-0.13` 区间。
  - `v5.3` 当前 warm-start 目录仍在跑：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427_20260428_032300`
    - 最新检查时已到约 `step=79`，loss 在 `0.04-0.15` 区间。
  - `v5.3.1` 最新 worker=2 warm-start 目录已失败：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427_20260428_033500`
    - 日志显示 backward 阶段 CUDA OOM。
  - `v5.3.4` 干净版继续健康：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_20260428_040800`
    - config 使用 `dataset_num_workers=2`。
    - 日志未再出现 `round it up`，已持续出 loss。
- 当前建议：
  - 要做严格 resume：保持原 `bs12`，先把 worker 降到 `1` 或 `0`，使用 `resume_training_state_dir`。
  - 要保留 worker=2：降低 batch size 后 warm-start 新实验，不再称作真 resume。

### 2026-04-28：v5.3.x phase2 warm-start 修复 validation 卡住并重启三条实验

- 用户要求：
  - 三条 `v5.3.x phase2` 实验继续以 worker=2 启动。
  - 不能关闭 validation。
  - 必须确认从历史 checkpoint 热启动，并看到第一条 loss。
- 问题 1：三条实验都曾卡在 `Preparing fixed validation samples...`。
  - 根因：`v5.3 / v5.3.1` 的 `tar_v53` 数据集是 paired sample，一个训练样本包含 video branch + image pseudo-video branch；旧的 validation 采样回退到 `iter(dataset)` 时会触发 image branch，容易被图像读取/尺寸过滤拖住。
  - 修复：
    - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`
    - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v532_yubari_frames.py`
    - 新增 `validation_video_iterator(...)`，validation 固定样本只从 video branch 取。
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
    - `collect_fixed_validation_samples(...)` 优先调用 `dataset.validation_video_iterator(...)`。
  - 结果：三条新实验均打印 `Prepared 3 fixed validation samples.`，validation 未关闭。
- 问题 2：`v5.3.2 step1300 warm-start` 启动脚本误用通用 `v5.3` 训练入口。
  - 错误表现：`FlashVSR-Stage1-Release-16GPU-v5-3-2-Phase2-WarmRestart-Step1300-bs10-Worker2-Seed20260428.sh` 里 `TRAIN_PY` 指向 `train_flashvsr_stage1_v5_3_lora.py`。
  - 修复：改为 `train_flashvsr_stage1_v5_3_2_lora.py`，保留 `v5.3.2` 的 Yubari single-frame image branch 逻辑。
- 统一设置：
  - `dataset_num_workers: 2`
  - `dataloader_in_order: false`
  - `dataloader_persistent_workers: true`
  - `dataloader_prefetch_factor: 2`
  - `batch_size: 10`
  - `validation_num_samples: 3`
  - `zero_init_lq_proj_in=false`
- 新启动目录与热启动来源：
  - `v5.3.2`
    - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_warmrestart_step1300_bs10_worker2_seed20260428_20260428_052600`
    - checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_unfreeze_from_step2000_seed20260426_20260426_142230/output/step-1300.safetensors`
    - 首批 loss：`step=1 loss=0.056311`，`step=2 loss=0.081109`，`step=5 loss=0.086437`
  - `v5.3.1`
    - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_warmrestart_step700_bs10_worker2_seed20260428_20260428_052700`
    - checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_unfreeze_from_step1500_seed20260427_20260427_001300/output/step-700.safetensors`
    - 首批 loss：`step=1 loss=0.091856`，`step=2 loss=0.063181`，`step=4 loss=0.069115`
  - `v5.3`
    - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_warmrestart_step700_bs10_worker2_seed20260428_20260428_052800`
    - checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_unfreeze_from_step1500_seed20260427_20260427_001300/output/step-700.safetensors`
    - 首批 loss：`step=1 loss=0.105829`，`step=2 loss=0.106001`，`step=5 loss=0.073650`
- 当前结论：
  - 这次不是关 validation 绕过问题，而是修复了 validation 采样路径。
  - 三条实验均已从历史 safetensors 热启动，worker=2，看到第一条 loss。

### 2026-04-29：v5.3 图像/视频支路分离可视化与 LQ projector 对齐 bug 修复

- 用户问题：
  - 加入图像联合训练后，大运动效果变差。
  - 怀疑 `v5.3` 的 video branch 和 image pseudo-video branch 没有真正分开。
  - 要求用 `v5.3 phase2 step-1000` checkpoint 检查从输入、VAE、LQ projector、DiT 多层 token 到输出前的支路状态。
- 使用 checkpoint：
  - 原路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_warmrestart_step700_bs10_worker2_seed20260428_20260428_052800/output/step-1000.safetensors`
  - 临时复制到两卡母机：`/mnt/task_wrapper/user_output/artifacts/checkpoints/probe/v53_step1000.safetensors`
- 新增检查脚本：
  - `wanvideo/model_training/flashvsr/scripts/probe_v53_branch_separation.py`
  - 脚本输出：
    - 原始输入：`inputs/video_gt.mp4`、`inputs/video_lq.mp4`、`inputs/image_pseudo_gt.mp4`、`inputs/image_pseudo_lq.mp4`
    - VAE 重建：`vae_recon/video_branch_recon.mp4`、`vae_recon/image_branch_recon.mp4`
    - token 可视化：`token_norms/`
    - 结构摘要：`branch_separation_summary.json`
- 关键检查目录：
  - 修复前问题目录：`/mnt/task_wrapper/user_output/artifacts/debug/v53_branch_probe_step1000_minimal_20260429_r3`
  - 修复后正确目录：`/mnt/task_wrapper/user_output/artifacts/debug/v53_branch_probe_step1000_minimal_20260429_r4_segmentalign`
- 定位到的根因：
  - `v5.3` 的 GT/VAE/DiT segment 本身是分开的：
    - video raw frames：`17`
    - image pseudo-video raw frames：`5`
    - DiT latent segment lengths：`[5, 2]`
    - token lengths：`[19200, 7680]`
  - 但是 LQ projector 输出的 latent-time 比 DiT 少一帧：
    - 17 帧 video LQ projector 输出 `4` 个 latent-time
    - 5 帧 image LQ projector 输出 `1` 个 latent-time
  - 旧代码使用全局 front padding，把整个 LQ token 序列一次性补到 DiT 总长度。
  - 对 `17 + 5` 的 paired sample，旧布局等价于：
    - `[pad, pad, V0, V1, V2, V3, I0]`
  - 但 DiT segment 期望是：
    - video segment：`5` 个 latent-time
    - image segment：`2` 个 latent-time
  - 因此 image segment 实际拿到了 `[V3, I0]`，最后一个 video LQ latent 泄漏进 image branch。
- 修复方式：
  - 在 `train_flashvsr_stage1_v5_3_lora.py` 中新增按 segment 对齐的 LQ padding。
  - 每个 segment 内单独补自己的 front latent frame：
    - video：`[pad, V0, V1, V2, V3]`
    - image：`[pad, I0]`
  - 不再允许 LQ token 跨 video/image branch 边界补齐。
  - 同一修法同时覆盖 streaming 和 nonstreaming LQ projector。
- 非流式 LR projector 检查：
  - 记录文件：`/mnt/task_wrapper/user_output/artifacts/debug/v534_nonstream_alignment_20260429/nonstream_lq_alignment_summary.json`
  - 结论：
    - `17 + 5`：DiT latent `[5,2]`，LQ latent `[4,1]`
    - `89 + 5`：DiT latent `[23,2]`，LQ latent `[22,1]`
    - 非流式 projector 也需要 per-segment padding，否则同样会发生 video LQ token 泄漏到 image branch。
- 额外修复：
  - `tar_streaming_dataset_v53.py` 之前没有把 YAML 中的 `degradation_config_path` 传给 base dataset。
  - 已补上该参数，后续 `tar_v53` 新启动实验会真正使用配置里的退化文件。
- 两卡母机重启检查：
  - 新脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Phase2-WarmRestart-Step1000-bs10-Worker2-Seed20260429-SegmentAlign.sh`
  - 新 config：`wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_phase2_warmrestart_step1000_bs10_worker2_seed20260429_segmentalign.yaml`
  - 新实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_warmrestart_step1000_bs10_worker2_seed20260429_segmentalign_20260429_0335`
  - 当前已出 loss：
    - `step=1 loss=0.082779`
    - `step=2 loss=0.067493`
    - `step=3 loss=0.087492`
    - `step=6 loss=0.072633`
- 注意：
  - 正在六节点上运行的 `v5.3.4 nonstream LR projector` 实验是在这次 per-segment LQ alignment 修复前启动的。
  - 该实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_worker1_20260428_1834`
  - 它当前能训练出 loss，但从代码时间点看仍可能带有旧的全局 LQ padding 问题。

### 2026-04-29：随机 projector 对照重启与 `v5.3.5` 48 卡修复版启动

- 用户要求：
  - 两卡母机停止旧实验，改为 `17f video + 5f image pseudo-video`，projector 随机初始化，从头训练。
  - 六节点 48 卡实验修正 per-segment LQ alignment 后重启，命名为 `v5.3.5`，设置为 `89f video + 5f image pseudo-video`，非流式 LR projector。
- manifest 备份：
  - 旧 Takano video manifest 已在：`s3://lxh/data/mainfest/takano_video_train_all.txt`
  - 本轮补充 image tar manifest：`s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt`
  - 本地运行路径：`/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_image_4k_tar_manifest.txt`
  - 行数：`189220`
- 两卡母机第一次启动失败：
  - 失败目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_segmentalign_20260429_0406`
  - 原因：从机 `myj7ukyewz` 启动时 `s3://lxh/data/mainfest/takano_image_4k_tar_manifest.txt` 尚未备份，`conductor s3 cp` 返回 404，rank0 等 rendezvous。
  - 处理：备份 manifest 后用新时间戳重启。
- 两卡母机有效新实验：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_segmentalign_20260429_0412`
  - 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Lora-17f-FullSources-bs12-lr1e5-AliyunDegra-RandomProj-SegmentAlign.sh`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_randomproj_segmentalign.yaml`
  - 已出 loss：
    - `step=1 loss=0.462243`
    - `step=2 loss=0.377958`
- 六节点 48 卡 `v5.3.5` 新实验：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_segmentalign_20260429_0405`
  - 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-SegmentAlign.sh`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_segmentalign.yaml`
  - 已出 loss：
    - `step=1 loss=0.474171`
    - `step=2 loss=0.517721`
    - `step=3 loss=0.928197`
    - `step=7 loss=0.473409`
- 本轮敏感设置确认：
  - 两条新实验都没有传入 `resume_stage1_checkpoint`。
  - 两条新实验都没有传入 `lq_proj_checkpoint`。
  - 启动命令均显式使用 `--zero_init_lq_proj_in false`，因此 projector 不是 zero init，也不是 FlashVSR projector init，而是随机/默认初始化。
  - LoRA 仍是 PEFT 默认初始化：`lora_A` 随机、`lora_B` 近似零，因此初始 LoRA 分支输出为零。
  - 训练数据结构均为 `tar_v53` paired sample：
    - 一条真实 video branch。
    - 一条 5 帧 image pseudo-video branch。
    - DiT token 层使用 packed segment，并使用本轮修复后的 per-segment LQ alignment。

### 2026-04-29：修正 Stage1 非流式 projector 的时间维对齐，重启 89f/17f 正式对照

- 问题复盘：
  - 进一步对照 FlashVSR 官方推理后确认，旧 `nonstreaming` projector 仍然会丢掉 warm-up 输出，表现为：
    - `89f video -> 22` 个 LQ latent-time；
    - `17f video -> 4` 个 LQ latent-time；
    - `5f image pseudo-video -> 1` 个 LQ latent-time。
  - 这和第一阶段非流式 SR teacher 的目标不一致；第一阶段应保持 WAN VAE 对齐：
    - `89f -> 23`；
    - `17f -> 5`；
    - `5f image pseudo-video -> 2`。
- 代码改动：
  - 文件：`wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - 新增 `lq_proj_temporal_mode=nonstreaming_aligned`。
  - `nonstreaming_aligned` 仍然整段走 3D conv/linear，但不再 drop 第一个 warm-up 输出，因此 LQ projector 的 latent-time 与 WAN VAE 对齐。
  - `_align_lq_latents_to_dit_tokens()` 增加 fast path：若 LQ tokens 已经等于 DiT tokens，直接返回，不再进入补零/裁剪逻辑。
- 新 48 卡 89f 实验：
  - 目的：替换旧 `v5.3.5 ... segmentalign`，修正为 `23 对 23`。
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_170100`
  - 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned23.sh`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23.yaml`
  - 当前状态：已启动，已过模型/LoRA 初始化，当前停在 `Preparing fixed validation samples...`，尚未看到第一条 loss。
- 新 16 卡 17f 实验：
  - 目的：替换旧 `17f streaming/4 对 5` 版本，修正为 `5 对 5`。
  - 初次使用 `bs12` 已出 loss，但显存约 `150GB/183GB`，偏满。
  - 已改为 `bs8` 重新启动。
  - 当前有效目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_171200`
  - 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-6-Lora-17f-FullSources-bs8-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned5.sh`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5.yaml`
  - 已出 loss：`step=1 loss=0.846186`
  - 显存约 `107GB/183GB`，GPU 利用率约 `89-100%`，比 `bs12` 更稳。
- 共同敏感设置：
  - 两条实验均为 `tar_v53` paired sample：一个真实 video branch + 一个由单张图生成的 5 帧 image pseudo-video branch。
  - 两条实验均使用 `image_video_joint_packed=true`，video/image segment 在 token 层分开。
  - 两条实验均使用 Aliyun full degradation：`params_aliyun_video_compression_v1.yaml`。
  - 两条实验均为随机/默认 projector 初始化：启动命令显式 `--zero_init_lq_proj_in false`，且没有传 `lq_proj_checkpoint`。
  - LoRA 仍为 PEFT 默认初始化，初始输出近似 0。

### 2026-04-29：修正 `nonstreaming_aligned` validation 对齐问题并重启 17f/48 卡

- 问题：
  - 训练主 pipeline 已经通过 `lq_proj_temporal_mode=nonstreaming_aligned` 做到：
    - `17f -> 5`；
    - `89f -> 23`。
  - 但在线 validation 里临时创建的 `WanFixedPromptFlashVSRStage1Pipeline` / `WanTextPromptLQPipeline` 没有继承该参数，仍使用默认 `streaming` projector。
  - 因此旧启动进程里 training forward 与 validation projector 严格来说没有完全对齐。
- 代码修复：
  - 文件：`wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - `WanFixedPromptFlashVSRStage1Pipeline.from_pretrained()` 增加 `lq_proj_temporal_mode` 参数。
  - `WanTextPromptLQPipeline.from_pretrained()` 增加 `lq_proj_temporal_mode` 参数。
  - `FlashVSRValidationCallback` 增加 `validation_lq_proj_temporal_mode`，并在构造 validation pipe 时传入。
  - `main()` 里将 `args.lq_proj_temporal_mode` 传给 validation callback。
- 2GPU 89f validation smoke：
  - no-validation smoke 先验证训练路径可跑：
    - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_noval_20260429_023451`
    - 已出 loss：`step=1 loss=0.030807`，`step=2 loss=0.167824`
  - validation smoke 再验证固定 validation 样本和在线推理路径：
    - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_val_20260429_024349`
    - 已通过：`Prepared 3 fixed validation samples.`
    - 已出训练 loss：`step=1 loss=0.043716`
- 17f 正式实验重启：
  - 旧目录 `..._20260429_171200` 是修复前启动，训练本身 aligned，但 validation 进程内仍是旧代码，因此停掉。
  - 新目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200`
  - 已通过：`Prepared 3 fixed validation samples.`
  - 已出 loss：
    - `step=1 loss=0.710992`
    - `step=2 loss=0.541623`
    - `step=3 loss=1.395321`
    - `step=4 loss=0.773073`
  - wandb run：`vdstmthm`
- 48 卡正式实验重启：
  - 旧目录 `..._20260429_170100` 停在修复前的 validation 逻辑，已停止。
  - 新目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_175800`
  - 已通过：`Prepared 3 fixed validation samples.`
  - 已出 loss：
    - `step=1 loss=1.065475`
    - `step=2 loss=0.568194`
    - `step=3 loss=1.974530`
    - `step=4 loss=1.481316`
    - `step=5 loss=0.882794`
    - `step=6 loss=1.261346`
    - `step=7 loss=0.831906`
  - wandb run：`07b3aldg`

### 2026-04-29：补 v5.3.5/v5.3.6 非流式对齐推理入口，并修正 Stage2 chunk causal 文档

- 推理代码：
  - 新增单视频入口：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned.py`
  - 新增目录批量入口：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned_batch.py`
  - 新增可复用启动脚本：`wanvideo/model_inference/flashvsr/history/run_stage1_v5_3_aligned_dir.sh`
- 该推理入口用于当前 `v5.3.5/v5.3.6` Stage1 非流式对齐实验：
  - 默认 `lq_proj_temporal_mode=nonstreaming_aligned`；
  - 默认 `input_bicubic_upscale=4.0`，即先按输入视频原始尺寸 bicubic 放大四倍，再送入模型；
  - 默认开启 color fix，`color_fix_method=adain`；
  - 批量入口只加载一次 Wan/VAE/projector/LoRA，再循环处理目录内视频，避免每个视频重复加载大模型。
- 文档修正：
  - 更新 `doc/flashvsr_three_stage_author_aligned_20260429.md`。
  - 新增 mask 图：`doc/flashvsr_stage2_chunk_causal_mask.svg`。
  - Stage2 causal 重新整理为 `chunk` 级别：
    - `chunk` 是 causal 单位；
    - `(2,8,8)` 是 block-sparse 单位；
    - chunk 内 full attention；
    - chunk 之间 causal；
    - 在允许的 chunk pair 内再做 `(2,8,8)` block top-k sparse。
  - 明确 `kv_ratio=3.0`、`local_range=9` 属于推理技巧，不进入 Stage2 训练；Stage2 训练只保留固定 top-k。
- 验证：
  - `python3 -m py_compile` 已通过两个新推理脚本。

### 2026-04-29：整理 v6 Stage2 当前训练计划

- 文档更新：
  - 文件：`doc/flashvsr_stage2_v6_design.md`
  - 已将 `doc/flashvsr_three_stage_author_aligned_20260429.md` 合并进该主文档；旧文件只保留跳转说明，避免两套解释并存。
- 当前确认：
  - Stage2 训练不接 `kv_ratio` / `local_range`，这两个属于推理 KV cache / locality 技巧。
  - Stage2 训练需要固定 top-k，并接入 `(2,8,8)` block-sparse attention。
  - causal 单位是 chunk：
    - `1 chunk = 2 latent-time`；
    - 开头 `6 latent-time = 3 chunks` full attention；
    - 后续 chunk 之间 causal。
  - LR projector 训练要切到 causal streaming：
    - `f0` 复制 3 次做 warm-up；
    - 后面每 4 帧输出 1 个 latent-time；
    - 每个 4 帧 group 都更新 cache，供后一个 group 使用；
    - `89f -> 22`。
  - v6 第一版训练参数：
    - 训练 DiT LoRA；
    - 训练 `LR Proj-In`；
    - 冻结 base WAN、VAE、prompt/text 相关模块；
    - block-sparse attention 不新增可学习参数。

### 2026-04-29：检查当前 `v5.3.5 89f` / `v5.3.6 17f` GPU 利用率

- 采样方式：
  - 对当前 89f 48GPU 实验所在 6 台机器、17f 16GPU 实验所在 2 台机器分别采样。
  - 每台机器采 20 次，每 3 秒一次。
  - 指标来自 `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw`。
- 89f 48GPU：
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_175800`
  - 6 台机器整体基本稳定在 `99.9%-100%`。
  - 只有主机 GPU0 最低采到一次 `84%`，没有 0%。
  - 显存平均约 `129-132GB/GPU`。
- 17f 16GPU：
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200`
  - `dd79sgu25m` 整体平均约 `92.7%`，其中 GPU0 采到一次 `0%`，其余 GPU 多数在 `95%` 左右。
  - `myj7ukyewz` 整体平均约 `95.4%`，无 0%。
  - 显存平均约 `130-132GB/GPU`。
- 结论：
  - 当前 89f 的 `dataset_num_workers=1` 没有表现为明显 GPU 饥饿；48 卡采样期间几乎满载。
  - 注意这里的 `dataset_num_workers=1` 是每个 rank 一个 worker，不是整个 48 卡只有一个 worker；48 卡总计约 48 个 DataLoader worker。
  - 17f 的 worker=2 在本轮采样下也能维持约 `93-95%`，但偶尔 step 边界会出现低谷。

### 2026-04-29：核查当前 Stage1 对齐状态，并重写 Stage2 v6 骨架

- Stage1 48GPU 远程核查：
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260429_175800`
  - 主机 `b8gkuie2ns` 的 snapshot / `resolved_args.yaml` 确认为：
    - `num_frames=89`
    - `image_branch_num_frames=5`
    - `lq_proj_temporal_mode=nonstreaming_aligned`
    - `image_video_joint_packed=true`
    - `batch_size=1`
    - `dataset_num_workers=1`
  - 六个节点均能看到同名 run 目录；输出仍以主机 artifacts 为准。
  - `run.log` 已到 `step=60`，`step=50` 触发 validation，并保存了 89 帧视频。
  - 结论：当前 Stage1 训练 / validation 均使用非流式对齐 projector，不是旧 streaming projector；当前一阶段可以作为 Stage2 初始化来源。
- Stage1 代码核查：
  - `FlashVSRLQProjIn.forward_nonstreaming()` 在 `nonstreaming_aligned` 下保留 warm-up 输出。
  - `WanVideoUnit_InputVideoEmbedderV53` 对 video branch 和 image pseudo-video branch 分别过 VAE，再在 latent-time 维拼接。
  - `FlashVSRUnit_LQVideoEmbedderV53` 对 `lq_video` 和 `image_lq_video` 分别过 projector，再拼接 LQ latent。
  - `flashvsr_stage1_model_fn()` 当前 `lq_latents.shape[1] == expected_tokens` 时直接返回，不触发旧 padding / crop fallback。
- Stage2 v6 重写：
  - 新训练入口：`wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py`
  - 新 attention patch：`diffsynth/models/wan_video_dit_stage2_v6.py`
  - 新推理入口：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6.py`
  - 新批量推理入口：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_batch.py`
  - 新 validation callback：`FlashVSRStage2ValidationCallback`
  - 新 smoke config：`wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_17f_videoonly_blocksparse.yaml`
  - 新 48GPU config：`wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse.yaml`
  - 新 smoke sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-BlockSparse.sh`
  - 新 48GPU sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse.sh`
- v6 当前实现边界：
  - video-only；
  - GT 先按 WAN VAE 得到 `1 + (F-1)/4`，再丢掉首 latent；
  - `89f -> GT 23 -> drop first -> 22`；
  - LQ projector 使用官方 streaming 行为，`f0` warm-up 不输出，`89f -> 22`；
  - DiT self-attention 替换为 chunk-causal + `(2,8,8)` block sparse；
  - `topk_ratio=2.0` 固定进入训练；
  - `kv_ratio/local_range/KV cache` 不进入 v6 第一版训练。
  - validation 固定抽 video-only 样本，保存 `hr.mp4 / lq.mp4 / sr.mp4 / meta.json`。
  - `meta.json` 记录 `input_num_frames` 和 `output_num_frames`，用于后续确认 Stage2 streaming 输出长度。
- 验证：
  - 本地 `python -m py_compile` 已通过 v6 训练、attention、推理文件。
  - 远端 `dd79sgu25m` 已确认同步后的 v6 训练 / 推理文件 `py_compile` 通过。

### 2026-04-29：将 v6 Stage2 mask 改为官方 `generate_causal_block_mask` 同款 chunk 划窗

- 背景：
  - 复查官方 FlashVSR `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/diffsynth/models/wan_video_dit.py`。
  - 官方存在 `generate_causal_block_mask()`：
    - 先在 `seqlen=f//2` 的 chunk 轴构造 mask；
    - `causal_mask = (j <= i) & (j >= i - local_num + 1)`；
    - `causal_mask[0,1] = True`；
    - `causal_mask[:2,2] = True`；
    - 再展开到 block-sparse kernel 需要的 block 矩阵。
- 结论：
  - 这个函数本质是 chunk 级 causal + local 历史窗口。
  - `causal_mask[0,1]` 和 `causal_mask[:2,2]` 正好把前三个 chunk 补成 full attention start window。
  - 这比之前 v6 自己直接写 `(key_chunk <= query_chunk) | start_window` 更贴近作者代码。
- 代码改动：
  - 修改 `diffsynth/models/wan_video_dit_stage2_v6.py` 的 `build_stage2_chunk_block_mask()`。
  - 新逻辑先构造 `chunks x chunks` mask：
    - `stage2_local_num=-1` 时按官方随机 `local_num` 逻辑采样；
    - 候选窗口为 `seqlen-3 / seqlen-4 / seqlen-2 / seqlen`；
    - 按官方硬编码补前三个 chunk full；
    - 删除额外 tail drop，末尾早期 chunk 是否可见完全交给 `local_num` 滑窗决定。
  - 然后用 `repeat_interleave(spatial_blocks)` 展开成 block 级矩阵。
- 文档：
  - 更新 `doc/flashvsr_stage2_v6_design.md`，明确 v6 mask 现在对齐官方 `generate_causal_block_mask()` 的 chunk 语义。
- 验证：
  - 本地 `python -m py_compile diffsynth/models/wan_video_dit_stage2_v6.py` 通过。
  - 本机 base 无 `torch`，未在本地打印实际 mask 矩阵；等用户提供测试机后再跑远端 mask 单元测试和 v6 smoke。

### 2026-04-29：v6 Stage2 mask 改为完全官方 local_num 窗口

- 根据用户确认，v6 不再保留额外 tail drop。
- `diffsynth/models/wan_video_dit_stage2_v6.py` 改为：
  - `build_stage2_chunk_block_mask(..., local_num=None)`；
  - `local_num=None` 时复用官方随机窗口采样；
  - 仍保留官方前三个 chunk full 的硬编码；
  - 只在 chunk 级构造 mask，再展开到 block 矩阵。
- `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py` 和 `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6.py` 新增 `stage2_local_num`，默认 `-1` 表示官方随机逻辑。
- 更新 `doc/flashvsr_stage2_chunk_causal_mask.svg`，移除 tail drop 图例，改为官方 local window 示例。

### 2026-04-30：轻退化 17 帧测试集与推理耗时对比

- 测试机：`tjmjauaq4d`。
- 生成轻退化合成测试集：
  - 本地脚本：`wanvideo/data/flashvsr/tests/run_prepare_light_x4_testsets_20260430.sh`
  - 远端目录：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_20260430`
  - 内容：5 个 Takano + 5 个 Yubari，`gt=1280x768`，`lq=320x192`，17 帧，8 fps。
  - 退化配置：`wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml`
  - 规则：Aliyun 风格轻退化，只保留第一阶段退化，去掉最后 bicubic 回原尺寸，LQ 保持 GT 的 1/4。
  - 云端备份：`s3://lxh/data/test/testset10_17f_aliyun_light_x4_lq_20260430`
- 生成真实测试集：
  - 原始本机目录：`/Users/lixiaohui/Library/CloudStorage/Box-Box/challenging_test_lxh`
  - 远端原始目录：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_raw_20260430`
  - 处理后目录：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x109_nativecrop_20260430`
  - 处理规则：不 resize，按原生比例 center crop 到 `320x109`，只保留前 17 帧，8 fps。
  - 脚本：`wanvideo/data/flashvsr/tests/process_challenging_real_native_crop.py`
  - 云端备份：`s3://lxh/data/test/challenging_test_lxh_17f_320x109_nativecrop_20260430`
- 推理对比目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_timed_light_x4_flash_seedvr_v536_20260430_by_dataset`
  - 对比方法：FlashVSR official、SeedVR-3B、SeedVR2-3B、`v5.3.6 step-2100`。
  - `v5.3.6` checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200/output/step-2100.safetensors`
  - `v5.3.6` 推理固定使用：`lq_proj_temporal_mode=nonstreaming_aligned`、`input_bicubic_upscale=4.0`、`color_fix_method=adain`、`num_frames=17`。
- 推理脚本修复：
  - `wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh` 增加 `OUT_FPS=8` 并显式传 `--out_fps`，避免真实视频 metadata 缺 `video_fps` 时 SeedVR 保存阶段报错。
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_full_cloud_padded.py` 对奇数尺寸输入改用 `yuv444p` padding，避免强制对齐到偶数尺寸。
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v5_3_aligned.py` 在 color fix 前将参考 LQ resize 到 SR 实际输出尺寸，解决真实测试集 `320x109 -> 1280x436` 后模型内部 round 到 `1280x448` 导致的 shape mismatch。
- 当前状态：
  - FlashVSR official 两组已完成。
  - SeedVR2 synthetic 已完成；real 因 `video_fps` metadata 问题已按 `--out_fps 8` 重跑。
  - SeedVR3 synthetic 已完成；real 正在重跑。
  - `v5.3.6` synthetic 已完成；real 修复 color fix 后正在重跑。

### 2026-04-30：`v5.3.6 step-2300` 测试与 v6 Stage2 smoke

- 测试机：`tjmjauaq4d`。
- checkpoint 来源：
  - 训练机 `dd79sgu25m`：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260429_175200/output/step-2300.safetensors`
  - 通过 `s3://lxh/tmp/v536_step2300_nonstreamproj_aligned5.safetensors` 中转到测试机。
- 推理输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v536_step2300_light_x4_20260430_by_dataset/synthetic/v5_3_6_step2300_nonstream_aligned`
  - `/mnt/task_wrapper/user_output/artifacts/inference/compare_v536_step2300_light_x4_20260430_by_dataset/real/v5_3_6_step2300_nonstream_aligned`
- 推理设置：
  - `lq_proj_temporal_mode=nonstreaming_aligned`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
  - `lq_proj_scale=1.0`
  - `num_frames=17`
  - synthetic 10/10 完成，耗时约 `216s`。
  - real 11/11 完成，耗时约 `138s`。
- v6 Stage2 smoke：
  - 本地修复 `train_flashvsr_stage2_v6_lora.py` 的 `PipelineUnit` import，改为从 `diffsynth.diffusion.base_pipeline` 导入。
  - 测试机 2 卡 smoke 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-17f-VideoOnly-BlockSparse.sh`
  - smoke 输出目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_smoke_2gpu_v6_lora_17f_videoonly_blocksparse_20260430_021032`
  - 已跑到 2 step：`loss=0.220676`、`loss=0.596688`。
  - 实际启用：`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_topk_ratio=2.0`。

### 2026-05-03：`v5.3.6 resume step4600+` 批量测试准备与启动

- 测试执行机器：`myj7ukyewz`。
- 训练来源：
  - `dd79sgu25m` / `myj7ukyewz` 上的 536 resume 实验已停止。
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_6_resume_step4600_seed20260501_lora_17f_fullsources_bs8_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned5_20260502_012300`
  - 从 `dd79sgu25m` 的 `output/` 上传到 `s3://lxh/tmp/v536_resume_step4600_output_20260503/`，再同步到 `myj7ukyewz`。
- 新增脚本：
  - 测试集生成：`wanvideo/data/flashvsr/tests/run_prepare_v536_eval_testsets_20260503.sh`
  - checkpoint 扫描测试：`wanvideo/model_inference/flashvsr/history/run_v536_scan_ckpts_17f_20260503.sh`
  - 真实视频处理脚本 `process_challenging_real_native_crop.py` 改为短视频不足目标帧数时 repeat last frame，避免 89f 真实测试集因短视频中断。
- 测试集：
  - 合成 17f：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_17f_aliyun_light_x4_lq_20260503`
  - 合成 89f：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503`
  - 真实 17f：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_17f_320x192_resizecrop_20260503`
  - 真实 89f：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503`
  - 合成集内容：5 个 Takano + 5 个 Yubari，Aliyun light x4 LQ，`gt=1280x768`，`lq=320x192`，8 fps。
  - 真实集来源：本机 `/Users/lixiaohui/Library/CloudStorage/Box-Box/challenging_test_lxh`，处理为 `320x192`，8 fps。
  - 四套测试集均备份到 `s3://lxh/data/test/` 下同名目录。
- 批量测试：
  - 输出目录：`/mnt/task_wrapper/user_output/artifacts/inference/v536_scan17_20260503_by_ckpt`
  - 当前 my 已拉取 25 个 `step-*.safetensors`，范围 `step-4700` 到 `step-7100`。
  - 先测试 17f synthetic/real 两套，89f 测试集作为后续备用。
  - 推理设置：`lq_proj_temporal_mode=nonstreaming_aligned`，`input_bicubic_upscale=4.0`，`color_fix_method=adain`，`lq_proj_scale=1.0`，`num_frames=17`。
  - 启动方式：8 卡并行，`MAX_PARALLEL=8`，脚本结束后自动执行 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`。
  - 启动检查：8 张卡均进入推理，显存约 `43.8GB/GPU`，GPU 利用率约 `94-100%`，启动早期已产出 32 个 mp4，未发现 `Traceback/RuntimeError/OOM`。
- 资源处理：
  - `dd79sgu25m` 已停止训练并启动占卡程序。
  - `myj7ukyewz` 用于测试，测试脚本结束后会自动启动占卡程序。

### 2026-05-06：v6 Stage2 89f worker2 smoke 与 48 卡正式训练启动

- 任务目标：启动二阶段 `v6` 训练，从一阶段 `v5.3.5 89f` 的 `step-10000.safetensors` 热启动 `lq_proj_in` 和 LoRA。
- 同步处理：
  - 检查本机 `tmux sync`，发现多个窗口停在 2026-05-02 的 rsync 断线报错。
  - 重新挂起六卡母机同步：`b8gkuie2ns / wfnwbym4v6 / kh5idf7f98 / hj65iqg9rh / zhki5rrddw / xwk6qjuej5`。
  - `tjmjauaq4d` 已 terminated，本次不参与。
- 新增配置和脚本：
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_lora_89f_videoonly_blocksparse_worker2.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Lora-89f-VideoOnly-BlockSparse-Worker2.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh`
- 关键设置：
  - `num_frames=89`
  - `dataset_mode=stage2_video_only`
  - `yubari_video_prob=0.5`
  - `takano_video_prob=0.5`
  - `dataset_num_workers=2`
  - `batch_size=1`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `validation_num_samples=0`
- checkpoint：
  - 本地路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
  - 六台机器均已补齐同一路径。非主节点从 `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/.../step-10000.safetensors` 拉取。
- 2GPU smoke：
  - 机器：`b8gkuie2ns`
  - 释放 GPU `0,1` 做 smoke，GPU `2-7` 保持占卡。
  - 输出目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_smoke_2gpu_v6_lora_89f_videoonly_blocksparse_worker2_20260506_smoke_b8_worker2`
  - 结果：`step=1 loss=0.038978`，GPU `0,1` 利用率达到 `100%`。
- 48GPU 正式训练：
  - 六节点：`b8gkuie2ns / wfnwbym4v6 / kh5idf7f98 / hj65iqg9rh / zhki5rrddw / xwk6qjuej5`
  - `MASTER_ADDR=240.12.149.199`
  - `MASTER_PORT=29606`
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2`
  - wandb 已开启。
  - 第一条 loss：`step=1 loss=0.115593`
  - 48 张 GPU 利用率约 `99-100%`，单卡显存约 `90GB`。
- 结论：
  - 89f Stage2 `v6` worker2 版本已从一阶段 `step-10000` 成功热启动并进入正式 48 卡训练。
  - 当前 GPU 利用率达到目标，数据加载没有成为启动阻塞。

### 2026-05-06：`v5.3.5 89f` checkpoint 扫描测试启动

- 任务目标：用与 `v5.3.5 89f` 训练一致的推理 setting，扫描一阶段 `v5.3.5` 多个 checkpoint，并与之前 `v5.3.6 17f` 的扫描结果一起下载到本机桌面。
- 测试机器：`myj7ukyewz`。
- checkpoint 来源：
  - S3：`s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output`
  - my 本地缓存：`/mnt/task_wrapper/user_output/artifacts/ckpts/v535_resume_step3000_89f_aligned23`
  - 只拉取 `step % 500 == 0` 的 `step-*.safetensors`，避免同步整个 output。
- 新增脚本：
  - `wanvideo/model_inference/flashvsr/history/run_v535_scan_ckpts_89f_20260506.sh`
- 测试集：
  - 合成 89f：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - 真实 89f：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503`
- 推理设置：
  - `num_frames=89`
  - `height=768`
  - `width=1280`
  - `lq_proj_temporal_mode=nonstreaming_aligned`
  - `input_bicubic_upscale=4.0`
  - `lq_proj_scale=1.0`
  - `projection_scale=1.0`
  - `color_fix_method=adain`
  - `fps=8`
- 资源处理：
  - 启动前发现 my 上 `gpu_stress_tc.sh` 父进程已退出，但 7 个 orphan 子进程仍占显存；已按 `nvidia-smi --query-compute-apps=pid` 精确清理。
  - 535 测试启动后 8 卡并行，采样显示每卡约 `30-50GB`，GPU 利用率约 `100%`。
  - 脚本结尾自动执行 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`，防止 my 测试结束后空卡。
- 输出目录：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/v535_scan89_20260506_by_ckpt`
  - 完成后同步：`s3://lxh/data/test/v535_scan89_20260506_by_ckpt`
  - 本机 watcher：`tmux download_v535_v536_20260506`
  - 本机目标目录：`/Users/lixiaohui/Desktop/flashvsr_eval_v535_v536_20260506`

### 2026-05-06：补 `v6.1` Stage2 推理链并启动 `v6` 全 ckpt 扫描

- 目的：
  - 给正在跑的 48 卡 `v6` Stage2 正式训练补一条独立的、可外部批量扫描 checkpoint 的 inference 链。
  - 不修改当前正式训练目录本身，只新增 `v6.1` 的 validation / inference 实现。
- 新增 / 整理的文件：
  - `diffsynth/models/wan_video_dit_stage2_v6_1.py`
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_1_lora.py`
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py`
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1_batch.py`
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_8way_on_my_20260506.sh`
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_every100_ckpts_20260506.sh`
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_every100_on_my_20260506.sh`
- `v6.1` 推理当前规则：
  - projector 使用 Stage2 streaming 逻辑；
  - DiT self-attention 使用 `block_sparse_chunk_causal`；
  - `topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
- 修复点：
  - 89f Stage2 streaming 推理只会产出 85 帧 SR，而不是 89 帧；
  - color fix 之前若直接拿 89 帧 LQ 对齐，会触发 frame count mismatch。
  - 已在 `infer_flashvsr_stage2_v6_1.py` 中改为：
    - 若 `len(sr_video) != len(lq_video)`，先裁到 `min(sr, lq)`；
    - 再进行 `apply_color_fix(...)`。
- 正式训练 checkpoint 目录确认：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output`
  - 当前已存在 `step-100` 到 `step-3300` 的 `step-*.safetensors`。
- 新扫描测试设定：
  - 测试机器：`myj7ukyewz`
  - 测试集：
    - synthetic：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
    - real：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503`
  - checkpoint 目录缓存：
    - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_every100_20260506`
  - 从 `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output`
    拉取 `step >= 100` 且 `step % 100 == 0` 的所有 ckpt。
  - 输出目录：
    - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_scan89_every100_20260506_by_dataset`
  - 输出结构改为：
    - `synthetic/step-xxxx/*.mp4`
    - `real/step-xxxx/*.mp4`
  - 脚本结尾自动恢复：
    - `bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`
- 说明：
  - 这条扫描链与当前正式训练分离，不改训练 run.log / optimizer / wandb；
  - 作用是用统一推理口径快速观察 `v6` 各阶段 checkpoint 的效果变化。

### 2026-05-09：`v6` Stage2 chunk 跳变定位 A/B/C/D probe

- 背景：
  - `v6.1 / v6.2` Stage2 89 帧推理出现周期性跳变；
  - 用户观察跳变疑似按 chunk 边界发生；
  - 需要区分问题来自 KV cache streaming、chunk causal mask、block sparse top-k，还是 LQ projector 的 4 帧边界。
- 固定测试 checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors`
- 固定测试输入：
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
- 四组 probe：
  - Probe A：`v6.2`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_local_num=11`，`stage2_topk_ratio=2.0`
  - Probe B：`v6.2`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_local_num=11`，`stage2_topk_ratio=4.0`
  - Probe C：`stage2_attention_mode=dense_full`
  - Probe D：`v6.2`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_local_num=11`，`stage2_topk_ratio=8.0`
- 输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeA_step10000_v62_fixedlocal11_20260509`
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeB_step10000_v62_fixedlocal11_topk4_20260509`
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeC_step10000_v62_densefull_20260509`
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeD_step10000_v62_fixedlocal11_topk8_20260509`
- 本地下载目录：
  - `/Users/lixiaohui/Desktop/stage2_v6_jump_probe_step10000_20260509`
- 逐帧拆解：
  - 固定拆 `takano_04_lq_sr.mp4`
  - 本地目录：`/Users/lixiaohui/Desktop/takano04_frame_compare_stage2_v6_20260509`
  - A/B/C/D 均为 `85` 帧；
  - FlashVSR official baseline 为 `89` 帧。
- 初步观察：
  - Probe A 仍有跳变，说明问题不能简单归因于 KV cache streaming；
  - 更可疑的是 LQ projector chunk 边界、stage2 block sparse / chunk mask 与官方逻辑未完全对齐、或 50-step 推理放大边界误差。
- 详细记录：
  - `doc/flashvsr_stage2_v6_jump_probe_20260509.md`

### 2026-05-09：`v6` Stage2 E/F/G 追加定位 probe

- 目的：
  - 在 A/B/C/D 已确认 `block_sparse_chunk_causal` 路径存在 chunk 级跳变后，继续拆分 50-step diffusion、colorfix 对齐、LQ projector chunk 数值三类因素。
- 本地代码改动：
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py`
    - 新增 `--color_fix_lq_start_frame`，用于控制 85 帧 SR 与 89 帧 LQ 做 colorfix 时的起始对齐位置。
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_2.py`
    - 同步新增 `--color_fix_lq_start_frame`。
  - `wanvideo/model_inference/flashvsr/diagnose_stage2_v6_projector_chunks.py`
    - 新增 projector chunk 统计脚本，dump 每个 4 帧 chunk 输出的 mean/std/min/max/l2。
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_2_probeEFG_takano04_on_my_20260509.sh`
    - 新增 E/F/G 一键测试脚本，使用 `myj7ukyewz` 0-5 卡并行，结束后自动恢复 8 卡占卡。
- 固定 checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors`
- 固定输入：
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq/takano_04_lq.mp4`
- 远端输出：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeEFG_takano04_20260509`
- 本地下载：
  - `/Users/lixiaohui/Desktop/stage2_v6_probeEFG_takano04_20260509`
  - `/Users/lixiaohui/Desktop/stage2_v6_probeEFG_takano04_20260509_frames`
- 测试内容：
  - E：`num_inference_steps=1`，判断 50-step 是否放大边界误差。
  - F：`no_colorfix`、`adain_lq_start=1`、`adain_lq_start=4`、`wavelet_lq_start=0`，判断首帧问题是否来自 colorfix 对齐。
  - G：输出 `takano04_projector_chunk_stats.csv`。
- G 的关键结果：
  - 89 帧输入经过 Stage2 projector 后输出 `22` 个 chunk；
  - 每个 chunk `3840` tokens；
  - chunk std 约 `0.0844-0.0857`，l2 约 `205-208`；
  - 未见某个 chunk 数值突然爆掉。
- 当前判断：
  - LQ projector 边界可能参与了分段，但 projector 统计本身没有明显崩坏；
  - 结合 Probe C dense full 不跳，优先怀疑 `block_sparse_chunk_causal` attention 的 mask / local window / top-k block pair 选择仍未完全对齐官方实现。
- 文档：
  - `doc/flashvsr_stage2_v6_jump_probe_20260509.md`

### 2026-05-09：合并 A-G、历史 v6.1/v6.2 与 FlashVSR official 逐帧对比

- 用户观察：
  - 除了 Probe C `dense_full` 外，其余 A/B/D、历史 `v6.1`、历史 `v6.2` 都有明显 chunk 级跳变；
  - FlashVSR official baseline 也有轻微边界变化，但幅度小很多，肉眼不明显。
- 本轮整理：
  - 发现用户已将 A-D 原结果移动到：
    - `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_20260509`
  - 将历史 `v6.1 / v6.2 step-10000` 加入同一对比集合。
- 最终本地整理目录：
  - `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509`
- 目录内容：
  - `videos_10_fullset/`：A-D、历史 `v6.1/v6.2` 的 10-video 输出；
  - `videos_takano04_only/`：A-F、历史 `v6.1/v6.2` 的 `takano_04` 单视频；
  - `frames_takano04/`：A-F、历史 `v6.1/v6.2`、FlashVSR official 逐帧；
  - `diagnostics/`：G projector chunk stats；
  - `contact_sheets/takano04_boundary_frames_A_to_F_v61_v62_flashvsr.png`：关键边界帧总览图。
- 代码清理：
  - 从正式 `v6.1/v6.2` inference 入口里移除临时 ablation 参数 `--color_fix_lq_start_frame` 和 offset colorfix 逻辑；
  - 正式推理恢复为：若 SR/LQ 帧数不同，则按前 `min(sr,lq)` 帧对齐做 colorfix。
- 当前结论：
  - `dense_full` 稳定，chunk sparse/causal 路径跳变；
  - 跳变不是单纯由 KV cache 或 top-k 太小导致；
  - 更可能是 chunk sparse/causal attention 的 mask / official kernel / training-inference 对齐细节还没完全复现。

### 2026-05-10：`v6` Stage2 Probe H，官方式 sparse mask / top-k 对齐测试

- 目的：
  - 针对 A/B/D/v6.1/v6.2 都有 chunk 级跳变，而 C `dense_full` 不跳的问题，继续验证是否是当前 top-k block pair 选择与 FlashVSR 官方实现不一致导致。
- 本地代码改动：
  - `diffsynth/models/wan_video_dit_stage2_v6_1.py`
    - 新增 `block_sparse_official_mask_attention(...)`；
    - 新增官方式 spatial local mask；
    - 将 top-k 从“每个 query block 独立选”改为“按 temporal chunk 聚合 spatial blocks 后选”，更接近官方 `generate_draft_block_mask()` 的思想。
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1.py`
    - `--stage2_attention_mode` 增加 `block_sparse_official_mask`。
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_2.py`
    - 同步支持 `block_sparse_official_mask`。
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_1_lora.py`
    - parser choices 同步增加 `block_sparse_official_mask`，后续训练可复用。
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_2_probeH_takano04_on_my_20260510.sh`
    - 新增 H 一键推理脚本，固定使用 `/mnt/conda_envs/flashvsr/bin/python`，避免误用 `b200` 环境。
- 固定 checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_89f_step500_20260508/step-10000.safetensors`
- 固定输入：
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq/takano_04_lq.mp4`
- H 设置：
  - `mode=full_dit_mask_no_kvcache`
  - `stage2_attention_mode=block_sparse_official_mask`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=11`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
- 远端输出：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_probeH_takano04_20260510`
- 本地整理：
  - `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeH_takano04_20260510`
  - 合并到 `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_probeABCDEFG_v61v62_20260509`
  - H 视频：`videos_takano04_only/H_official_mask_topk2_local11.mp4`
  - H 拆帧：`frames_takano04/H_official_mask_topk2_local11`
  - 总览图：`contact_sheets/takano04_boundary_frames_A_to_H_v61_v62.png`
- 结果：
  - H 没有恢复到 C `dense_full` 的稳定状态，视觉上仍更接近 sparse/chunk 路径；
  - 这说明跳变不是“top-k 没按官方式 chunk 分组选”这一点单独造成；
  - 当前更应该怀疑官方 CUDA block-sparse kernel 内部细节、训练/推理 sparse 路径差异，或 sparse/chunk causal 本身需要后续 pixel/distillation 约束。
- 结论：
  - 不建议在当前 H 结果下直接重训同类 sparse Stage2；
  - C `dense_full` 应继续作为稳定上界；
  - 如果继续 sparse 方向，需要先确认官方训练到底是否与推理用同一个 sparse block pair/kernel 逻辑。

### 2026-05-10：建立稳定实验登记表并备份 89f Stage1/Stage2 主线产物

- 目的：
  - 仓库代码和实验分支已经变多，需要区分普通 worklog 与“可进入下一阶段”的稳定实验记录；
  - 用户要求明确 Stage1 `v5.3.5` 稳定母本和 Stage2 `v6` 训练分别对应的代码、启动脚本、config、实验目录、机器和 ckpt。
- 新增稳定登记文档：
  - `doc/flashvsr_stable_experiment_registry.md`
- 登记的 Stage1 稳定实验：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300`
  - 主机：`b8gkuie2ns`
  - 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
  - sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-RandomProj-Img5-NonStreamProj-Aligned23-ResumeStep3000-Seed20260501.sh`
  - yaml：`wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_resume_step3000_seed20260501.yaml`
  - 稳定 ckpt：`output/step-10000.safetensors`
- 登记的 Stage2 稳定实验：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2`
  - 主机：`b8gkuie2ns`
  - 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_lora.py`
  - sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh`
  - yaml：`wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml`
  - 稳定 ckpt：`output/step-10000.safetensors`
- 备份策略：
  - 只备份最小可复用产物：`step-10000.safetensors`、`run.log`、`launch_command.sh`、`snapshot/`；
  - Stage1 额外备份 `output/validation/step-10000/`；
  - 不备份 `output/training_state/`、DeepSpeed optimizer/rng/scheduler 等断点续训状态。

### 2026-05-10：整理 Stage1/Stage2 稳定母本的 clean 代码入口并完成 b8 smoke

- 目的：
  - 原稳定母本代码在多轮 debug 中保留了较多开关、tensor dump、preview、branch/GC 打印等旁路逻辑；
  - 用户要求保留原母本不动，单独整理出 clean 版代码、yaml、sh，作为后续继续实验的干净入口。
- 新增 Stage1 clean 文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_clean_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_48gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-48GPU-v5-3-5-Clean-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-NonStreamProj-Aligned23.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_smoke_2gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-Smoke-2GPU-v5-3-5-Clean-Lora-89f-FullSources-bs1-lr1e5-AliyunDegra-NonStreamProj-Aligned23.sh`
- 新增 Stage2 clean 文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_clean_lora.py`
  - `diffsynth/models/wan_video_dit_stage2_v6_clean.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-Clean-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-Clean-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh`
- 本地验证：
  - clean 训练入口 `py_compile` 通过；
  - clean sh `bash -n` 通过。
- 远端同步与验证：
  - 检查并恢复本机 `sync` tmux 中 `b8gkuie2ns` 同步窗口；
  - 远端 `/mnt/task_runtime/lucidvsr` 中 clean 文件已同步；
  - 远端 `py_compile` 和 `bash -n` 通过。
- Stage1 clean smoke：
  - 机器：`b8gkuie2ns`
  - 第一次失败原因：b8 上 8 张卡仍有占卡 Python 进程，每张约 `166GB` 显存，导致 2GPU smoke OOM；
  - 清空占卡进程后用同一套配置重跑成功；
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_v5_3_5_clean_lora_89f_fullsources_bs1_lr1e5_aliyundegra_nonstreamproj_aligned23_20260510_clean_stage1_smoke2`
  - 结果：`step=1 loss=0.034188`，`step=2 loss=0.329382`。
- Stage2 clean smoke：
  - 机器：`b8gkuie2ns`
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_smoke_2gpu_v6_clean_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260510_clean_stage2_smoke`
  - 关键确认：
    - `lq_proj_in` 从 Stage1 `step-10000` 导入，`keys=8, missing=0, unexpected=0`；
    - DiT LoRA 从 Stage1 `step-10000` 导入，`keys=480`；
    - `Stage2 v6 attention mode: block_sparse_chunk_causal`；
    - `Stage2 v6 topk_ratio: 2.0`。
  - 结果：`step=1 loss=0.270946`，`step=2 loss=0.008904`。
- 文档更新：
  - `doc/flashvsr_stable_experiment_registry.md` 增加 clean 版 code/sh/yaml/smoke 记录；
  - 本 worklog 增加 clean 版整理与 smoke 结果。

### 2026-05-10：补测 Stage2 早期 checkpoint，并启动低学习率 v6.3 对照实验

- 背景：
  - 用户观察 Stage2 sparse/chunk 推理存在 chunk 边界跳变，需要确认早期 checkpoint 是否已经出现该问题；
  - 同时怀疑 Stage2 `lr=1e-5` 可能偏大，要求即使论文未明确降低学习率，也启动一个更小学习率的 48GPU 对照实验。
- 论文核对：
  - 重新查阅 `FlashVSR/2510.12747v1.pdf`；
  - 论文 4.1 Training Details 写明三个 stage 都使用 AdamW，`learning rate=1e-5`，`weight decay=0.01`；
  - 论文没有明确说 Stage2 需要降低学习率。
- 早期 checkpoint 测试：
  - 测试机器：`myj7ukyewz`
  - 测试代码：`wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_early_ckpts_on_my_20260510.sh`
  - 测试推理：`v6.1` streaming / KV-cache 版本；
  - 测试 ckpt：`step-10, step-50, step-100, step-500, step-1000, step-1500, step-2000, step-2500, step-3000, step-3500, step-4000, step-4500, step-5000`
  - 源 checkpoint 目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_20260506_014800_stage2_v6_48gpu_worker2/output`
  - 远端输出：
    `/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_early_ckpts_v61_20260510`
  - S3 备份：
    `s3://lxh/tmp/stage2_v6_early_ckpts_v61_20260510`
  - 本地下载：
    `/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_early_ckpts_v61_20260510`
  - 下载结果：13 个 checkpoint，每个 10 个视频，共 130 个 `mp4`。
- v6.3 低学习率对照：
  - 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_1_lora.py`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_3_lora_89f_videoonly_bs1_lr3e6_blocksparse_worker2_val.yaml`
  - sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-3-Lora-89f-VideoOnly-bs1-lr3e6-BlockSparse-Worker2-Val.sh`
  - 继承 checkpoint：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
  - 关键设置：`lr=3e-6`，`num_frames=89`，`batch_size=1`，`dataset_num_workers=2`，`validation_num_samples=3`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_topk_ratio=2.0`，`stage2_local_num=-1`。
  - validation 路径：使用 `v6.1` 的 streaming / KV-cache inference validation。
  - 48GPU 机器：
    `b8gkuie2ns, wfnwbym4v6, kh5idf7f98, hj65iqg9rh, zhki5rrddw, xwk6qjuej5`
  - 实验目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v6_3_lora_89f_videoonly_bs1_lr3e6_blocksparse_worker2_val_20260510_180600`
  - 启动前发现本机 `sync` 中 `wfn/kh/hj/zh/xwk` 五个窗口自 5 月 8 日已断，导致从机缺少新的 v6.3 sh；
  - 已恢复这 5 个同步窗口，并确认从机 `V63_SH_PRESENT` 后重启；
  - 启动结果：wandb 已开启，`lq_proj_in` 与 LoRA 均从 Stage1 `step-10000` 导入成功；
  - 前 9 步 loss：`0.084352, 0.086376, 0.101635, 0.116139, 0.085866, 0.071553, 0.120919, 0.061436, 0.093177`；
  - b8 观察显存约 `143GB/GPU`，多数 GPU 利用率达到 `100%`。

### 2026-05-11：恢复本机 sync，并整理 Stage3 DMD 训练计划

- 本机 `sync` tmux：
  - 检查到 9 个窗口覆盖当前机器：`b8gkuie2ns, wfnwbym4v6, kh5idf7f98, hj65iqg9rh, zhki5rrddw, xwk6qjuej5, dd79sgu25m, myj7ukyewz, tjmjauaq4d`；
  - 发现 `dd79sgu25m` 与 `tjmjauaq4d` 同步窗口已掉；
  - 将 9 个窗口改为循环同步形式：每 30 秒执行一次 `bolt task sync <task_id>`，避免一次同步结束后窗口空置；
  - 当前 `tjmjauaq4d` 返回 `Task terminated -- aborting.`，同步循环会持续重试。
- Stage3 论文核对：
  - 论文路径：`/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/2510.12747v1.pdf`；
  - 明确 Stage3 名称为 `Distribution-Matching One-Step Distillation`；
  - Stage3 loss 包含 `DMD + flow matching + pixel MSE + LPIPS`；
  - 论文给出 LPIPS 权重 `lambda=2`；
  - 论文写明 Stage3 采用和 Stage2 一样的视频训练设置，不再使用 image branch。
- 新增文档：
  - `doc/flashvsr_stage3_dmd_plan_20260511.md`
  - 内容包括 Stage3 与 Stage2 的关系、`G_one / G_real / G_fake` 三模型角色、DMD / FM / MSE / LPIPS 的实现路线、需要外部 DMD 代码对齐的风险点、以及建议新增的 `v7` 代码/config/sh 清单。

### 2026-05-11：补测 v6.3 低学习率 Stage2 checkpoint

- 用户要求：
  - 对新训练的 `v6.3` 实验按每隔 400 step 做一次测试；
  - 测试方式沿用之前 `v6.1` Stage2 streaming / KV-cache inference；
  - 测完下载到桌面，并恢复 `myj7ukyewz` 占卡。
- 新增测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_3_scan89_step400_on_my_20260511.sh`
  - 该脚本只在本地仓库新增，通过 `sync` 同步到远端；
  - 使用 `/mnt/conda_envs/flashvsr/bin/python`；
  - 使用 `infer_flashvsr_stage2_v6_1_batch`；
  - 关键推理设置：`num_frames=89`，`input_bicubic_upscale=4.0`，`color_fix_method=adain`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_topk_ratio=2.0`，`stage2_local_num=-1`，`stage2_kv_ratio=3.0`。
- checkpoint 来源：
  - `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_48gpu_v6_3_lora_89f_videoonly_bs1_lr3e6_blocksparse_worker2_val_20260510_180600/output`
  - 当前云端可测 checkpoint：`step-400, step-800, step-1200, step-1600, step-2000`。
- 测试机器：
  - `myj7ukyewz`
  - 远端输出：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_3_scan89_v61_step400_20260511`
  - S3 备份：`s3://lxh/tmp/stage2_v6_3_scan89_v61_step400_20260511`
  - 本地下载：`/Users/lixiaohui/Desktop/stage2_and_jump probe/stage2_v6_3_scan89_v61_step400_20260511`
- 测试结果：
  - 5 个 checkpoint 均完成；
  - 每个 `step-*` 输出 10 个视频；
  - `v6.1` streaming 推理输出 85 帧，日志中 `color_fix` 出现 `sr=85 lq=89 using=85` 属于预期行为。
- 资源状态：
  - 测试结束后 `myj7ukyewz` 已恢复 `occupy`；
  - 8 张 GPU 均约 `166GB` 显存占用，利用率 `100%`。
- 额外说明：
  - 本机 `conductor` 下载因 Notary 凭证缺失失败；
  - 最终使用 `bolt task scp -r` 从 `myj7ukyewz` artifacts 直接下载到桌面。

### 2026-05-11：新增 Stage2 v6.4 首帧对齐训练版本

- 背景判断：
  - 重新核对 FlashVSR 官方 `Causal_LQ4x_Proj` 与推理代码后，确认官方 LQ 侧会用首帧 repeat 3 次构成第一个 warmup block；
  - 第一个 warmup block 只建立 causal conv cache，不输出 LQ condition latent；
  - 后续每 4 帧输出一个 LQ condition latent；
  - 官方推理输入会在尾部 repeat 4 帧，并打印 `Target Frames (8n-3)`，即 `89` 帧 pipeline 输入对应 `85` 帧有效输出。
- v6 / v6.1 问题：
  - 旧 Stage2 写法是 `GT 89 -> WAN VAE 23 -> drop z0 -> target 22`；
  - 该写法会把本来不是首 latent 的 `z1` 放到 22-latent 序列首位监督，和 decoder 将首 latent 当特殊首帧 latent 处理的语义存在风险。
- v6.4 改法：
  - LQ / noise 仍按 `num_frames=89` 得到 `22` 个 streaming latent 位置；
  - GT target 不再从 89 帧 VAE 结果里裁掉 `z0`；
  - GT target 改为取前 `target_frames = noise_latents * 4 - 3 = 85` 帧直接过 WAN VAE；
  - 因此 target 直接得到 `22` 个 latent，并保留第一个 latent 参与 flow loss；
  - 这个版本对应用户提出的“LQ 未来 4 帧信息 + 首帧 cache，用来预测当前 target latent”的实验假设。
- 新增代码：
  - 训练代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
  - 48GPU 配置：`wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml`
  - 48GPU 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v6-4-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`
  - 40GPU 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`
  - 40GPU accelerate 模板：`wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_5node40gpu_nooffload.template.yaml`
  - 2GPU smoke 配置：`wanvideo/model_training/flashvsr/configs/history/stage2_release_smoke_2gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2.yaml`
  - 2GPU smoke 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-Smoke-2GPU-v6-4-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2.sh`
- 验证：
  - 本地已执行 `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`；
  - 已确认 v6.4 代码里不存在旧的 `input_latents = input_latents[:, :, 1:]` 裁剪逻辑。

### 2026-05-11：`v6.4` 40GPU 正式启动，绕开 `wfnwbym4v6` D-state 节点

- 背景：
  - 48GPU 启动时 `wfnwbym4v6` 出现 D-state 进程卡死，`kill -9` / `gpu-reset` / `bolt retry` 均无法恢复；
  - 用户决定本轮不再使用该节点，改用 5 个健康节点做 40GPU 启动。
- 使用节点：
  - `b8gkuie2ns`
  - `kh5idf7f98`
  - `hj65iqg9rh`
  - `zhki5rrddw`
  - `xwk6qjuej5`
- 启动脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`
- 训练代码：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
- 配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml`
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260511_40gpu_v64_lr1e5_r2`
- 初始化来源：
  - Stage1 `v5.3.5` 89f checkpoint：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- 关键确认：
  - `Stage2 v6 loaded lq_proj_in ... keys=8, missing=0, unexpected=0`
  - `Stage2 v6 loaded LoRA ... keys=480`
  - `Stage2 v6 attention mode: block_sparse_chunk_causal`
  - `Stage2 v6 topk_ratio: 2.0`
  - `Prepared 3 fixed Stage2 validation samples.`
  - WandB run：`train_stage2_release_40gpu_v6_4_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260511_40gpu_v64_lr1e5_r2`
- 训练状态：
  - 已出 loss：`step=1 loss=0.088556`，`step=2 loss=0.064814`；
  - 已触发 `step-10` checkpoint；
  - validation 目录已出现：`output/validation/step-10/sample_000`；
  - 各节点 GPU 利用率检查时基本在 `90-100%`，单卡显存约 `142-143GB`。

### 2026-05-12：启动 `v6.4.1`，将 Stage2 top-k 改为官方对齐的 chunk-grouped 版本

- 背景：
  - 复盘 `v6.1/v6.2` 与 probe A-H 后，确认跳变问题不能只归因于学习率；
  - 旧 Stage2 sparse top-k 更偏按全局 block/query 组织，和官方口径“先按 chunk causal/local 合法区域，再在合法 block pair 内做 top-k”不够一致；
  - 用户明确要求不要把 `local spatial mask` 加入训练，因为它属于超高分辨率推理技巧。
- 代码改动：
  - 文件：
    - `diffsynth/models/wan_video_dit_stage2_v6.py`
    - `diffsynth/models/wan_video_dit_stage2_v6_clean.py`
    - `diffsynth/models/wan_video_dit_stage2_v6_1.py`
  - `_select_topk_blocks(...)` 改成 chunk-grouped top-k：
    - Q/K 仍按 `(2,8,8)` block 做 pooling；
    - 先应用 chunk causal / temporal allowed mask；
    - 再按 `(batch, head, chunk)` 维度，把该 chunk 下所有合法 spatial block pair 放在一起选 top-k；
    - 没有加入 spatial local mask；
    - block-sparse path 仍保留 dense fallback 的遮罩防护，但当前训练目标是进入 block-sparse chunk causal 路径。
- 本地验证：
  - 已通过：
    - `python -m py_compile diffsynth/models/wan_video_dit_stage2_v6.py diffsynth/models/wan_video_dit_stage2_v6_clean.py diffsynth/models/wan_video_dit_stage2_v6_1.py`
- 新增配置与启动脚本：
  - config：`wanvideo/model_training/flashvsr/configs/history/stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val.yaml`
  - sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-40GPU-v6-4-1-Lora-89f-VideoOnly-bs1-lr1e5-BlockSparse-Worker2-Val.sh`
- 启动节点：
  - `b8gkuie2ns`
  - `kh5idf7f98`
  - `hj65iqg9rh`
  - `zhki5rrddw`
  - `xwk6qjuej5`
  - 继续绕开 `wfnwbym4v6`。
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100`
- 初始化来源：
  - Stage1 `v5.3.5` 89f checkpoint：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- 关键设置：
  - `num_frames=89`
  - `batch_size=1`
  - `learning_rate=1e-5`
  - `dataset_num_workers=2`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - validation 使用 v6.1 streaming KV-cache 路径。
- 启动确认：
  - `Stage2 v6 loaded lq_proj_in ... keys=8, missing=0, unexpected=0`
  - `Stage2 v6 loaded LoRA ... keys=480`
  - `Stage2 v6 attention mode: block_sparse_chunk_causal`
  - `Stage2 v6 topk_ratio: 2.0`
  - `Prepared 3 fixed Stage2 validation samples.`
  - WandB run：`train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100`
- 当前训练状态：
  - 已出 loss：
    - `step=1 loss=0.100667`
    - `step=2 loss=0.129647`
    - `step=9 loss=0.098759`
  - 已触发 `step-10` validation，`sample_000` 已生成 `hr.mp4` / `lq.mp4` / `sr.mp4` / `meta.json`；
  - 主机 `b8gkuie2ns` 采样时 8 卡均 `100%`，显存约 `100GB/GPU`；
  - 进入 validation 后主机显存约 `129GB/GPU`，多数 GPU util `100%`，GPU0 在采样时约 `68-88%`；
  - 从节点 `kh/hj/zh/xwk` 均已进入训练并占用约 `102-103GB/GPU`，GPU util 在采样时约 `40-100%` 波动。

### 2026-05-12：测试 `v6.4.1` Stage2 checkpoint 每 200 步结果

- 测试目标：
  - 对 `v6.4.1` 89f Stage2 实验按 `step-200` 间隔做 v6.1 inference；
  - 先快速查看早期训练结果是否继续存在 chunk 跳变，以及不同 step 的变化趋势。
- 来源实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100`
- 测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_v641_step200_on_my_20260512.sh`
- 测试机器：
  - `myj7ukyewz`
- 测试设置：
  - 输入：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - 推理：`infer_flashvsr_stage2_v6_1_batch`
  - `num_frames=89`
  - `num_inference_steps=50`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `stage2_kv_ratio=3.0`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
- 已测试 checkpoint：
  - `step-200`
  - `step-400`
  - `step-600`
  - `step-800`
  - `step-1000`
  - `step-1200`
  - `step-1400`
  - `step-1600`
  - `step-1800`
- 输出位置：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_scan89_v61_step200_20260512`
  - 云端：`s3://lxh/tmp/stage2_v6_4_1_scan89_v61_step200_20260512`
  - 本地：`/Users/lixiaohui/Desktop/stage2_v6_4_1_scan89_v61_step200_20260512`
- 完成状态：
  - 共 `100` 个 mp4；
  - 每个 step 目录均为 `10` 个视频；
  - 测试完成后 `myj7ukyewz` 已恢复 8 卡占卡程序，8 张卡均显示约 `166GB` 显存占用与 `100%` GPU util。

### 2026-05-12：补测 `v6.4.1 step-1800` 的 `85 + tail repeat 4` 推理对照

- 目的：
  - 用户观察到当前 89f v6.1 推理尾帧可能变坏；
  - 需要确认是否和“测试集原本就是 89 帧，而不是官方式 85 帧输入后复制最后一帧补到 89 帧”有关。
- 测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_v641_step1800_tailpad85_on_my_20260512.sh`
- 输入构造：
  - 源输入：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - 新输入：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_tailpad85_20260512/lq`
  - 对每个视频执行：
    - 取前 `85` 帧；
    - 复制第 `85` 帧 `4` 次；
    - 得到新的 `89` 帧输入。
- checkpoint：
  - `v6.4.1 step-1800`
  - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage2_v6_4_1_89f_step200_20260512/step-1800.safetensors`
- 推理设置：
  - `infer_flashvsr_stage2_v6_1_batch`
  - `num_frames=89`
  - `num_inference_steps=50`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `stage2_kv_ratio=3.0`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
- 输出位置：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_step1800_v61_tailpad85_20260512`
  - 云端：`s3://lxh/tmp/stage2_v6_4_1_step1800_v61_tailpad85_20260512`
  - 本地：`/Users/lixiaohui/Desktop/stage2_v6_4_1_step1800_v61_tailpad85_20260512`
- 完成状态：
  - `step-1800_tailpad85` 下共 `10` 个 SR 视频；
  - 下载到桌面后总 mp4 为 `20` 个，其中包含 `10` 个 padded 输入与 `10` 个输出；
  - 测试完成后 `myj7ukyewz` 已恢复 8 卡占卡程序。

### 2026-05-12：按官方有效帧规则重测 `v6.4.1` 从 `step-1800` 开始的 checkpoint

- 目的：
  - 明确 `v6.4.1` 后续测试统一采用 FlashVSR stage2 的官方有效输出规则；
  - 对 89 帧输入，正式结果只保留 85 帧有效输出，不把最后 4 帧 buffer 区放入对比结果。
- 测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage2_v6_1_scan89_v641_step200_on_my_20260512.sh`
- 运行参数：
  - `MIN_STEP=1800`
  - `STEP_MOD=200`
  - `OUTPUT_ROOT=/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_official85f_from1800_20260512`
  - `S3_OUTPUT_DIR=s3://lxh/tmp/stage2_v6_4_1_official85f_from1800_20260512`
- 测试设置：
  - 输入：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - `num_frames=89`
  - `num_inference_steps=50`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `stage2_kv_ratio=3.0`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
- 已测试 checkpoint：
  - `step-1800`
  - `step-2000`
  - `step-2200`
  - `step-2400`
  - `step-2600`
  - `step-2800`
  - `step-3000`
- 输出位置：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage2_v6_4_1_official85f_from1800_20260512`
  - 云端：`s3://lxh/tmp/stage2_v6_4_1_official85f_from1800_20260512`
  - 本地：`/Users/lixiaohui/Desktop/stage2_v6_4_1_official85f_from1800_20260512`
- 完成状态：
  - 每个 step 目录均为 `10` 个输出视频；
  - 本地抽样确认每个输出视频为 `85` 帧；
  - 测试完成后 `myj7ukyewz` 已恢复 8 卡占卡程序。

### 2026-05-13：阅读 DMD2 / OSEDiff 论文并补充 Stage3 v7 设计文档

- 参考论文：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/DMD2/2405.14867v2.pdf`
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/OSEDiff/2406.08177v3.pdf`
- 本次只读论文，不梳理代码实现。
- 主要结论：
  - DMD2 的关键是 real score 与 fake score 的差，`G_fake` 必须动态学习 fake distribution；
  - DMD2 强调 two time-scale update rule，正式 v7 中 `G_fake` 应有独立 optimizer，`fake_update_ratio` 初始建议至少为 `5`；
  - DMD2 的 GAN loss 不在 FlashVSR Stage3 公式中，v7 第一版不主动加入；
  - OSEDiff 对 v7 的价值主要是 `MSE + LPIPS` reconstruction 和 latent-space regularization 思想，不能直接照搬 SD 图像模型结构；
  - v7 的稳妥实现顺序应为：先 one-step + pixel/LPIPS smoke，再接 `G_fake` 更新，最后接真正 DMD loss。
- 已更新文档：
  - `doc/flashvsr_stage3_dmd_plan_20260511.md`
  - 新增 `2026-05-13：阅读 DMD2 / OSEDiff 论文后的实现判断` 一节。

### 2026-05-13：恢复 8 台机器 sync，并新增 Stage3 `v7-A` 训练入口

- 恢复本机固定同步会话：
  - tmux session：`sync`
  - 覆盖机器：`b8gkuie2ns wfnwbym4v6 kh5idf7f98 hj65iqg9rh zhki5rrddw xwk6qjuej5 dd79sgu25m myj7ukyewz`
  - 每个窗口执行：`cd ~/Library/CloudStorage/Box-Box/code && bolt task sync <machine>`
- 新增 Stage3 `v7-A` 本地代码，未启动 smoke：
  - 训练代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
  - 配置：`wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon.yaml`
  - 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`
- `v7-A` 目标：
  - 只打通 one-step student + Wan decoder + `MSE + 2 * LPIPS`；
  - 复用 Stage2 sparse-causal student 结构；
  - 暂不接 `G_real/G_fake/DMD/GAN`。
- 本地静态检查：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`
  - YAML 解析通过。

### 2026-05-13：Stage3 `v7-A` 2 卡 smoke 与验收结论

- 远端机器：
  - `dd79sgu25m`
- smoke 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_smoke_fix`
- 使用 checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-3300.safetensors`
  - 该文件从 `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/.../step-3300.safetensors` 拉到 `dd79sgu25m`
- 本次 smoke 前发现的问题：
  - `lpips` 不在 `flashvsr` 环境中，已用 `/mnt/conda_envs/flashvsr/bin/python -m pip install lpips` 安装；
  - 初版 `v7-A` 直接 decode 完整 89 帧 latent，Wan decoder 在 1280x768 下 OOM。
- 修复：
  - `Stage3AOneStepReconLoss` 改为：
    - `flow loss` 仍监督整段 latent；
    - `MSE/LPIPS` 只 decode temporal prefix；
    - 当前 smoke 设置 `stage3_recon_num_latents=2`，实际 `decoded_frames=5`。
- 已确认：
  - Stage2 projector 导入成功：`lq_proj_in keys=8, missing=0, unexpected=0`；
  - Stage2 LoRA 导入成功：`LoRA keys=480`；
  - 训练参数导出包含 `488` 个 key，其中 `lq_proj_in` 存在、LoRA key 为 `480`；
  - 第一条 Stage3 子 loss 已输出：
    - `loss=0.422148`
    - `flow=0.184295`
    - `mse=0.004079`
    - `lpips=0.116887`
    - `recon_latents=2`
    - `decoded_frames=5`
- 当前未完全解决：
  - `step-1.safetensors` 已保存成功；
  - 但 Deepspeed `accelerator.save_state()` 在保存 frozen param 状态时报错：
    - `ValueError: failed to find frozen Parameter ... in named params`
  - 判断原因是 `v7-A` 在 forward 内使用 frozen VAE decode，Deepspeed 保存训练状态时把 VAE frozen param 纳入检查，但名字映射不完整。
- 后续处理：
  - `v7-A` 的 one-step + pixel/LPIPS 逻辑已经通过一次 forward/backward 证明可用；
  - 正式长训前必须修复或绕开 Deepspeed training state 保存问题；
  - 更推荐把 Wan decoder / LPIPS 作为 Stage3 loss-only frozen module 管理，不让 Deepspeed 保存它们的 optimizer/training state。

### 2026-05-13：Stage3 `v7-A` pixel loss 增加首帧 4 倍权重

- 背景：
  - Wan VAE 对首帧有特殊不压缩处理；
  - Stage3 解码后计算 pixel loss 时，如果首帧不加权，首帧监督与后续压缩段不对齐，容易造成首帧漂移。
- 代码改动：
  - 文件：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
  - 新增参数：
    - `stage3_first_frame_pixel_weight`
    - `stage3_first_frame_lpips_weight`
  - 默认配置：
    - `stage3_first_frame_pixel_weight=4.0`
    - `stage3_first_frame_lpips_weight=1.0`
  - 当前只对 pixel/MSE loss 做首帧 4 倍加权；
  - LPIPS 暂不放大首帧，但保留独立参数，后续可单独开启。
- 实现细节：
  - MSE 按逐帧 loss 计算；
  - 第 0 帧 loss 乘以 `4.0`；
  - 最终仍按原始帧数平均，因此首帧 pixel 梯度确实是普通逐帧平均下的 4 倍，而不是用权重和重新归一化。
- 本地检查：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`

### 2026-05-13：Stage3 `v7-A` 修复 Deepspeed state 保存与 resume 验证

- 问题：
  - 初版 `v7-A` 在 forward 内把 `LPIPS` 赋到 `pipe._stage3_lpips`；
  - 这会把 LPIPS/VGG frozen 参数注册进 Deepspeed 管理的 module tree；
  - `accelerator.save_state()` 保存 optimizer/state 时找不到这些后注册 frozen 参数对应的 named param，报错：
    - `ValueError: failed to find frozen Parameter ... in named params`
- 修复：
  - 文件：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_a_lora.py`
  - 新增 `_get_stage3_lpips(pipe, net)`；
  - LPIPS 现在作为 loss-only cache 存在，不再注册成 `nn.Module` 子模块；
  - 如果旧属性 `_stage3_lpips` 已进入 `_modules`，会显式移除，避免污染 Deepspeed state。
- 同时修复启动脚本：
  - 文件：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-A-Lora-89f-VideoOnly-OneStepRecon.sh`
  - 新增 `EXTRA_ARGS` 透传；
  - 允许后续 smoke / resume 显式追加 `--resume_training_state_dir ...`，避免环境变量被静默丢弃。
- 完整 smoke 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_smoke_statefix`
- 验证结果：
  - 2 卡训练完整跑到 `max_train_steps=20`；
  - 成功保存 `step-1 / step-2 / step-5 / step-10 / step-20.safetensors`；
  - 成功保存对应 `training_state/step-1 / step-2 / step-5 / step-10 / step-20`；
  - 无 `Traceback / RuntimeError / ValueError`。
- resume 验证目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_a_lora_89f_videoonly_onestep_recon_20260513_v7a_true_resume_from_step10`
- resume 验证结果：
  - `launch_command.sh` 确认包含：
    - `--resume_training_state_dir .../v7a_smoke_statefix/output/training_state/step-10`
  - 日志确认：
    - `[resume] training state loaded step=10 epoch_id=0`
  - 从 step 10 继续训练到 step 20；
  - 成功保存新的 `step-20.safetensors` 和 `training_state/step-20`。
- 当前结论：
  - `v7-A` 已经不是“只能 forward/backward”的临时代码；
  - 当前版本已经验证了 trainable checkpoint 保存、Deepspeed training state 保存、以及 training state resume；
  - 下一步可以在此基础上进入 `v7-B` 的 `G_fake / DMD` 逻辑，而不是继续修基础训练链路。

### 2026-05-13：Stage3 `v7-B` 独立分支与 2 卡 smoke

- 目的：
  - 保留 `v7-A` 作为稳定 one-step reconstruction 分支；
  - 复制出 `v7-B`，开始承接后续 `G_fake / DMD` 逻辑；
  - 避免在 `v7-A` 上继续叠实验代码。
- 新增文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-B-Lora-89f-VideoOnly-FakeFM.sh`
- 当前实现边界：
  - `v7-B` 仍复用 `v7-A` 的 one-step student + Wan decoder reconstruction loss；
  - 新增 `stage3_fake_checkpoint / stage3_fake_fm_weight / stage3_fake_update_ratio` 参数；
  - `stage3_fake_fm_weight` 默认 `0.0`；
  - 如果误设为非 0，会直接报错；
  - 原因是当前 `launch_training_task` 只有一个 optimizer，不能正确实现 DMD2 那种 `student/G_fake` 双模型、双 optimizer、交替更新。
- smoke 机器：
  - `dd79sgu25m`
- Stage2 初始化：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-3300.safetensors`
- smoke 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_20260513_v7b_smoke`
- 验证结果：
  - `py_compile` 通过；
  - 启动脚本 `bash -n` 通过；
  - Stage2 `lq_proj_in` 导入成功：`keys=8, missing=0, unexpected=0`；
  - Stage2 LoRA 导入成功：`keys=480`；
  - 已输出 loss 分量：
    - `flow`
    - `mse`
    - `lpips`
  - 完成 `20` step；
  - 保存 `step-1 / step-2 / step-5 / step-10 / step-20.safetensors`；
  - 保存 `training_state/step-1 / step-2 / step-5 / step-10 / step-20`。
- 代表性日志：
  - `Stage3 v7-B: one-step student + G_fake scaffold + Wan decoder reconstruction smoke path`
  - `stage3_fake fm_weight=0.0 update_ratio=5 checkpoint=None`
  - `step=20 loss=0.669951`
- 后续：
  - 如果要做真正 `v7-B` DMD，需要新增专用 dual-optimizer runner；
  - 该 runner 至少要支持：
    - student/generator optimizer；
    - `G_fake` optimizer；
    - `fake_update_ratio`；
    - DMD loss 中 `G_real/G_fake` 预测差分；
    - fake score model 的独立保存与 resume。

### 2026-05-13：Stage3 文档审计，确认 `v7-A/v7-B` 与论文仍有关键差距

- 用户指出：
  - 论文里 pixel/LPIPS loss 不是固定 decode 前缀；
  - 正确做法是每个 iteration 随机选 2 个 latent decode；
  - 未选 latent 只走 DMD/FM/flow；
  - 还必须处理 Wan decoder 第 0 latent 的首帧特殊语义。
- 代码核对：
  - `v7-A` 当前代码：
    - `z_decode = z_pred[:, :, :recon_num].contiguous()`
  - `v7-B` 当前代码：
    - 同样固定取前缀；
  - 因此两者都只是 fixed-prefix one-step reconstruction smoke，不是论文版 Stage3 random latent decode。
- 文档更新：
  - 文件：`doc/flashvsr_stage3_dmd_plan_20260511.md`
  - 新增：
    - FlashVSR / DMD2 / OSEDiff 论文和代码证据库；
    - Stage3 强制自查表；
    - `v7-A` 逐项审计；
    - `v7-B` 逐项审计；
    - 下一步 random latent decode 实现顺序和验收标准。
- 当前结论：
  - `v7-A` 保留为稳定 one-step + fixed-prefix pixel/LPIPS smoke；
  - `v7-B` 接下来先修 random latent decode 和 GT 对齐；
  - 修完 random latent decode 后，再进入真正 `G_fake / DMD` dual-optimizer runner。

### 2026-05-13：Stage3 `v7-B` random latent decode 修正

- 目的：
  - 修正 `v7-A/v7-B` 固定前缀 decode 与 FlashVSR Stage3 论文“每轮随机选 2 个 latent decode”的不一致。
- 第一版改动：
  - `v7-B` 增加 `_sample_stage3_recon_window(...)` 和 `_latent_window_to_frame_range(...)`；
  - 每步随机选择连续 2 个 latent；
  - pixel/LPIPS 只对该 window 对应的 GT frame range 计算；
  - 非全局首帧 window 不再错误使用首帧 4 倍 pixel weight。
- 第一版 smoke 结果：
  - 远端：`dd79sgu25m`
  - 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-B-Lora-89f-VideoOnly-FakeFM.sh`
  - 输出目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_20260513_v7b_randomdecode_smoke`
  - 结果：失败，Wan decoder OOM。
  - 原因：如果随机 window 抽到后部 latent，第一版会 decode `[0, recon_end)` 全 prefix；1280x768 下接近整段 decode，显存不可接受。
- 第二版修正：
  - 新增 `_build_stage3_decode_window(...)`；
  - `recon_start==0` 时仍 decode `[0, recon_end)`；
  - `recon_start>0` 时只 decode `[recon_start-1, recon_end)`，其中 `recon_start-1` detach 作为 Wan decoder causal context；
  - pixel/LPIPS 只裁 selected window 对应 local frames；
  - Wan decoder 改为 `tiled=True`。
- 当前边界：
  - 第二版是显存受控的工程实现；
  - 它符合“选中 latent 才吃 pixel/LPIPS 梯度”的要求；
  - 但它不是完整 previous-prefix exact decode，需要 smoke 和后续作者确认。
- 第二版 smoke 继续暴露的问题：
  - 全 prefix OOM 已缓解，但 VAE tiled decode 反传仍在小额分配时 OOM；
  - 根因是 Wan decoder 虽然 frozen，pixel/LPIPS 仍需要保存 decoder activation 来对 `z_pred` 求梯度。
- 第三步修正：
  - 增加 `_stage3_decode_with_checkpoint(...)`；
  - 对 `pipe.vae.decode(...)` 使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`；
  - 目标是用重算换显存，不改变 Stage3 pixel/LPIPS loss 语义。
- 第三步 smoke 继续暴露的小问题：
  - `pipe.vae.decode(...)` 外层封装会把结果回收到 CPU；
  - `x_gt` 已在当前 GPU 上，MSE 时报 `cuda/cpu` device mismatch；
  - 修复为 `x_pred = _stage3_decode_with_checkpoint(...).to(device=pipe.device)`。

### 2026-05-14：Stage3 `v7-B` 显存排查顺序重定

- 背景：
  - Stage3 继续出现 180GB 级显存压力；
  - 用户指出作者实现 H100 80GB 级别可训练，说明不能只靠 tile/offload 绕过；
  - 当前需要先确认 Stage1/2 底座是否过重，以及 Stage3 decoder 语义是否严格对齐。
- 本地代码改动：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py`
  - 撤掉自定义 spatial tile-level training decode 作为主线；
  - 改成 `_stage3_decode_selected_window_full_frame(...)`：
    - prefix `[0:recon_start)` 只在 `torch.no_grad()` 下推进 Wan decoder cache；
    - selected `[recon_start:recon_end)` 做 full-frame decode 并保留梯度；
    - `stage3_decoder_cpu_offload` 只包 selected decode checkpoint；
    - LPIPS 仍逐帧串行计算。
- 文档：
  - 新增 `doc/flashvsr_stage3_memory_alignment_plan_20260514.md`；
  - 明确后续顺序：
    - 先做小分辨率 correctness smoke；
    - 再做 Stage2 flow-only / one-step z_pred / MSE decode only / MSE+LPIPS 显存分解；
    - 如果 Stage2 底座已经过高，优先查 block sparse kernel、gradient checkpointing、trainable 参数和 debug graph。
- 当前状态：
  - 本地 `py_compile` 已通过；
  - 还未启动新的远端 smoke。

### 2026-05-14：Stage3 `v7-B` full-frame decode smoke 与 worker 扫描

- 远端机器：
  - `6ai5mpi47f`
- 目标：
  - 验证撤掉自定义 tile-level training decode 后，`prefix no-grad cache + selected 2 latent full-frame decode + pixel/LPIPS` 是否能跑通；
  - 同时扫描 `dataset_num_workers`，找 Stage3 当前最稳设置。
- 通过的主 smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_v7b_fullframe_notile_contigfix`
  - 结果：跑到 `step=20`，保存 `step-1/2/5/10/20.safetensors`。
- worker 扫描：
  - `worker=1`：`step1≈168s`，`step2≈120s`，明显慢，停止；
  - `worker=2`：4 step 正常，`step1≈135s`，后续有效 step 约 `31-48s`；
  - `worker=4`：4 step 可保存，但没有速度收益，结束时有 noisy terminate；
  - `worker=8`：OOM，原因是 worker/子进程 CUDA context 和主训练峰值叠加。
- 对比 `v6.4.1`：
  - `v6.4.1` 40GPU / bs1 / worker2 正式训练稳定后约 `14.2-15.0s/step`；
  - `v7-B` 2GPU smoke 因增加 Wan decoder + LPIPS reconstruction 分支，后续有效 step 约 `31-48s`，不能直接和 40GPU 训练硬比，但新增分支的开销已经明确。
- GC offload 扫描：
  - `use_gradient_checkpointing_offload=true`：约 120GB，后续有效 step 约 `31-48s`；
  - `use_gradient_checkpointing_offload=false`：约 140GB，后续有效 step 约 `31-45s`；
  - 速度收益不大但确实更快；速度优先时可用 `GC offload=false`，若正式多卡 OOM 或启动不稳再回退。
- 当前结论：
  - Stage3 `v7-B` 当前默认建议：`dataset_num_workers=2`、`stage3_decoder_cpu_offload=true`、`use_gradient_checkpointing_offload=true`；
  - 后续再拆分 `flow-only / one-step / MSE / LPIPS` 做显存分解，而不是继续盲目加 worker。

### 2026-05-14：定位 DataLoader worker 显存波动根因并完成 Stage3 显存分解

- 背景：
  - 用户怀疑 `dataset_num_workers` 开大后显存波动，说明数据侧仍有人使用 GPU；
  - 需要确认 worker 是否创建 CUDA context，并继续完成 `doc/flashvsr_stage3_memory_alignment_plan_20260514.md` 里的显存分解计划。
- 根因定位：
  - `diffsynth/diffusion/runner.py` 中 DataLoader 的 `worker_init_fn` 原本会调用 CUDA 相关逻辑；
  - `dataset_num_workers=8` 时，`nvidia-smi --query-compute-apps` 里能看到每个 worker 进程约 `614MB` CUDA context；
  - 这就是 worker 开大后显存随 worker 数上涨、甚至 OOM 的直接原因。
- 修复：
  - `runner.py` 新增 `_init_data_worker_no_cuda(...)`，只设置 CPU seed，不调用 CUDA；
  - `launch_training_task(...)` 和 `launch_data_process_task(...)` 两条 DataLoader 路径都改用 `_init_data_worker_no_cuda(...)`；
  - `parquet_tar_dataset_v2.py` 的旧退化 wrapper 改为强制 CPU degradation；
  - `aliyun_video_degradation.py` / `realesrgan_kernels.py` 的 CUDA seed 只在 degradation model device 真正为 CUDA 时执行。
- worker=8 修复验证：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_v7b_worker8_after_workerinit_fix`
  - 结果：跑到 `step=4`，没有 worker CUDA context OOM；
  - 修复后不再出现一串 614MB worker CUDA 进程；
  - 但 2GPU smoke 下速度没有稳定优于 `worker=2`，因此正式默认仍先用 `dataset_num_workers=2`。
- Stage3 显存分解新增开关：
  - `train_flashvsr_stage3_v7_b_lora.py` 新增 `--stage3_compute_z_pred / --no-stage3_compute_z_pred`；
  - 可分别测 flow-only、z_pred-only、MSE-only、MSE+LPIPS。
- 显存分解 probe：
  - Probe1 flow-only：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe1_flow_only`
    - 显存约 `40/43GB`。
  - Probe2 z_pred-only：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe2_zpred_only`
    - 显存约 `41/43GB`。
  - Probe3 MSE-only：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe3_mse_only`
    - 显存约 `119/121GB`。
  - Probe4 MSE+LPIPS：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_b_lora_89f_videoonly_fakefm_641data_20260514_probe4_mse_lpips_full`
    - 显存约 `119/121GB`。
- 结论：
  - worker 显存波动已经确认并修复，问题不是 CPU 退化本身，而是 worker init/旧 wrapper 触发 CUDA；
  - Stage3 当前 120GB 级峰值主要来自 Wan decoder selected-window backward；
  - one-step `z_pred` 本身不是显存主因；
  - LPIPS 逐帧串行后显存增量不明显，主要增加耗时和 loss；
  - 下一步如果要继续压显存，应优先优化 Wan decoder selected-window backward，而不是继续调 DataLoader worker。
- 底座核对：
  - `block_sparse_attn_func` 在远端 `flashvsr` 环境中可用；
  - Stage2/Stage3 attention 如果 CUDA extension 不可用会直接报错，不会静默走 dense fallback；
  - LPIPS 通过 `object.__setattr__` 挂在 pipe 缓存里，不注册为 DeepSpeed module；
  - 当前 trainable 参数量来自 LoRA + `lq_proj_in`，仍需后续决定是否降低 LoRA rank 或进一步检查 projector 参数量。

### 2026-05-14：收口 Stage3 `v7-B`，确认 641 帧规则与最快可用 worker/offload 组合

- 背景：
  - 用户观察到 `v6.4.1 / 641` 的测试结果较好，因此决定 Stage3 继续对齐 `641`，不再默认对测试集做额外 `85+repeat4` 预处理。
  - 同时需要明确 `block_sparse_attn` 是否真的被使用，以及 `worker/offload` 最快可用组合。
- `641` 89 帧规则核对：
  - 训练代码锚点：`wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
  - `num_frames=89` 时：
    - LQ projector 输出 `22` 个 latent chunks；
    - `target_frames = 22 * 4 - 3 = 85`；
    - GT 只取前 `85` 帧过 Wan VAE，得到 `22` 个 target latents；
    - loss 和 inference 都对齐这 `22` 个 latent / `85` 个有效输出帧。
  - 结论：
    - `641` 输入是正常 `89` 帧；
    - 但监督 / 输出是 `85` 有效帧；
    - 后续 `v7-B / v7-C` 先沿用该规则；
    - `85+repeat4` 只保留为 tail-padding ablation，不作为默认。
- `block_sparse_attn` 核对：
  - 远端裸 Python 直接 `import block_sparse_attn` 曾因 `libc10.so` 报错；
  - 激活 `flashvsr` 环境并先 `import torch` 后，`from block_sparse_attn import block_sparse_attn_func` 成功；
  - 训练路径里 `torch` 先被导入，`wan_video_dit_stage2_v6_1.py` 再导入 `block_sparse_attn_func`；
  - 如果 CUDA extension 不可用，代码会直接抛 `RuntimeError`，不会静默 fallback 到 dense attention；
  - 因此旧 `641` 能正常训练时，实际走的是 block sparse attention 路径。
- worker / offload 当前速度结论：
  - `worker=2 + GC offload on`：稳定，约 `120GB`，后续有效 step 约 `31-48s`；
  - `worker=2 + GC offload off`：稳定，约 `140GB`，后续有效 step 约 `31-45s`，当前最快已完成配置；
  - `worker=8 + GC offload on`：worker CUDA context 修复后能跑，但 2GPU smoke 未稳定快于 `worker=2`；
  - `worker=4/8 + GC offload off`：尝试后长时间停在 `0it`，不作为正式设置。
- 当前推荐：
  - 速度优先时，先用 `dataset_num_workers=2 + use_gradient_checkpointing_offload=false + stage3_decoder_cpu_offload=true`；
  - 如果正式多卡出现 OOM 或启动不稳定，回退到 `dataset_num_workers=2 + use_gradient_checkpointing_offload=true`。
- B 阶段收口：
  - CPU 在线退化、worker no-CUDA init、full-frame selected decode、首帧权重、显存分解、block sparse 路径均已完成验证；
  - 下一步进入 `v7-C`，需要新写 DMD dual-optimizer runner。

### 2026-05-14：补充 Stage3/C 前决策与退化 ablation 计划

- 用户确认：
  - `v6.4.1 / 641` 测试效果较好，后续 Stage3 先沿用 `641` 规则。
  - 正常取 `89` 帧输入，不默认外部构造 `85 + repeat4`。
  - 内部仍按 `89 -> 22 latent -> 85 有效监督/输出` 执行。
- Stage3 初始化：
  - 后续 Stage2 pretrain 固定使用 `641 step-6000` checkpoint；
  - 不再默认使用早期 `3k` checkpoint。
- `G_real` 决策：
  - 如果 FlashVSR 论文明确 `G_real` 来自 Stage1/full-attention teacher，则 `v7-C` 直接固定用 Stage1 teacher；
  - 不再继续在 Stage1/Stage2 teacher 之间摇摆。
- 显存优化边界：
  - Wan decoder selected-window backward 当前是约 `+40GB` 级显存增量；
  - 后续优化不能靠 tile，也不能改变 prefix no-grad / selected grad 语义；
  - 优先检查 frozen teacher/fake/decoder/LPIPS 是否误入 autograd graph，以及是否能串行释放。
- 退化 ablation 计划：
  - 用 5 个视频；
  - 固定同一份退化参数；
  - 分别 CPU/GPU apply；
  - 导出到桌面做肉眼对比；
  - 目的只验证 CPU 在线退化迁移是否改变视觉结果，不改变训练默认逻辑。
- 已执行退化 ablation：
  - 脚本：`wanvideo/data/flashvsr/tests/compare_degradation_cpu_gpu.py`
  - 远端目录：`/mnt/task_wrapper/user_output/artifacts/inference/degradation_cpu_gpu_ablation_20260514`
  - 本地目录：`/Users/lixiaohui/Desktop/degradation_cpu_gpu_ablation_20260514`
  - 每个样本包含：`gt.mp4`、`lq_cpu.mp4`、`lq_gpu.mp4`、`absdiff_x8.mp4`、`params.json`。
  - 数值结果：5 个视频的 mean abs diff 约 `0.076-0.197`。
  - 结论：CPU/GPU 固定参数退化差异明显，不能在未肉眼确认前直接认为 CPU 退化和 GPU 退化完全等价。

### 2026-05-14：改用训练入口 dump 验证 Stage3 CPU 退化真实输入

- 背景：
  - 用户反馈 `lq_gpu` ablation 结果几乎只有灰色轮廓，说明外部 CPU/GPU 退化对比本身可能没有复现训练路径；
  - 继续在外部脚本里复刻退化流程，仍不能保证和训练 DataLoader / collate / forward 前输入完全一致。
- 发现：
  - 旧 `compare_degradation_cpu_gpu.py` 默认走 `params_aliyun_video_compression_v1_half.yaml`，退化强度比轻量测试集大；
  - 旧脚本直接对 `320x192` 小视频继续做 `x4` LQ，会得到很小的 LQ，视觉上容易变成灰色轮廓；
  - 退化里仍有噪声和视频编码随机性，即使固定外层 params，CPU/GPU repeat 也不能严格 bitwise 对齐；
  - 因此外部 ablation 不能作为判断“训练送进模型前 LQ 是否烂掉”的最终证据。
- 代码改动：
  - 在 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_b_lora.py` 增加 rank0-only dump hook；
  - 新增参数：
    - `--debug_dump_training_batch_dir`
    - `--debug_dump_training_batch_limit`
    - `--debug_dump_training_batch_fps`
  - dump 位置在 Stage3 `forward()` 最开始、`get_pipeline_inputs()` 之前；
  - 保存的是 DataLoader 产出的 `video` 和 `lq_video`，也就是模型输入转换前的真实训练 batch。
- 验证方式：
  - 远端机器：`6ai5mpi47f`
  - 源视频：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_dump_source_videos_20260514/source18_1280x768.mp4`
  - manifest：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_dump_source_videos_20260514/source18_manifest.txt`
  - 说明：Stage2/Stage3 video dataset 当前不采样首帧，要求源视频帧数严格大于 `num_frames`；因此 17 帧测试源需要准备 18 帧视频。
- 训练内 dump 结果：
  - 远端目录：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7b_training_input_dump_cpu_degra_17f_20260514`
  - 本地目录：`/Users/lixiaohui/Desktop/stage3_v7b_training_input_dump_cpu_degra_17f_20260514`
  - 关键文件：
    - `before_model/sample_000/gt_before_model.mp4`
    - `before_model/sample_000/lq_before_model.mp4`
    - `before_model/sample_000/meta.json`
    - `before_model/batch_meta.json`
    - `run.log`
  - 实际训练代码跑通 `step=1`，loss 为 `0.562992`；
  - 该 dump 比外部退化 ablation 更可信，因为它直接来自当前训练代码中送入模型前的 batch。
- 后续判断：
  - 如果 `lq_before_model.mp4` 肉眼正常，说明 CPU 退化路径在训练入口前没有把 LQ 弄烂；
  - 如果 `lq_before_model.mp4` 已经异常，问题就在数据/退化/resize-crop，而不是 Stage3 模型本体；
  - 89 帧 full setting 可以复用同一 dump hook，但 CPU 退化会更慢，应只在需要时做一次性确认。

### 2026-05-14：启动 Stage3 `v7-C0` dual optimizer runner 骨架

- 背景：
  - `v7-B` 已经收口 one-step reconstruction、random latent decode、CPU 退化和显存分解；
  - 完整 DMD 需要 `G_real / G_fake` 和 dual optimizer，不能继续硬塞进现有单 optimizer runner。
- 新增代码：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c0_lora_89f_videoonly_dualopt_641data.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C0-Lora-89f-VideoOnly-DualOpt-641Data.sh`
- 当前范围：
  - C0 只验证独立 runner 能同时管理 student optimizer 和 fake optimizer；
  - `G_fake` 暂时是 `Stage3CFakeScalarModel` placeholder，不代表完整 DMD；
  - student 仍复用 `v7-B` one-step reconstruction path。
- 新文档：
  - `doc/flashvsr_stage3_v7c_dmd_runner_plan_20260514.md`
- 本地验证：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py` 通过。
- 远端 smoke：
  - 机器：`6ai5mpi47f`
  - 实验目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c0_lora_89f_videoonly_dualopt_641data_20260514_v7c0_smoke_r3`
  - Stage2 初始化：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
  - 已跑完 `max_train_steps=4`，看到 `step=1/2/3/4` loss，说明 student reconstruction path + fake optimizer skeleton 能进入训练；
  - 已保存 `step-1.safetensors`、`step-2.safetensors`、`step-4.safetensors`；
  - 已保存 `training_state/step-1/flashvsr_stage3c_extra.pt`、`step-2`、`step-4`，证明 fake model / optimizer / scheduler 的额外 state 路径可用。
- 注意：
  - 第一次 smoke OOM 是 GPU 0/1 残留旧 full-memory 占卡进程导致；
  - 清理占卡后，0/1 用于 smoke，2-7 低显存占卡，C0 正常出 loss；
  - C0 不是完整 DMD，只是 dual-optimizer runner 骨架验收，后续还要接真实 `G_real/G_fake`。

### 2026-05-14：Stage3 `v7-C2/C3` 接入 frozen `G_real/G_fake` probe

- 背景：
  - `v7-C0` 只证明 dual optimizer / fake extra state 路径可用；
  - 下一步需要逐步接入真实 `G_real/G_fake`，但不能直接打开完整 DMD 和 fake optimizer 更新。
- 代码改动：
  - 修改 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py`；
  - 新增 frozen probe 通用路径：
    - `--stage3_real_checkpoint`
    - `--stage3_real_attention_mode`
    - `--stage3_real_probe_every`
    - `--stage3_fake_probe_checkpoint`
    - `--stage3_fake_probe_attention_mode`
    - `--stage3_fake_probe_every`
  - `G_real/G_fake` probe 都设置为 `requires_grad=False`、`eval()`、`torch.no_grad()`；
  - probe loss 只记录日志 / wandb，不进 student loss，也不进 optimizer。
- C2 smoke：
  - 机器：`6ai5mpi47f`
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c2_lora_89f_videoonly_realprobe_641data_20260514_v7c2_smoke_r1`
  - Student：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
  - `G_real`：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
  - 已看到 student loss `1.146446` 和 real probe loss `0.136480`；
  - 已保存 `output/step-1.safetensors` 和 `training_state/step-1/flashvsr_stage3c_extra.pt`；
  - 结论：frozen `G_real` probe 可运行，但 89f dense full attention 成本很高，不适合后续每步全量无脑 probe。
- C3 smoke：
  - 机器：`6ai5mpi47f`
  - 新增文件：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c3_lora_89f_videoonly_fakeprobe_641data.yaml`
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C3-Lora-89f-VideoOnly-FakeProbe-641Data.sh`
  - 最终通过目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c3_lora_89f_videoonly_fakeprobe_641data_20260514_v7c3_smoke_r5_17f_512x256`
  - Student / `G_fake` probe 都加载 Stage2 `v6.4.1 step-6000`；
  - 已看到 student loss `1.030844` 和 fake probe loss `0.319069`；
  - 已保存 `output/step-1.safetensors` 和 `training_state/step-1/flashvsr_stage3c_extra.pt`。
- 注意：
  - C3 曾用 `320x192` smoke 失败，因为 latent grid `(4,12,20)` 不能被 block sparse 窗口 `(2,8,8)` 整除；
  - 改为 `512x256` 后通过；
  - C3 仍不是完整 DMD，只是 frozen `G_fake` no-grad forward/probe 验收；
  - 下一步进入 C4：开始计算 logging-only DMD direction，但仍不更新 `G_fake`。

### 2026-05-15：Stage3 `v7-C4/C5` DMD direction 与 FlashAttention 验证

- 代码改动：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py`。
- C4 新增 logging-only DMD direction probe：
  - `_stage3c_probe_predict_x0(...)`
  - `_maybe_run_stage3c_dmd_probe(...)`
  - `--stage3_dmd_probe_every`
- C5 新增 DMD2-style student loss：
  - `_maybe_run_stage3c_dmd_student_loss(...)`
  - `--stage3_dmd_weight`
  - 形式为 `0.5 * mse(z, (z - grad).detach())`，其中 `grad=((z-real_x0)-(z-fake_x0))/mean(abs(z-real_x0))`。
- 新增文件：
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c4_lora_89f_videoonly_dmdprobe_641data.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C4-Lora-89f-VideoOnly-DMDProbe-641Data.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C5-Lora-89f-VideoOnly-DMDLoss-641Data.sh`
- 远端验证机器：`6ai5mpi47f`。
- FlashAttention 验证：
  - debug 文件：`/mnt/task_wrapper/user_output/artifacts/debug/flashattn_v7c4_20260515_r2/flash_attention_branches.log`
  - 结果显示 `branch=flash_attn_2`，说明 `dense_full` 分支不是 torch dense fallback。
- 当前状态：
  - C4/C5 的 student、`G_real`、`G_fake` 单独前向均已跑通；
  - DMD x0 prediction 路径较慢，后续需要继续拆解 `G_real/G_fake` x0 调用成本；
  - 不改 `worker=0` 作为验收手段，数据路径不是当前主要问题。

补充 C5 smoke 结果：

- 通过目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c5_lora_89f_videoonly_dmdloss_641data_20260515_v7c5_smoke_r3_9f_256x256_nogather`
- 关键日志：`[stage3c_train] epoch=0 step=1 loss=1.703014 student=1.606964 fake_skeleton=0.00000100 fake_scale=0.000003 real_probe=0.304703 fake_probe=0.147766 dmd_student=0.096049 dmd_grad=1.029076`
- 已保存：`output/step-1.safetensors` 和 `output/training_state/step-1/flashvsr_stage3c_extra.pt`。
- 结论：C5 已经把 DMD2-style student loss 接入 student graph，并完成 backward / optimizer / checkpoint；`G_real/G_fake` 仍 frozen/no-grad，下一步才是把 `G_fake` 变成真实 trainable fake score model。
- 验证后已在 `6ai5mpi47f` 启动低显存占卡 `tmux lxh`，8 张卡 util 约 99%-100%。

补充 Stage3 总计划回看：

- 回读并更新 `doc/flashvsr_stage3_dmd_plan_20260511.md`。
- 新增第 27 节：`2026-05-15：v7-C 回看，防止偏离论文原始计划`。
- 核对结论：
  - `v7-C5` 已对齐 one-step student、frozen `G_real`、pixel/LPIPS decode、DMD student gradient 和 `dense_full` FlashAttention 路径；
  - `v7-C5` 仍未完成完整 DMD2，因为 `G_fake` 当前还是 frozen probe，不是真实 trainable fake score model；
  - 后续必须实现真实 `G_fake` copy、独立 optimizer/scheduler、fake FM update、alternating update ratio 和完整 save/resume。
- 以后汇报里应把当前 C5 称为 `DMD student-gradient path validation`，不能称为完整 Stage3。

### 2026-05-15：Stage3 `v7-C6` trainable `G_fake` smoke 通过

- 代码改动：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py`。
- 新增配置与脚本：
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh`
- 本轮修正：
  - `stage3_fake_checkpoint` 现在会创建完整 `FlashVSRStage3BTrainingModule` 作为 trainable `G_fake`；
  - 新增 `stage3_fake_attention_mode`，C6 smoke 使用 `dense_full`；
  - `G_fake` 使用独立 `fake_optimizer`；
  - `G_fake` 不进入 Deepspeed prepare，改为 fake backward 后手动 all-reduce gradient，保持各 rank 同步；
  - fake extra state 保存 trainable LoRA / `lq_proj_in` 参数和 fake optimizer / scheduler；
  - 修复 student forward 误接收 `stage3_fake_fm_weight` 的 guard 问题，fake FM 只在 dedicated runner 的 fake 分支里计算。
- 远端验证机器：`6ai5mpi47f`。
- 通过目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r2_9f_256x256`
- 关键日志：
  - `Stage3 v7-C6 trainable G_fake loaded trainable_params=570961408`
  - `[stage3c_train] epoch=0 step=1 loss=1.483968 student=1.389879 fake_loss=0.00162535 fake_scale=0.000000 real_probe=0.160513 fake_probe=0.438550 dmd_student=0.092463 dmd_grad=1.008987`
- 保存验证：
  - `output/step-1.safetensors`
  - `output/training_state/step-1/flashvsr_stage3c_extra.pt`
  - `flashvsr_stage3c_extra.pt` 中 `fake_model_is_full_stage3=True`，`fake_model` trainable keys 为 `488`。
- Resume 验证：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_resume_r2_9f_256x256`
  - 从 `...v7c6_smoke_r2.../output/training_state/step-1` 恢复；
  - 日志确认加载 `G_fake` trainable state：`loaded trainable G_fake state keys=488 missing=1019 unexpected=0`；
  - 已继续跑到 `step=2`：`loss=1.412708 student=1.309235 fake_loss=0.00766978 dmd_student=0.095803 dmd_grad=1.038058`。
- 结论：
  - C6 已经从 frozen fake probe 推进到 trainable `G_fake`；
  - fake FM loss、DMD student loss、fake optimizer state 都已通过最小 smoke；
  - 仍需后续验证 89f 正式尺寸、fake substep ratio 和长训 resume。
- 验证结束后，`6ai5mpi47f` 已重新启动低显存占卡 `tmux lxh`。

补充 C6 full-temporal smoke：

- `89f / 256x256` 通过：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r3_89f_256x256`
  - 日志：`loss=1.205961 student=1.106393 fake_loss=0.00660759 real_probe=0.147064 fake_probe=0.308552 dmd_student=0.092960 dmd_grad=0.999603`
  - recon window：`latent_window=[20,22)`，`frame_window=[77,85)`，`detached_context_latents=20`。
- `89f / 512x256` 通过：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r4_89f_512x256`
  - 日志：`loss=1.477752 student=1.151675 fake_loss=0.00033964 real_probe=0.178705 fake_probe=0.086649 dmd_student=0.325738 dmd_grad=1.902128`
  - recon window：`latent_window=[20,22)`，`frame_window=[77,85)`，`detached_context_latents=20`。
- 结论：
  - C6 不只在 9f toy smoke 可跑，已经验证过真实 Stage2 时间长度 `89f -> 22 latents`；
  - 当前下一步应转向多节点/正式尺寸短训，或者补 fake substep ratio，而不是继续只在 2GPU 小尺寸上堆验证。

补充正式尺寸 C6 验证：

- `89f / 1280x768 / no validation` 通过。
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r5_89f_1280x768_noval`
  - 日志：`[stage3c_train] epoch=0 step=1 loss=1.231092 student=1.192990 fake_loss=0.00091958 real_probe=0.016255 fake_probe=0.306179 dmd_student=0.037182 dmd_grad=0.671462`
  - 结论：正式训练尺寸下 forward / backward / fake update / DMD student loss / checkpoint 均可运行。
- 正式尺寸 validation 初次失败：
  - 目录：`..._20260515_v7c6_smoke_r6c_89f_1280x768_val` 和 `..._r8_89f_1280x768_val_auxoffload_barrier`
  - 现象：已保存 `step-1.safetensors`，且 validation 写出 `hr.mp4/lq.mp4`，但 `sr.mp4` 前 rank0 OOM。
  - 定位：validation callback 没有 `torch.inference_mode()`，导致保存点 validation forward 构建计算图；同时 C6 常驻 `G_real/G_fake`，进一步放大 rank0 显存。
- 已修复 validation：
  - 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_c_lora.py`
  - 改动：保存点手动 validation，validation 前临时 offload `G_real/G_fake` 到 CPU，validation 主体使用 `torch.inference_mode()`，validation 后全 rank barrier。
- `89f / 1280x768 / validation` 通过。
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_smoke_r9_89f_1280x768_val_infermode`
  - 输出：`output/step-1.safetensors`、`output/training_state/step-1/flashvsr_stage3c_extra.pt`、`output/validation/step-1/sample_000/{hr,lq,sr,meta}.mp4/json`
  - 结论：正式尺寸 2GPU validation smoke 通过，下一步进入 2GPU 短训和 resume 验证。

预备 48GPU 正式配置：

- 新增 config：`wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml`
- 新增 sh：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh`
- 本地检查：
  - YAML 可解析；
  - `bash -n` 通过。
- 关键设置：
  - `num_frames=89`，`height=768`，`width=1280`；
  - `takano_video_prob=0.5`，`yubari_video_prob=0.5`；
  - `dataset_num_workers=2`；
  - `stage3_fake_fm_weight=0.1`，`stage3_dmd_weight=0.1`；
  - validation 使用 one-step，`validation_num_samples=1`。
- 当前没有启动 48GPU，等待 2GPU 短训与 resume smoke 完成。

补充正式尺寸短训与 resume 验证：

- `89f / 1280x768 / no validation / max_train_steps=10` 通过。
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_short10_89f_1280x768_noval`
  - 关键 loss：
    - `step=1 loss=1.231092 student=1.192990 fake_loss=0.00091958 dmd_student=0.037182 dmd_grad=0.671462`
    - `step=5 loss=0.942800 student=0.877925 fake_loss=0.00111603 dmd_student=0.063760 dmd_grad=0.892547`
    - `step=10 loss=0.508211 student=0.470638 fake_loss=0.01109592 dmd_student=0.026477 dmd_grad=0.547276`
  - 已保存：`output/step-10.safetensors`、`output/training_state/step-10/flashvsr_stage3c_extra.pt`、Deepspeed optimizer / scheduler / random state。
  - 日志确认 random latent window 多次变化，并且 `decode_latents=[0,end)`、`detached_context_latents` 正常出现，符合 full-prefix detach 验证目标。
- `resume step10 -> step12` 通过。
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_v7c6_resume12_from_short10_89f_1280x768_noval`
  - resume 来源：`...v7c6_short10_89f_1280x768_noval/output/training_state/step-10`
  - 日志确认：
    - `[stage3c_resume] loaded trainable G_fake state keys=488 missing=1019 unexpected=0`
    - `[stage3c_resume] loaded extra fake state: .../flashvsr_stage3c_extra.pt`
    - `[stage3c_resume] loaded student/fake state step=10 epoch_id=0`
  - 关键 loss：
    - `step=11 loss=1.082293 student=1.009620 fake_loss=0.00114987 dmd_student=0.071523 dmd_grad=0.943789`
    - `step=12 loss=0.921560 student=0.900640 fake_loss=0.00789572 dmd_student=0.013024 dmd_grad=0.396666`
  - 结论：student Deepspeed state、trainable `G_fake` state、fake optimizer/scheduler extra state 都能恢复并继续训练。
- 当前结论：
  - `v7-C6` 已经完成 2GPU 全尺寸 first-loss、validation、short run、resume 四项验收；
  - 下一步可以进入 48GPU 正式训练启动，但启动后仍需确认 6 节点均进入训练循环并看到第一条 loss。

### 2026-05-15：Stage3 `v7-C6` 48GPU 正式训练启动

- 使用 6 节点母机：`t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk`。
- 主节点：`t5qdtykjsw`，`MASTER_ADDR=240.12.138.137`，`MASTER_PORT=29571`。
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data_20260515_043300_v7c6_48gpu`
- 启动文件：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-C6-Lora-89f-VideoOnly-TrainableFake-641Data.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_c6_lora_89f_videoonly_trainablefake_641data.yaml`
- 关键设置：
  - `num_frames=89`，`height=768`，`width=1280`，`batch_size=1`；
  - `takano_video_prob=0.5`，`yubari_video_prob=0.5`；
  - `dataset_num_workers=2`，`dataloader_prefetch_factor=1`；
  - `stage3_recon_num_latents=2`；
  - `stage3_fake_fm_weight=0.1`，`stage3_dmd_weight=0.1`；
  - `stage3_decoder_cpu_offload=true`；
  - `validation_num_samples=1`，one-step validation。
- 启动前检查：
  - 6 节点远端均已同步到包含 `_stage3c_run_validation_with_aux_offload(...)` 的代码；
  - 48GPU sh/yaml 在 6 节点均存在；
  - 启动前已停掉 `lxh/lxh_aux` 占卡。
- 当前运行状态：
  - 已出 `step=1` 和 `step=2` loss；
  - `step=1 loss=1.137422 student=1.073758 fake_loss=0.00815974 real_probe=0.193740 fake_probe=0.075488 dmd_student=0.055505 dmd_grad=0.798404`
  - `step=2 loss=1.373095 student=1.304030 fake_loss=0.00247871 dmd_student=0.066586 dmd_grad=0.853061`
  - 已保存 `output/step-1.safetensors`、`output/step-2.safetensors`；
  - 已保存 `training_state/step-1`、`training_state/step-2` 和 `flashvsr_stage3c_extra.pt`；
  - 已生成 `validation/step-1/sample_000/{hr,lq,sr,meta}` 和 `validation/step-2/sample_000/{hr,lq,sr,meta}`。
- 结论：
  - `v7-C6` 已经从 2GPU 验收进入 48GPU 正式训练；
  - 训练、checkpoint、trainable `G_fake` extra state、one-step validation 都已在 48GPU 路径上跑通；
  - 后续需要持续观察 step time、显存峰值、validation 输出质量和是否存在长训内存爬升。

### 2026-05-15：Stage3 `v7-D` 48GPU 离线 W&B 正式训练

- 背景：
  - 复跑 `v7-C6` 48GPU online W&B 后，仍出现 `wandb: Network error (ConnectTimeout)`。
  - 结论：当前问题不是 `v7-D` validation 分支导致，而是 48GPU 环境里 W&B online 初始化/联网不稳定。
- 改动：
  - 新增离线配置：`wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb.yaml`
  - 新增离线启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-641Data-OfflineWandb.sh`
  - 修改基础 Stage3 v7-D 启动脚本，后续新实验默认设置 `WANDB_DIR=${RUN_DIR}`，让 W&B 离线文件直接进入 artifacts 实验目录。
  - 新增 W&B 离线同步脚本：`wanvideo/model_training/flashvsr/scripts/sync_wandb_offline_loop.sh`
    - 当前已启动的 run 因 W&B 进程已写到 `/mnt/task_runtime/lucidvsr/wandb`，同步脚本会先镜像到 `${RUN_DIR}/wandb`，再执行 `wandb sync`。
    - 后续新 run 会直接写入 `${RUN_DIR}/wandb`。
- 当前实验：
  - 目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb_20260515_v7d_48gpu_offlinewandb_fixed`
  - rank0 机器：`t5qdtykjsw`
  - W&B 本地持久目录：`.../wandb/offline-run-20260515_011822-6nu0uhd7`
  - W&B 同步 tmux：`wandb_sync_v7d`
- 运行状态：
  - 已进入训练并出 loss：
    - `step=1 loss=5.080862`
    - `step=2 loss=1.219400`
    - `step=3 loss=1.442616`
    - `step=4 loss=1.069691`
    - `step=5 loss=1.588794`
  - validation-from-batch 正常缓存样本，未再卡在独立 validation 取样。
- 结论：
  - `v7-D` 已经用 offline W&B 跑通 48GPU 正式训练；
  - 当前 W&B 文件已镜像到 artifacts，避免只留在 `/mnt/task_runtime/lucidvsr/wandb` 导致机器回收后丢失；
  - 后续若 online 同步仍受网络影响，不影响训练和本地 W&B 文件保存。

### 2026-05-15：Stage3 `v7-D stable snapshot` 48GPU 正式训练与 10 视频推理补测

- 目的：
  - 回退到已验证稳定的 `v7-C6` 代码快照，再只叠加 `v7-D` 必需差异，避免前一轮 W&B / validation debug 引入不稳定残留。
- 新增 / 使用代码：
  - 训练代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d_stable.py`
  - 配置：`wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_stable_authorweights_offline.yaml`
  - 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh`
  - 批量推理入口：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py`
- `v7-D stable` 相对 `v7-C6` 的差异：
  - `stage3_fake_fm_weight=1.0`
  - `stage3_dmd_weight=1.0`
  - `stage3_first_frame_pixel_weight=4.0`
  - `stage3_first_frame_lpips_weight=4.0`
  - DMD spike guard：`stage3_dmd_grad_norm_max=5.0`，`stage3_dmd_spike_policy=skip`
  - W&B 固定 offline，且 `WANDB_DIR=${RUN_DIR}`，离线文件直接写入 artifacts 实验目录。
- 48GPU 正式实验：
  - 6 节点母机：`t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk`
  - 主节点：`t5qdtykjsw`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_authorweights_offline_final`
  - W&B 离线目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_authorweights_offline_final/wandb/offline-run-20260515_023433-0ziqlukz`
  - W&B 后台同步 tmux：
    - `wandb_sync_v7d_stable`
- 训练状态：
  - 已确认正式出 loss，不再是 0 利用率卡死：
    - `step=1 loss=1.710403 student=1.073758 fake_loss=0.08159743 dmd_student=0.555047 dmd_grad=0.798404 dmd_skip=0`
    - `step=4 loss=1.345553 student=0.856090 fake_loss=0.04022082 dmd_student=0.449243 dmd_grad=0.710419 dmd_skip=0`
    - `step=13 loss=2.641670 student=0.999295 fake_loss=0.24517365 dmd_student=1.397202 dmd_grad=1.239299 dmd_skip=0`
  - 观察到 `stage3_v7_b_loss` 日志里非首帧 window 的 `effective_first_frame_* = 1.0` 是正常现象；只有随机窗口覆盖全局首 latent 时才应为 `4.0`。
- smoke / failed 目录整理：
  - smoke 集中目录：`/mnt/task_wrapper/user_output/artifacts/exp/_smoke_20260515`
  - 失败 debug 集中目录：`/mnt/task_wrapper/user_output/artifacts/exp/_failed_v7d_wandb_debug_20260515`
- 10 视频合成测试补测：
  - 测试机器：`8nh48ucn8b`
  - 测试集：
    - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - 测试 checkpoint：
    - `v7-C6 step-100`
    - `v7-C6 step-200`
  - 测试脚本语义：
    - `infer_flashvsr_stage3_v7_d_batch`
    - `num_inference_steps=1`
    - `tiled=false`
    - `stage2_attention_mode=block_sparse_chunk_causal`
    - `stage2_topk_ratio=2.0`
    - `stage2_local_num=-1`
    - `stage2_kv_ratio=3.0`
    - `input_bicubic_upscale=4.0`
    - `color_fix_method=adain`
  - 远端结果目录：
    - `/mnt/task_wrapper/user_output/artifacts/inference/v7d_c6_step100_200_10synthetic_notile_20260515`
  - 本地桌面结果目录：
    - `/Users/lixiaohui/Desktop/v7d_c6_step100_200_10synthetic_notile_20260515`
  - 结果：
    - `step-100`：10 个视频，`162s`，约 `16.2s/video`
    - `step-200`：10 个视频，`162s`，约 `16.2s/video`
  - 测试结束后，`8nh48ucn8b` 已恢复 8 卡占卡。

### 2026-05-15：Stage3 `v7-D1` 分支准备

- 目的：
  - 在 `v7-D stable` 已经跑通的基础上单独复制出 `v7-D1`，不直接修改正在跑的 `v7-D stable`。
  - 内部 validation 固定为 2 个视频，并且 validation 必须走 Stage2/Stage3 最终一致的 one-step streaming KV-cache 推理路径。
  - 修正日志与 metadata 中仍残留的 `v7-B` / `v7-C` 命名，避免后续看 log 混淆。
- 新增代码：
  - 训练代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d1_lora.py`
  - 48GPU 配置：`wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d1_lora_89f_videoonly_authorweights_trainablefake_641data_offlinewandb.yaml`
  - 48GPU 启动脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D1-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-641Data-OfflineWandb.sh`
  - 2GPU smoke 配置：`wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d1_lora_89f_videoonly_authorweights_val2_notile.yaml`
  - 2GPU smoke 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D1-Lora-89f-VideoOnly-AuthorWeights-Val2-NoTile.sh`
  - 扫描测试脚本预备：`wanvideo/model_inference/flashvsr/history/run_stage3_v7_d1_scan89_step500_incremental.sh`
- 关键设定：
  - `validation_num_samples=2`
  - `stage3_validation_from_training_batch=true`
  - `stage3_validation_tiled=false`
  - validation mode：`stage3_v7_d1_streaming_kvcache_one_step`
  - DMD spike guard 保持开启：`stage3_dmd_grad_norm_max=5.0`，`stage3_dmd_spike_policy=skip`
  - W&B 继续 offline，并且 `WANDB_DIR=${RUN_DIR}`，离线文件写入 artifacts 实验目录。
- validation 代价说明：
  - validation 由 rank0 执行，其他 rank 会在 checkpoint 边界等待；
  - 因此内部 val 保持 2 个视频，只用于训练健康检查；
  - 10-video benchmark 仍通过外部扫描脚本单独跑，不塞进训练内 validation。

### 2026-05-15：放弃 `v7-D1`，回退到 `v7-D stable`

- 背景：
  - `v7-D1` 的 2GPU smoke 已确认能进入训练循环，并且 step1 出 loss。
  - `v7-D1` 的 validation gating 生效：step1 只缓存 `1/2` 个样本，不会过早触发 validation。
  - 但 48GPU 运行时 step2 计算异常重，rank0 显存维持在约 `156-157GB`，单步耗时明显过长。
- 结论：
  - `v7-D1` 新增的 2-video streaming validation / batch cache / 额外逻辑不再继续作为当前主线。
  - 当前回退到上一轮已跑通的 `v7-D stable`，优先保证 Stage3 正式训练能稳定出 loss。
- 当前正式回退实验：
  - 6 节点母机：`t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk`
  - 主节点：`t5qdtykjsw`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return`
  - 训练代码：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d_stable.py`
  - 配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d_stable_authorweights_offline.yaml`
  - 启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh`
- W&B：
  - 使用 offline 模式。
  - `WANDB_DIR=${RUN_DIR}`，离线文件写入实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return/wandb/`
  - 后台同步 tmux：
    - `wandb_sync_v7d_stable`
  - 同步脚本：
    - `wanvideo/model_training/flashvsr/scripts/sync_wandb_offline_loop.sh`
  - 说明：`wandb sync` 网络超时时会在 timeout 窗口内重试；训练本身不依赖在线同步。
- 已确认状态：
  - 48 个 rank 已连接，NCCL/TCPStore 不再是当前卡点。
  - 第一条 loss 已出现：
    - `[stage3_v7_b_loss] loss=1.073758 flow=0.019114 mse=0.020162 lpips=0.517241 ...`
  - 后续又出现纯 flow step：
    - `loss=0.193740 flow=0.193740`
    - `loss=0.075488 flow=0.075488`
  - GPU 利用率进入正常计算区间，rank0 节点观测多数卡约 `97-100%`。
  - rank0 节点显存约 `67-87GB`，明显低于 `v7-D1` 的 `150GB+` 峰值。
- 备注：
  - 当前 Codex 侧频繁出现的 `maximum number of unified exec processes` 是本地工具进程上限提示，不是远端训练问题。
  - 后续远端检查尽量使用短命令抓 tmux/log，不再开长时间 `exec` 等待。

### 2026-05-15：`v7-D stable` W&B 离线同步改成 t5 打包、6a 上传

- 背景：
  - 48 卡训练节点能正常写 W&B offline 文件，但 6 个训练节点访问 `https://api.wandb.ai` 会超时；
  - `6ai5mpi47f` 可以访问 W&B 并成功执行 `wandb sync`；
  - 因此不再让训练节点直接同步 W&B，而是改成“训练主节点打包到 S3，6a 拉取后上传”。
- 新增脚本：
  - t5/训练主节点打包上传：
    - `wanvideo/model_training/flashvsr/scripts/package_wandb_offline_to_s3_loop.sh`
  - 6a 拉取并同步：
    - `wanvideo/model_training/flashvsr/scripts/sync_wandb_offline_from_s3_loop.sh`
- 启动脚本改动：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D-StableSnapshot-AuthorWeights-OfflineWandb.sh`
  - 从旧的 `wandb_sync_v7d_stable` 直连同步，改成 `wandb_package_v7d_stable` 打包上传。
- 当前正在跑的 `v7-D stable` 已手工挂上：
  - t5 tmux：`wandb_package_v7d_stable`
  - 6a tmux：`wandb_sync_from_s3_v7d_stable`
  - S3 中转包：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return.tar.gz`
- 验证结果：
  - t5 已成功上传离线包到 S3；
  - 6a 已成功下载并同步两个 offline run：
    - `offline-run-20260515_011822-6nu0uhd7`
    - `offline-run-20260515_041453-04kunrao`
- 注意：
  - 新脚本显式补了远端 `PATH`、`NOTARY_CONFIG_FILE` 和 `AWS_CA_BUNDLE`，避免从 tmux 启动时丢环境导致 `conductor` / `aws` / Notary 失败。
  - 后续如果 W&B 在线网络不通，优先复用这套 relay，不再反复修训练节点直连 W&B。

### 2026-05-15：新增 `v7-D2` 独立代码线，修 DMD shared noisy point 与 fake ratio 语义

- 背景：
  - `doc/flashvsr_stage3_v7d_dmd_review_20260515.md` 中确认当前 `v7-D stable` 的 DMD student loss 存在语义问题：
    - `G_real` 和 `G_fake` 分别调用 `_stage3c_probe_predict_x0()`；
    - 该函数内部各自采样 `timestep/noise/noisy_latents`；
    - 因此 real/fake score difference 不是在同一个 noisy point 上比较。
  - 同时，旧参数 `stage3_fake_update_ratio` 容易被误解成 DMD2 的 `dfake_gen_update_ratio`。当前实现里它实际表示“每 N 个 student step 更新一次 fake”，并不是“每个 generator update 前 fake 多更新 N 次”。
- 处理原则：
  - 不动 `v7-D1`；
  - 不热改正在跑的 `v7-D stable return`；
  - 从 `v7-D stable` 单独复制一条 `v7-D2` 线，只修 DMD shared-noise 和 fake ratio 命名语义；
  - validation 仍保持当前轻量 one-step full decode，不改流式 validation。
- 新增文件：
  - 训练代码：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py`
  - 48GPU 配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb.yaml`
  - 48GPU 启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh`
- 代码改动：
  - `_stage3c_probe_predict_x0(...)` 新增 `dmd_point` / `return_dmd_point`：
    - 第一次 real probe 采样 `timestep/noise/noisy_latents`；
    - 返回 `dmd_point`；
    - fake probe 复用同一个 `dmd_point`；
    - 保证 DMD probe 和 DMD student loss 中 `G_real/G_fake` 在同一个 noisy latent / timestep 上比较。
  - `_maybe_run_stage3c_dmd_probe(...)` 和 `_maybe_run_stage3c_dmd_student_loss(...)` 都改成 shared `dmd_point`。
  - 新增 `_stage3c_fake_update_every_n_steps(args)`：
    - 新参数 `stage3_fake_update_every_n_steps` 明确表示“每 N 个 student step 更新一次 G_fake”；
    - 旧参数 `stage3_fake_update_ratio` 只作为 legacy alias 保留；
    - 日志中同时打印 `fake_update_every_n_steps` 和 `legacy_fake_update_ratio`，避免误以为 ratio=5 是 DMD2 fake 多步更新。
  - 当 `G_fake` 因 `stage3_fake_update_every_n_steps` 被跳过时，fake optimizer / fake scheduler 也不再 step，并在训练日志中输出 `fake_update=0/1`。
- 配置改动：
  - `stage3_fake_update_every_n_steps: 1`
  - `stage3_fake_update_ratio: 1` 保留为 legacy alias；
  - W&B 名称改为 `flashvsr-stage3-48gpu-v7-d2-sharednoise-authorweights-offline`。
- 启动脚本改动：
  - `TRAIN_PY` 指向 `train_flashvsr_stage3_v7_d2_lora.py`；
  - `CONFIG_PATH` 指向 v7-D2 shared-noise config；
  - `OUTPUT_TAG` 改为 `train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb`；
  - W&B offline package tmux 改为 `wandb_package_v7d2_sharednoise`；
  - `TRAIN_PROCESS_PATTERN` 改为 `train_flashvsr_stage3_v7_d2_lora.py`。
- 本地检查：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py` 通过；
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh` 通过；
  - 本机缺 `yaml` 模块，未做本地 YAML load 检查。
- 当前远端状态：
  - 本地 `v7d48` watch/ssh pane 曾因 `Broken pipe` 掉线；
  - 已重新连接 `t5qdtykjsw` 并恢复 30 秒刷新 watch；
  - watch 显示主节点 GPU 仍在高利用率，未重启/停止当前 `v7-D stable return`。

### 2026-05-15：停止 `v7-D stable return`，启动 `v7-D2 shared-noise` 48GPU

- 先复核 `doc/flashvsr_stage3_v7d_dmd_review_20260515.md` 中同事 0 的新增回复：
  - 认可 shared-noise DMD 改法；
  - 认可 `fake_update_every_n_steps` / legacy ratio 澄清；
  - 提醒 D2 启动脚本 PATH 需要包含 `/root/.local/share/pipx/venvs/awscli/bin` 和 `/usr/local/bin`；
  - 提醒 6a W&B relay 需要单独挂；
  - Stage1 teacher wrapper 等价性仍是后续单独验证项，不混入本轮 D2。
- 采纳的补丁：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh`
  - PATH 从：
    - `/mnt/conda_envs/flashvsr/bin:/root/.local/bin:/miniforge/bin:$PATH`
  - 改为：
    - `/mnt/conda_envs/flashvsr/bin:/root/.local/share/pipx/venvs/awscli/bin:/root/.local/bin:/usr/local/bin:/miniforge/bin:$PATH`
- 本地静态检查：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py` 通过；
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh` 通过。
- 远端同步/静态检查：
  - 主节点：`t5qdtykjsw`
  - 远端路径 `/mnt/task_runtime/lucidvsr` 已看到 D2 三个文件；
  - 远端 `/mnt/conda_envs/flashvsr/bin/python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py` 通过；
  - 远端 `bash -n ...D2...sh` 通过；
  - 主节点 IP：`240.12.138.137`。
- 已停止旧 run：
  - 旧实验：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d_stable_authorweights_offline_20260515_v7d_stable_return`
  - 在 6 节点上停止：
    - `train_flashvsr_stage3_v7_d_stable.py`
    - 对应 accelerate / multiprocessing 子进程；
    - 远端 tmux `v7d_stable_48gpu`；
    - 主节点旧 W&B package tmux `wandb_package_v7d_stable`。
  - 6 节点均确认 `AFTER_STOP_PIDS=0`。
- 新 run：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb_20260515_v7d2_sharednoise_48gpu`
  - 训练代码：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py`
  - 配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb.yaml`
  - 启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D2-Lora-89f-VideoOnly-AuthorWeights-TrainableFake-SharedNoise-OfflineWandb.sh`
  - 6 节点 tmux：
    - `v7d2_48gpu`
  - 启动参数：
    - `MASTER_ADDR=240.12.138.137`
    - `MASTER_PORT=29521`
    - `RUN_TS_OVERRIDE=20260515_v7d2_sharednoise_48gpu`
- W&B relay：
  - 主节点 D2 package tmux：
    - `wandb_package_v7d2_sharednoise`
  - S3 中转包：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb_20260515_v7d2_sharednoise_48gpu.tar.gz`
  - 6a 同步 tmux：
    - `wandb_sync_from_s3_v7d2_sharednoise`
  - 手动强制打包并同步一次后，6a 已成功 sync D2 offline run：
    - `offline-run-20260515_060357-7hp1ycz4`
- 当前训练状态：
  - 已看到 step 1/2 checkpoint：
    - `output/training_state/step-1`
    - `output/training_state/step-2`
  - 已看到 validation：
    - `output/validation/step-1`
    - `output/validation/step-2`
  - 已看到训练日志：
    - `step=1 loss=1.191537 student=1.073758 fake_loss=0.11777902 fake_update=1 real_probe=0.193740 fake_probe=0.075488 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
    - `step=2 loss=1.369715 student=1.308358 fake_loss=0.05788712 fake_update=1 dmd_student=0.003470 dmd_grad=0.062931 dmd_skip=0`
    - `step=3 loss=0.693872 student=0.652548 fake_loss=0.03312649 fake_update=1 dmd_student=0.008198 dmd_grad=0.095331 dmd_skip=0`
  - 说明：
    - step 1 的 `dmd_student=0` 符合 real/fake 初始同权重时 DMD 差分接近 0 的预期；
    - step 2/3 开始出现非零 DMD student loss，说明 shared-noise DMD path 已进入训练；
    - 当前未见 Traceback / OOM / CUDA error。
- watch：
  - 本地 `v7d48` 已切换为 D2 监控，30 秒刷新主节点 loss / GPU。

### 2026-05-15：补做 Stage1 teacher / Stage3 wrapper projector 等价性验证

- 背景：
  - `doc/flashvsr_stage3_v7d_dmd_review_20260515.md` 中仍未关闭的风险是：`G_real/G_fake` 虽然加载 Stage1 `v5.3.5` checkpoint 并设置 `dense_full` attention，但它们实际由 Stage3B/Stage2 wrapper 构造，可能不是 Stage1 v5.3.5 full-attention teacher 的严格 forward 语义。
  - 主要怀疑点集中在 LQ projector temporal mode：Stage1 v5.3.5 稳定母本使用 `nonstreaming_aligned` / `aligned23`，而当前 Stage2/Stage3 wrapper 在 `FlashVSRStage2Pipeline.from_pretrained()` 中把 `lq_proj_in` 的 `temporal_mode` 硬编码为 `streaming`。
- 新增验证脚本：
  - `wanvideo/model_training/flashvsr/tests/check_stage1_stage3_teacher_projector_equivalence.py`
  - 只加载 Stage1 checkpoint 中的 `lq_proj_in` 权重，不加载完整 DiT；
  - 用同一份权重、同一个 deterministic 输入，分别跑：
    - Stage1 预期：`nonstreaming_aligned`
    - Stage3 wrapper 当前行为：`streaming`
    - 参考旧模式：`nonstreaming`
  - 如果 projector 输出 shape 已不同，则完整 teacher forward 不可能严格等价。
- 6a 资源状态：
  - 远端机器：`6ai5mpi47f`
  - GPU0/GPU1 用于验证；
  - GPU2-7 已继续用 `lxh_occupy_2_7` 占卡。
- 远端日志：
  - `/mnt/task_wrapper/user_output/artifacts/debug/teacher_equiv_20260515/projector_equiv_89f_256x256.log`
  - `/mnt/task_wrapper/user_output/artifacts/debug/teacher_equiv_20260515/projector_equiv_17f_256x256.log`
- 使用 checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- 89f 结果：
  - checkpoint keys：`lq_proj=8`，`lora=480`
  - `nonstreaming_aligned` 输出：`(1, 5888, 1536)`
  - `streaming` 输出：`(1, 5632, 1536)`
  - `nonstreaming` 输出：`(1, 5632, 1536)`
  - 结论：`RESULT=FAIL_NOT_EQUIVALENT reason=shape_mismatch`
- 17f 结果：
  - `nonstreaming_aligned` 输出：`(1, 1280, 1536)`
  - `streaming` 输出：`(1, 1024, 1536)`
  - `nonstreaming` 输出：`(1, 1024, 1536)`
  - 结论：`RESULT=FAIL_NOT_EQUIVALENT reason=shape_mismatch`
- 当前判断：
  - 这已经不是完整 forward 的数值容差问题，而是在进入 DiT 之前 LQ conditioning token 数量不同。
  - 因此当前 v7-D2 的 `G_real/G_fake` 更准确地说是“Stage1 权重 + Stage3/Stage2 streaming wrapper”，不是严格的 Stage1 v5.3.5 `nonstreaming_aligned` full-attention teacher/copy。
  - 后续若要继续对齐 FlashVSR 论文里的 `G_real/G_fake = Stage1 full-attention teacher/copy`，应优先修改 Stage3B teacher/fake wrapper，使 `G_real/G_fake` 能使用 Stage1 v5.3.5 的 projector temporal mode；这需要单独复制新版本，不应热改正在跑的 v7-D2。

### 2026-05-15：v7-D2 强制首帧 window 权重验证

- 目标：
  - 验证当 Stage3 random latent decode 抽到全局首帧时，`pixel MSE` 和 `LPIPS` 的首帧权重是否都按结论乘以 `4.0`。
  - 该验证独立于长期训练，只跑一次 2GPU smoke。
- 本地代码改动：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d2_lora.py`
    - 新增仅调试用环境变量 `FLASHVSR_STAGE3_FORCE_RECON_START`。
    - 默认不设置时不改变随机采样逻辑。
    - 设置为 `0` 时强制 `_sample_stage3_recon_window()` 返回以首 latent 开头的 window。
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d2_firstframe_weightcheck.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D2-FirstFrameWeightCheck.sh`
- 远端机器：
  - `8nh48ucn8b`
- 远端实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d2_firstframe_weightcheck_20260515_firstframe_weightcheck_clean2`
- 运行设置：
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `FLASHVSR_STAGE3_FORCE_RECON_START=0`
  - `FLASHVSR_STAGE3_DEBUG_LOSS=1`
  - `stage3_recon_num_latents=2`
  - `stage3_first_frame_pixel_weight=4.0`
  - `stage3_first_frame_lpips_weight=4.0`
- 过程中遇到的问题：
  - 第一次 smoke OOM，不是代码逻辑问题，而是 8n 上占卡脚本没有遵守外层 `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7`，仍占用了 GPU0/1。
  - 清理所有 GPU 进程后，重新只跑 0/1 两卡验证成功。
- 关键日志：
  - `stage3_v7_d2_firstframe_weightcheck force_recon_start=0`
  - `expect latent_window=[0,2) first_frame_pixel_weight=4.0 first_frame_lpips_weight=4.0`
  - `[stage3_v7_b_loss] loss=0.466705 flow=0.107180 mse=0.007805 lpips=0.175860 latent_window=[0,2) frame_window=[0,5) decode_latents=[0,2) local_frame_window=[0,5) recon_latents=2 decoded_frames=5 context_mode=full_prefix decoder_cpu_offload=True detached_context_latents=0 first_frame_pixel_weight=4.0 first_frame_lpips_weight=4.0 compute_z_pred=True need_reconstruction=True`
- 结论：
  - 抽到首帧时，当前 v7-D2 loss 分支确实对 `MSE` 和 `LPIPS` 都启用了首帧 `4.0` 权重。
  - `latent_window=[0,2)` 对应 `frame_window=[0,5)`，符合 WAN 首 latent 单帧、后续 latent 四帧的语义。
  - 这个验证只证明首帧权重进入 loss，不证明 `G_real/G_fake` teacher wrapper 已严格等价 Stage1；teacher 等价性仍按单独验证推进。
- 资源恢复：
  - 验证结束后，8n 已重新启动 `lxh` 占卡程序。

### 2026-05-15：`v7-D3` Stage1 teacher wrapper 对齐与 6a smoke 验证

- 背景：
  - 同事 0 在 `doc/flashvsr_stage3_v7d_dmd_review_20260515.md` 中明确认为 projector 等价性问题必须修；
  - `v7-D2` 的 `G_real/G_fake` 是 Stage1 checkpoint weights + Stage2/3 streaming wrapper，不是严格 Stage1 v5.3.5 `nonstreaming_aligned` full-attention teacher/copy；
  - 因此复制新线 `v7-D3`，不热改正在跑的 `v7-D2`。
- 新增 / 修改文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
    - `FlashVSRStage2Pipeline.from_pretrained()` / `FlashVSRStage2TrainingModule` 新增 `lq_proj_temporal_mode` 参数，默认仍为 `streaming`，保持旧线兼容；
    - `flashvsr_stage2_model_fn()` 新增 `lq_latent_alignment`，支持 `trim_front_to_match`，用于 teacher 23 positions 对齐 student 22 positions；
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1` 时打印 `[stage3_teacher_align]`。
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py`
    - 从 D2 复制；
    - `G_real/G_fake` 构造新增 `stage3_real_lq_proj_temporal_mode` / `stage3_fake_lq_proj_temporal_mode`；
    - D3 config 默认二者均为 `nonstreaming_aligned`；
    - DMD probe、DMD student loss、generic probe forward 均根据 teacher wrapper 自动设置 `lq_latent_alignment=trim_front_to_match`；
    - student 仍保持 Stage3 streaming one-step 语义。
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_lora_89f_videoonly_authorweights_stage1teacher_aligned_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_teacher_aligned.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-OfflineWandb.sh`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-TeacherAligned.sh`
- FlashAttention 注意：
  - D3 仍设置 teacher `attention_mode=dense_full`；
  - dense full attention 路径仍走 Stage2 patched attention 中的 `flash_attention(...)`，没有退回手写 dense attention。
- 本地静态检查：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py` 通过；
  - `bash -n` 检查 D3 release / smoke 两个脚本通过。
- 6a 远端静态检查：
  - 远端机器：`6ai5mpi47f`
  - 远端目录：`/mnt/task_runtime/lucidvsr`
  - `/mnt/conda_envs/flashvsr/bin/python -m py_compile ...stage2_v6_4... ...stage3_v7_d3...` 通过；
  - D3 smoke 脚本 `bash -n` 通过。
- 第一次 D3 smoke：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_smoke1`
  - 结果：失败，但定位有效；
  - 报错：
    - `Stage2 requires exact LQ/DiT token match ... x=84480, lq=88320 ... alignment=exact`
  - 原因：
    - `real_probe_every` 在 step0 触发 generic `probe_model.forward()`，该路径还没有带 D3 teacher alignment；
  - 处理：
    - 在 `FlashVSRStage3BTrainingModule.forward()` 中补入 teacher wrapper alignment。
- 主 D3 smoke：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_smoke2`
  - 结果：通过 step 1 并正常退出；
  - 关键日志：
    - `G_real ... attention_mode=dense_full lq_proj_temporal_mode=nonstreaming_aligned`
    - `G_fake ... attention_mode=dense_full lq_proj_temporal_mode=nonstreaming_aligned`
    - `[stage3_teacher_align] mode=trim_front_to_match grid=(22, 48, 80) expected_tokens=84480 aligned_lq_shape=(1, 84480, 1536)`
    - `real` / `fake` DMD probe 均使用 `timestep=368.000000`，`shared_point=True`；
    - `noise_pred=(1, 16, 22, 96, 160)`，`x0=(1, 16, 22, 96, 160)`；
    - `[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 fake_scale=0.000000 real_probe=0.107180 fake_probe=0.059744 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
  - 结论：
    - `G_real/G_fake` 已切到 Stage1 `nonstreaming_aligned` projector 语义；
    - teacher 23-position LQ conditioning 已显式裁到 student 22 positions；
    - 初始 `G_fake == G_real` 时 DMD zero-diff sanity 成立；
    - fake optimizer 路径实际执行，`fake_update=1` 且 `fake_loss` 非零。
- 中间 window smoke：
  - 设置：
    - `FLASHVSR_STAGE3_FORCE_RECON_START=10`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_mid10_smoke`
  - 结果：通过 step 1；
  - 关键日志：
    - `latent_window=[10,12) frame_window=[37,45) decode_latents=[0,12) local_frame_window=[0,8) detached_context_latents=10 first_frame_pixel_weight=1.0 first_frame_lpips_weight=1.0`
    - `[stage3c_train] ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
  - 结论：
    - 中间 window frame mapping 正常；
    - 非首帧 window 没有误用 4 倍首帧权重。
- 尾部 window smoke：
  - 设置：
    - `FLASHVSR_STAGE3_FORCE_RECON_START=20`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_teacher_aligned_tail20_smoke`
  - 结果：通过 step 1；
  - 关键日志：
    - `latent_window=[20,22) frame_window=[77,85) decode_latents=[0,22) local_frame_window=[0,8) detached_context_latents=20 first_frame_pixel_weight=1.0 first_frame_lpips_weight=1.0`
    - `[stage3c_train] ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
  - 结论：
    - 尾部 window 不越界；
    - 非首帧 window 权重正常。
- 当前 6a 资源：
  - smoke 结束后 GPU0/GPU1 已释放；
  - GPU2-7 继续由 `lxh_occupy_2_7` 占卡；
  - `wandb_sync_from_s3_v7d2_sharednoise` 和 `wandb_sync_from_s3_v7d_stable` 仍在运行。
- 当前判断：
  - `v7-D3` 已关闭 `v7-D2` 的 teacher projector temporal-mode 主问题；
  - 仍建议在正式 48GPU 前保留 D2 作为 ablation，不覆盖其结果；
  - D3 已具备上 48GPU 前的 smoke 证据，但尚未启动 48GPU 长训。

### 2026-05-15：`v7-D3` 按同事二次审查补 guard / temporal map / optimizer ownership

- 背景：
  - 同事 0 对第 18 节 D3 实现做了二次审查；
  - 认可主方向，但指出还应补：
    - `stage3_fake_checkpoint` guard；
    - `trim_front_to_match` 的 temporal map 日志；
    - optimizer ownership 参数变化级检查；
    - D3 snapshot 完整性；
    - validation meta 旧命名。
- 代码补丁：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py`
    - parse 阶段新增 guard：
      - 当 `stage3_fake_fm_weight > 0` 或 `stage3_dmd_weight > 0` 时，必须提供 `stage3_fake_checkpoint`；
      - 防止只用 yaml 直接启动时退回 scalar fake placeholder。
    - validation meta 从 `stage3_v7_b_one_step_recon` 改为：
      - `validation_mode=stage3_v7_d3_one_step_direct_decode`
      - `validation_mode_detail=not_streaming_kvcache_validation`
    - 新增 `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=1` 调试日志；
      - 在第一步 optimizer 前后计算少量参数 checksum；
      - 输出 student / fake / real 是否变化。
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
    - `[stage3_teacher_align]` 日志增加 temporal map：
      - teacher token 数；
      - teacher position 数；
      - drop / keep 的 teacher positions；
      - 对齐到的 student positions。
  - D3 release / smoke sh：
    - snapshot 增加 `train_flashvsr_stage2_v6_4_lora.py`；
    - smoke snapshot 也补 `diffsynth/models/wan_video_dit_stage2_v6_1.py`。
- 本地静态检查：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_lora.py` 通过；
  - D3 release / smoke 脚本 `bash -n` 通过。
- 6a 远端静态检查：
  - `/mnt/conda_envs/flashvsr/bin/python -m py_compile ...stage2_v6_4... ...stage3_v7_d3...` 通过；
  - D3 smoke 脚本 `bash -n` 通过。
- 6a patchcheck smoke：
  - 远端机器：
    - `6ai5mpi47f`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_teacher_aligned_20260515_v7d3_patchcheck_smoke`
  - 设置：
    - `CUDA_VISIBLE_DEVICES=0,1`
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=1`
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=1`
    - `FLASHVSR_STAGE3C_DMD_DEBUG=0`
  - temporal map 关键日志：
    - `[stage3_teacher_align] ... teacher_tokens_before=88320 teacher_positions_before=23 drop_teacher_positions=[0,1) keep_teacher_positions=[1,23) student_positions=[0,22) note=teacher_position0_is_nonstreaming_aligned_warmup`
  - optimizer ownership 关键日志：
    - `[stage3d3_optimizer_ownership] student_changed=True fake_changed=True real_changed=False ... fake_update=1`
  - 训练 step：
    - `[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
- 结论：
    - D3 当前显式丢弃 teacher `nonstreaming_aligned` warm-up position 0，保留 teacher positions `[1,23)` 对齐 student `[0,22)`；
    - 第一轮 optimizer 后 student/fake trainable 参数 checksum 变化，frozen real checksum 不变；
    - smoke 结束后 GPU0/GPU1 释放，GPU2-7 继续由 `lxh_occupy_2_7` 占卡。

### 2026-05-15：复制干净线 `v7-D3.1`，验证后替换 `v7-D2` 48GPU

- 背景：
  - 用户要求保留已经验证过的 `v7-D3`，另复制一条干净正式线；
  - 同事 0 也提醒：上 48 卡前必须确认长训关闭：
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0`
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0`
  - 仍需诚实保留边界：完整 Stage1 teacher deterministic forward 数值等价还没做；D3.1 只能说修掉 D2 最大 teacher wrapper 偏差，并通过 shape / temporal map / optimizer ownership smoke。
- 新增干净线文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_1_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-Clean-OfflineWandb.sh`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D3-1-TeacherAligned-Clean.sh`
- D3.1 相对 D3 的清理：
  - 保留：
    - `G_real/G_fake` 的 `nonstreaming_aligned` wrapper；
    - DMD 入口 `trim_front_to_match` alignment；
    - `stage3_fake_checkpoint` guard；
    - validation meta 的 D3.1 one-step direct decode 命名；
    - release/smoke snapshot 完整性。
  - 去掉：
    - D3 中只用于 smoke 的 `_stage3d3_param_checksum(...)`；
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG` 对应的 optimizer checksum 打印逻辑。
  - 长训脚本显式默认：
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0`
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0`
  - clean smoke 脚本也默认：
    - `FLASHVSR_STAGE3C_DMD_DEBUG=0`
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0`
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0`
- 本地静态检查：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d3_1_lora.py wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py` 通过；
  - D3.1 release / smoke 脚本 `bash -n` 通过。
- 6a clean smoke：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d3_1_teacher_aligned_clean_20260515_v7d31_clean_smoke`
  - 结果：
    - `[stage3c_train] epoch=0 step=1 loss=0.601123 student=0.539232 fake_loss=0.06189094 fake_update=1 ... dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
    - `stage3_teacher_align` 计数：`0`
    - `stage3d3_optimizer_ownership` 计数：`0`
  - 结论：
    - D3.1 clean 线能跑；
    - 长训不刷 teacher alignment / optimizer checksum 审查日志。
- 停止旧 `v7-D2`：
  - 旧实验：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d2_lora_89f_videoonly_authorweights_trainablefake_sharednoise_offlinewandb_20260515_v7d2_sharednoise_48gpu`
  - 6 节点：
    - `t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk`
  - 已停止：
    - `v7d2_48gpu` tmux；
    - `train_flashvsr_stage3_v7_d2_lora.py`；
    - 主节点 `wandb_package_v7d2_sharednoise`。
  - 6 节点均确认：
    - `PIDS_D2=0`
- 启动 `v7-D3.1` 48GPU：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu`
  - 6 节点 tmux：
    - `v7d31_48gpu`
  - 启动参数：
    - `MASTER_ADDR=240.12.138.137`
    - `MASTER_PORT=29531`
    - `RUN_TS_OVERRIDE=20260515_v7d31_clean_48gpu`
    - `FLASHVSR_STAGE3_TEACHER_ALIGN_DEBUG=0`
    - `FLASHVSR_STAGE3_OPT_OWNERSHIP_DEBUG=0`
  - 6 节点远端静态检查均通过：
    - `D31_NODE_STATIC_OK`
  - 当前训练状态：
    - 6 节点均有 `PIDS_D31=9`；
    - 6 节点均有 GPU 高占用；
    - 主节点已出 step 1：
      - `step=1 loss=0.854109 student=0.722192 fake_loss=0.13191710 fake_update=1 real_probe=0.093054 fake_probe=0.101467 dmd_student=0.000000 dmd_grad=0.000000 dmd_skip=0`
    - 已生成：
      - `output/training_state/step-1`
      - `output/training_state/step-2`
      - `output/validation/step-1`
      - `output/validation/step-2`
    - 未见 `Traceback` / `RuntimeError` / `CUDA out` / `Killed` / `ValueError`。
  - debug 确认：
    - 主节点 run.log 中 `stage3_teacher_align` 计数：`0`
    - `optimizer_ownership` 计数：`0`
- W&B relay：
  - 主节点 release 脚本已启动：
    - `wandb_package_v7d31_stage1teacher_clean`
  - S3 包：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_clean_offlinewandb_20260515_v7d31_clean_48gpu.tar.gz`
  - 6a 已启动长期同步 tmux：
    - `wandb_sync_from_s3_v7d31_stage1teacher_clean`
  - 已手动触发一次打包并同步成功：
    - `offline-run-20260515_011822-6nu0uhd7`
    - `offline-run-20260515_091237-60budl7y`
  - watch：
    - 本机 `watch` 前 6 个窗口已切到 D3.1 48GPU 监控；
    - 6a / 8n 窗口恢复 `watch -n 1 nvidia-smi`。

## 2026-05-16 Stage1 v5.3.5 USMGT 专用分支隔离

- 目标：
  - 从 `v5.3.5` Stage1 89f 稳定母本 `step-10000` warm-start；
  - 使用新视频源：
    - `s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/`
  - 视频源不再混旧 Takano/Yubari；
  - image branch 继续使用旧 Takano image manifest；
  - 对 GT 加 Real-ESRGAN 风格 USM sharpness，再用 sharpen 后的 GT 生成 LQ；
  - 学习率降为 `5e-6`。
- 新视频 manifest：
  - 本地：
    - `wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt`
  - S3 备份：
    - `s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt`
  - 统计：
    - `7593` 个 `.tar` shard。
- 试跑结果：
  - 初版直接在全局 `streaming_dataset.py` 上加 USM 后，16GPU 能启动并出第一条 loss：
    - 实验目录：
      - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000`
    - `step=1 loss=0.040199`
    - 首步约 `220s/it`
    - 60 秒 GPU 利用率采样：平均约 `38%`，`35/60` 秒接近 0。
  - 结论：
    - 不是通信死锁；
    - 主要瓶颈是 4K tar + 89f + CPU 退化/CPU USM 数据路径；
    - 该版本不适合作为正式训练。
- 风险处理：
  - 用户指出 `streaming_dataset.py` 是全局入口，不应继续塞入 USM/GPU 退化实验逻辑。
  - 已撤掉全局 `streaming_dataset.py` / `tar_streaming_dataset_v53.py` / `train_flashvsr_stage1_v5_3_lora.py` 中本轮新增的 USM 参数和 USM 处理。
  - 保留此前主线已有逻辑：
    - discovery 空 cache 保护；
    - 退化默认 CPU；
    - `nonstreaming_aligned` Stage1 projector 对齐逻辑。
- 新增隔离文件：
  - USMGT 专用 dataset：
    - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53_usmgt.py`
  - USMGT 专用训练入口：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`
  - USMGT smoke / 16GPU 启动脚本已改为调用专用训练入口：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Smoke-2GPU-v5-3-5-USMGT-Resume10000.sh`
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh`
- 专用 USMGT 配置：
  - `degradation_device: auto`
  - `gt_sharpen: true`
  - `gt_sharpen_backend: torch`
  - `gt_sharpen_device: auto`
  - `dataset_num_workers: 2`
- 本地检查：
  - `py_compile` 通过：
    - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53_usmgt.py`
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`
  - `bash -n` 通过：
    - 两个 USMGT 启动脚本。
- 注意：
  - 停旧 16GPU USM 实验时曾出现一次本地 shell 变量提前展开，远端命令显示为 `pkill -f ""`；已立即 `Ctrl-C` 中断。
  - 后续继续远端操作前必须重新确认目标机器和进程，避免误杀非目标训练。

## 2026-05-16 Stage1 v5.3.5 USMGT 隔离版 16GPU 正式启动

- 新 2 节点 / 16GPU 机器：
  - rank0：`bfs6vaz4d6`
  - rank1：`i6hf4scd4y`
- 初始化确认：
  - 两台机器均完成 `bash /mnt/task_runtime/bolt_lxh/setup_after_docker1.sh`；
  - `flashvsr` 环境存在；
  - `/mnt/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors` 和 `Wan2.1_VAE.pth` 均存在；
  - USMGT 专用训练入口已同步到远端：
    - `/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_5_usmgt_lora.py`
- 关键 debug 结论：
  - 首次 `GPU USM/退化 + dataset_num_workers=2` smoke 报错：
    - `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
  - 原因：
    - DataLoader worker 默认 fork，worker 内部调用 `.to(cuda)` 会触发 CUDA fork 重初始化问题。
  - 修复：
    - 保持 `degradation_device: auto`
    - 保持 `gt_sharpen_device: auto`
    - 保持 `dataset_num_workers: 2`
    - 在配置中加入：
      - `dataloader_multiprocessing_context: spawn`
  - 2GPU smoke 成功：
    - `step=1 loss=0.061863`
    - `step=2 loss=0.012738`
    - 首步约 `80s/it`
- 16GPU 正式实验：
  - 启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-5-USMGT-Resume10000-bs1-lr5e6.sh`
  - 配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_5_lora_89f_fullsources_bs1_lr5e6_aliyundegra_usmgt_resume10000.yaml`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
  - W&B：
    - online 正常；
    - run id：`8dnrur64`
  - 数据：
    - video：`takano_video_20250205_test_4k_tar_manifest.txt`
    - image：`takano_image_4k_tar_manifest.txt`
    - validation 固定样本：`3`
  - warm-start：
    - `v5.3.5` Stage1 `step-10000`
    - lq projector：`keys=8, missing=0, unexpected=0`
    - LoRA：`keys=480`
  - 已确认出 loss：
    - `step=1 loss=0.081164`
    - `step=2 loss=0.105821`
    - `step=3 loss=0.102083`
    - `step=4 loss=0.093880`
    - `step=5 loss=0.065258`
  - GPU 状态：
    - 首轮训练期间主节点 util 约 `72% - 100%`；
    - 主节点显存约 `84GB - 123GB`；
    - 未见 OOM。

## 2026-05-16 Stage3 v7 D3.2 数据低利用率定位与 48GPU 重启

- 用户约束：
  - 不减少 save 点；
  - 不取消 `stage3_decoder_cpu_offload`，否则显存风险不可接受；
  - 不 resume，修完后干净重启训练。
- 低利用率定位：
  - 在 `train_flashvsr_stage3_v7_d3_1_lora.py` 增加受控 step timing：
    - `FLASHVSR_STAGE3_TIMING_DEBUG=1` 时打印 `data/student/probe/dmd/fake/backward/optim/save_sched`；
    - 默认关闭，正式长训不刷日志。
  - 2GPU timing 显示首要瓶颈是 `data` wait：
    - 例：`data=91.296s -> 0.286s -> 62.721s`；
    - `save_sched=0`，不是保存点造成。
  - 在 `streaming_dataset.py` 增加受控数据 timing：
    - `FLASHVSR_DATA_TIMING_DEBUG=1` 时打印 open/decode/degrade/convert；
    - 默认关闭。
  - 数据 timing 结论：
    - decode 通常约数秒到十秒级；
    - online CPU degradation 可到 `61s/99s/116s`；
    - convert 约 `1s`；
    - 因此周期性 0 util 主要来自在线 CPU 退化和数据等待。
- 发现并修复的明确 bug：
  - DataLoader `spawn` worker 内 `torch.distributed` 未初始化，原 dataset 只靠 `dist` 判定 rank，导致不同 distributed rank 的 worker 可能取到相同样本；
  - timing 日志观察到 rank0/rank1 同时处理相同视频样本，浪费 CPU/I/O 并放大 slowest-rank wait；
  - `streaming_dataset.py` 新增 env fallback：
    - 优先使用 `torch.distributed`；
    - 未初始化时使用 `RANK/WORLD_SIZE`；
    - worker RNG seed 纳入 rank；
    - tar URL datapipe 先按 rank 分片，再按 worker 分片。
- 诊断验证：
  - env-rank fix 后 rank0/rank1 不再重复处理同一批样本；
  - workers=4、prefetch=1、offload=true 的 2GPU 诊断结果：
    - `data=91.664s -> 0.287s -> 0.376s`；
    - 说明首步仍可能受慢样本/预热影响，但后续数据等待明显改善。
- 正式 D3.2 文件：
  - config：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml`
  - wrapper：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-2-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-OfflineWandb.sh`
  - 保留：
    - `save_steps: 500`
    - `extra_save_steps: "1,2,5,10,20,50,100,200"`
    - `stage3_decoder_cpu_offload: true`
    - validation 设置不改；
  - 调整：
    - `dataset_num_workers: 4`
- 占卡程序处理：
  - 五个非 t5 节点的 `gpu_hold_debug` session 被精确停止；
  - 发现 session 已不存在但 8 个 `/tmp/gpu_hold_debug.py` 子进程仍占 GPU；
  - 先用硬编码 PID 列表 `ps -o pid,args` 确认均为 `/tmp/gpu_hold_debug.py`，再只 kill 这些精确 PID；
  - 未使用 `pkill`、未使用模糊进程匹配。
- 权限修复：
  - 首次 D3.2 启动失败，退出 `V7D32_EXIT=126`；
  - 原因是 D3.2 wrapper `exec` 到 D3.1 base script，但远端 base script 缺 executable bit；
  - 本地 `chmod +x` base script 后等待同步，远端 t5/a9 均确认 `-rwxr-xr-x`。
- 48GPU 正式重启：
  - 6 节点：
    - `t5qdtykjsw a9suya6gxe 67dxkwcb7m ui9n6p293s g48bd6x4h7 gx2intv5rk`
  - master：
    - `MASTER_ADDR=240.12.138.137`
    - `MASTER_PORT=29541`
  - run timestamp：
    - `RUN_TS_OVERRIDE=20260516_v7d32_datafix_48gpu_fresh2`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
  - 启动后状态：
    - 已进入 accelerate/NCCL 初始化和模型加载；
    - 日志确认 `stage3_decode context_mode=full_prefix decoder_cpu_offload=True`；
    - a9 节点显存约 `5.6GB-6.3GB`，部分卡 util 达 `100%`；
    - 后续主节点显存约 `77GB`，多卡 util 约 `97%-100%`；
    - 已出现训练 loss：
      - `[stage3_v7_b_loss] loss=0.091168 ... decoder_cpu_offload=True`
      - `[stage3_v7_b_loss] loss=0.209730 ... decoder_cpu_offload=True`
  - 继续观察：
    - 6 个 48 卡节点显存稳定约 `146GB-157GB / 183GB`；
    - 多轮 watch 大多数 GPU util 为 `97%-100%`，个别 GPU 某一秒低到 `9%/55%/72%` 但不是全节点一起掉 0；
    - 主节点已到 `step=6`，最新观察：
      - `step=6 loss=0.425419 student=0.248533 fake_loss=0.16397065 dmd_student=0.012915 dmd_grad=0.119689 dmd_skip=0`
    - `run.log` 暂未见 `Traceback / RuntimeError / CUDA out / Killed / ValueError`。
  - W&B：
    - 当前是 offline W&B，不是训练进程在线直连 W&B；
    - 本地 offline run：
      - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2/wandb/offline-run-20260515_123145-y2alwjkp`
    - 后台 package session：
      - `wandb_package_v7d31_stage1teacher_clean`
    - session 名沿用 D3.1，但 `RUN_DIR` 指向 D3.2 fresh2；
    - 初始 S3 包已上传：
      - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2.tar.gz`
    - package 间隔为 `3600s`，需要后续 `wandb sync` 后才会出现在 W&B 网页。

## 2026-05-16 Stage1 USMGT 数据与可视化检查补充

- 当前 16GPU USMGT 配置确认：
  - `degradation_device: auto`
  - `gt_sharpen: true`
  - `gt_sharpen_backend: torch`
  - `gt_sharpen_device: auto`
  - `dataset_num_workers: 2`
  - `dataloader_multiprocessing_context: spawn`
  - `validation_num_samples: 3`
- 远端 CUDA 可用时，USM 和退化均按 `LOCAL_RANK` 解析到 `cuda:<LOCAL_RANK>`，即当前训练是 GPU 上做 GT sharpen 和 LQ degradation。
- 新 Takano 20250205 4K manifest 已确认备份：
  - `s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt`
  - 共 `7593` 个 tar shard。
- 可视化导出：
  - 新增只读脚本：`tools/flashvsr_export_usmgt_samples.py`
  - 导出内容：
    - 3 组 video branch：`gt_usm` / `lq_degraded`
    - 3 组 image branch：`image_gt_usm` / `image_lq_degraded`
    - 5 组 GT sharpness 对比：`gt_raw` / `gt_usm`
  - 为避免抢正在训练的 GPU 显存，可视化导出使用 CPU/opencv 路线；训练本身仍为 GPU auto 路线。
  - 远端：`/mnt/task_wrapper/user_output/artifacts/usmgt_checks/takano20250205_usmgt_checks_20260516`
  - 本机：`/Users/lixiaohui/Desktop/takano20250205_usmgt_checks_20260516`
- conductor 经验：
  - 远端普通后台 shell 里直接 conductor 上传曾报 `Unable to locate credentials`；
  - 同一机器用 `zsh -lc` 正常执行 `conductor s3 ls` 和 `conductor s3 cp` 成功；
  - 后续不要直接认定 conductor 不可用，先确认是否是非登录 shell 没继承认证环境。
- 新视频源抽样：
  - 正在从 `s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/` 的 tar 中抽 100 个原始视频到桌面；
  - 本机目录：`/Users/lixiaohui/Desktop/takano20250205_100videos`。

## 2026-05-16 Stage3 D3.2 W&B 远端自动同步修复

- 用户反馈：
  - W&B 网页里没看到 D3.2/`v7d32` 实验；
  - 明确要求不要尝试本机同步，优先找新的 16 卡机器作为远端 relay；
  - 补充 conductor 经验：主机、从机、本地一般都有 conductor 凭证；如果拉不下来，先怀疑 shell 环境不对，本机还要先 `proxy_off`。
- 已读新同事记录：
  - 新 16GPU Stage1 USMGT 任务在 `bfs6vaz4d6` / `i6hf4scd4y`；
  - `bfs6vaz4d6` 的 Stage1 W&B online run 正常，run id 为 `8dnrur64`。
- t5 检查：
  - `zsh -lc` 下 `conductor s3 ls` 正常；
  - `~/.netrc` 存在；
  - `wandb status` 显示 `api_key: null`，但这不能代表 netrc 登录态不可用；
  - 后续直接实测 `wandb sync` 成功，所以 t5 也可以作为 W&B cloud sync 端；
  - 旧 session `wandb_package_v7d31_stage1teacher_clean` 仍在，但名字沿用 D3.1、间隔 3600s。
- `bfs6vaz4d6` 检查：
  - `zsh -lc` 下 `conductor s3 ls` 正常；
  - 普通 `wandb status` 也显示 `api_key: null`，但 `~/.netrc` 存在；
  - 用 `wandb sync` 实测成功，不需要本机介入。
- 手动同步结果：
  - S3 包：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2.tar.gz`
  - 当前 offline run：
    - `offline-run-20260515_123145-y2alwjkp`
  - W&B sync 返回：
    - `https://wandb.ai/veralee/flashvsr/runs/y2alwjkp ... done.`
- 新增远端自动链路：
  - t5 direct sync:
    - tmux session `wandb_sync_d3_2_t5_direct`
    - 每 900s 直接执行：
      - `wandb sync .../wandb/offline-run-20260515_123145-y2alwjkp`
    - t5 有 `~/.netrc`，直接一次性 sync 已确认返回 `done`；
    - 循环版加 `timeout 180s`，避免某轮 `wandb sync` 卡住后阻塞后续重试。
  - t5:
    - tmux session `wandb_package_d3_2_fresh2`
    - 每 900s 打包 D3.2 run 的 `wandb/` 并上传同一个 S3 包；
    - 首轮确认包含：
      - `offline-run-20260515_011822-6nu0uhd7`
      - `offline-run-20260515_123145-y2alwjkp`
    - 首轮上传后 S3 包大小更新为 `133620` bytes，时间 `2026-05-15 13:09:22 -0700`。
  - `bfs6vaz4d6`:
    - tmux session `wandb_sync_d3_2`
    - 每 900s 从 S3 下载最新包并执行：
      - `wandb sync .../wandb/offline-run-20260515_123145-y2alwjkp`
    - 首轮与手动刷新均返回 `done`。
- 当前 W&B 页面应查看：
  - project/entity：`veralee/flashvsr`
  - run id：`y2alwjkp`
  - URL：`https://wandb.ai/veralee/flashvsr/runs/y2alwjkp`
- 后续注意：
  - t5 现在已确认既可 package，也可直接 `wandb sync`，当前应作为主 W&B 同步链路；
  - `bfs6vaz4d6` 也已确认可作为 cloud sync relay，但现在只是冗余/备用路径；
  - 不要仅凭 `wandb status` 里的 `api_key: null` 判断不能同步，需检查 `~/.netrc` 或直接做一次不打印 token 的 `wandb sync` 实测；
  - 如果后续 conductor 在非交互/非登录 shell 报凭证问题，不要马上判断无凭证，先换正常 `zsh -lc` 环境验证；
  - 本机若必须查 conductor，需要先在 zsh 里 `proxy_off`。
  - 如果本机当前 shell 里 `bolt task list` 卡住，不要在该 shell 里死等；到本机 `lxh` tmux session 新开一个窗口，先执行 `proxy_off`，再运行 `bolt task list`，通常可正常返回。

## 2026-05-16 Stage1 USMGT step>=100 测试准备

- 用户要求：
  - 对昨天启动的 16GPU Stage1 USMGT 微调实验，从 `step-100` 开始测 10 个测试视频；
  - 测完下载到桌面，用于判断 USMGT/新 Takano 4K 特训后是否变好。
- 目标实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
  - 当前 rank0：`bfs6vaz4d6`
  - 当前已确认 ckpt：`step-100.safetensors`、`step-200.safetensors`、`step-500.safetensors`、`step-1000.safetensors`
- ckpt 中转：
  - 已从 rank0 上传到：
    - `s3://lxh/tmp/usmgt_stage1_takano20250205_step100_20260516/`
- 新增测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage1_usmgt_scan89_step100_20260516.sh`
  - 默认测试集：
    - LQ：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
    - GT：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/gt`
  - 测试集缺失时会从旧测试机 artifacts 恢复：
    - `s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
    - `s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/gt`
  - 推理设置：
    - `infer_flashvsr_stage1_v5_3_aligned_batch.py`
    - `num_frames=89`
    - `lq_proj_temporal_mode=nonstreaming_aligned`
    - `input_bicubic_upscale=4.0`
    - `num_inference_steps=50`
    - `color_fix_method=adain`
    - 保存 SR，同时保存输入 LQ，并复制 GT。
- 当前阻塞：
  - 当前 `bolt task list` 仅显示两个运行任务：
    - 新 2 节点 16GPU：正在跑 Stage1 USMGT；
    - 6 节点母机：检查 `t5qdtykjsw` / `67dxkwcb7m` 均有 v7 Stage3 Python 训练进程和对应远端 tmux，不是单纯占卡；
  - 因此暂未释放 GPU 跑测试，避免误杀正在进行的训练；
  - 等用户释放测试机/允许暂停某个 v7 节点后，可直接在目标机器拉取 ckpt 并执行上述脚本。

## 2026-05-16 Stage3 D3.2 step>=100 非 tiled 推理评测

- 用户要求：
  - 对当前 Stage3 D3.2 已有 checkpoint，从 `step-100` 及之后开始，用正确三阶段测试代码跑 10 个合成测试集；
  - 三阶段测试应沿用二阶段流式/KV-cache 推理路径，但变成一步；
  - 不能用 tiled 版本作为正式观感评测，因为 tiled 会影响用户判断效果。
- 当前三阶段测试代码确认：
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d.py`
  - `wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py`
  - 这两个入口复用 `infer_flashvsr_stage2_v6_1.py` 的 `build_stage2_pipe` / `run_single_video` / `infer_from_lq_streaming`，并把 `num_inference_steps` 固定为 `1`；
  - 即：Stage2 官方风格 streaming/KV-cache 路径 + Stage3 one-step。
- 测试集：
  - 远端本地默认路径：
    - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
  - 新机缺失时从旧 my artifacts 恢复：
    - `s3://bolt-prod-2320845741/tasks/myj7ukyewz/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
- 当前 D3.2 checkpoint 来源：
  - 训练目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
  - 已确认本地训练输出里存在：
    - `output/step-100.safetensors`
    - `output/step-200.safetensors`
    - `output/step-500.safetensors`
  - 已从 t5 上传到 ckpt 中转：
    - `s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/`
- 新增安全测试脚本：
  - `wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`
  - 特点：
    - 不执行 `pkill`；
    - 不恢复/启动占卡；
    - 自动从 S3 拉测试集和 checkpoint；
    - 支持 `TILED=0/1`；
    - 默认输出 summary/timing 并同步到 S3。
- 叠在 6 节点母机上的试跑：
  - 节点：`gx2intv5rk`
  - 首次 `TILED=0` 叠在训练卡上，VAE decode OOM：
    - 非 tiled decode 需要额外申请约 `18.98 GiB`；
    - 当时 GPU0 只剩约 `18.50 GiB`；
    - 结论：不能叠在训练卡上做正式非 tiled 评测。
  - 临时 `TILED=1` 可以跑通，但用户明确不希望用 tiled 结果看效果；
  - 已停止该 tiled 评测，仅留下临时输出，不作为正式结果。
- 新 4 卡机器初始化：
  - task id：`6ikhpjzv3z`
  - 已按“新机初始化”流程完成：
    - 本机 `bolt task patchsshconfig 6ikhpjzv3z`；
    - 本机 `sync` tmux 新增 `6ikhpjzv3z` 窗口；
    - 本机 `watch` tmux 新增 `6ikhpjzv3z` 窗口并运行 `watch -n 1 nvidia-smi`；
    - 远端执行 `/mnt/task_runtime/bolt_lxh/setup_after_docker1.sh`；
    - `conda env list` 已确认存在 `/mnt/conda_envs/flashvsr`；
    - `conductor s3 ls s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/` 可列出 `step-100/200/500.safetensors`；
    - `/mnt/task_runtime/lucidvsr` 已通过本机 sync 同步，未在远端手写项目代码。
- 正式非 tiled Stage3 D3.2 评测：
  - 机器：`6ikhpjzv3z`
  - 远端 tmux：`stage3_eval:d32_100plus_g01`
  - 使用 GPU：`0,1`
  - 运行参数：
    - `TILED=0`
    - `GPU_LIST=0,1`
    - `MAX_PARALLEL=2`
    - `num_inference_steps=1`
    - `input_bicubic_upscale=4.0`
    - `color_fix_method=adain`
    - `stage2_attention_mode=block_sparse_chunk_causal`
    - `stage2_topk_ratio=2.0`
    - `stage2_local_num=-1`
    - `stage2_kv_ratio=3.0`
  - 远端输出：
    - `/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
  - S3 输出：
    - `s3://lxh/data/test/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
  - 完成结果：
    - `step-100`: `10/10`, `status=0`, `196s`, `19.6s/video`
    - `step-200`: `10/10`, `status=0`, `197s`, `19.7s/video`
    - `step-500`: `10/10`, `status=0`, `180s`, `18.0s/video`
    - `summary_counts.txt`: `step-100 10`, `step-200 10`, `step-500 10`
    - `timing_summary.txt` 已同步到 S3。
- 额外只读观察：
  - Stage3 评测结束后 GPU0/1 释放为 0；
  - GPU2/3 上有 `infer_flashvsr_stage1_v5_3_aligned_batch.py` 的 Stage1 USMGT batch 推理进程，显存约 `49456 MiB`/卡；
  - 这不是本次 Stage3 评测启动的进程，未做任何停止或 kill 操作。

## 2026-05-16 Stage1 USMGT Takano 20250205 评测补测
- 目标：
  - 测试 16 卡 Stage1 USMGT/Takano20250205 fine-tune 的 early checkpoints。
- 测试机器：
  - `6ikhpjzv3z`
  - 使用 GPU `2,3`；GPU `0,1` 上有其他旧评测任务，未触碰。
- 被测训练实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
- 被测 ckpt：
  - `step-100`
  - `step-200`
  - `step-500`
  - `step-1000`
- 推理设置：
  - Stage1 aligned inference。
  - `lq_proj_temporal_mode=nonstreaming_aligned`，不走 Stage2/streaming cache。
  - `input_bicubic_upscale=4.0`
  - `num_inference_steps=50`
  - `color_fix_method=adain`
  - `save_input_lq=true`
- 测试集：
  - 10 个 synthetic 89f 视频。
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503`
- 远端结果：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step100_20260516`
- S3 结果：
  - `s3://lxh/artifacts/inference/stage1_usmgt_takano20250205_step100_20260516`
- 本地下载：
  - `/Users/lixiaohui/Desktop/stage1_usmgt_takano20250205_step100_20260516`
- 完成校验：
  - `usmgt_step100`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step200`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step500`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step1000`: `10` 个 SR，`10` 个 saved LQ。

## 2026-05-17 Stage1 USMGT Takano 20250205 新 ckpt 补测
- 目标：
  - 检查 16 卡 Stage1 USMGT/Takano20250205 fine-tune 是否有新 ckpt，并补测新增模型。
- 训练主机：
  - `bfs6vaz4d6`
- 测试机器：
  - `6ikhpjzv3z`
  - 只使用 GPU `2,3`；GPU `0,1` 上的旧任务未触碰。
- 被测训练实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
- 新发现 ckpt：
  - `step-1500`
  - `step-2000`
  - `step-2500`
  - 训练当时仍在继续，日志约到 `step=2652`。
- 推理设置：
  - Stage1 aligned inference。
  - `lq_proj_temporal_mode=nonstreaming_aligned`，不走 Stage2/streaming cache。
  - `input_bicubic_upscale=4.0`
  - `num_inference_steps=50`
  - `color_fix_method=adain`
  - `save_input_lq=true`
- 测试集：
  - 10 个 synthetic 89f 视频。
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503`
- 远端结果：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
- S3 结果：
  - `s3://lxh/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
- 本地下载：
  - `/Users/lixiaohui/Desktop/stage1_usmgt_takano20250205_step1500_2500_20260517`
- 完成校验：
  - `usmgt_step1500`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step2000`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step2500`: `10` 个 SR，`10` 个 saved LQ。
- 收尾：
  - `6ikhpjzv3z` GPU `2,3` 已重新启动占卡程序。
  - 最后检查 GPU `2,3` 显存约 `125128 MiB`，GPU 利用率分别约 `100%` / `84%`。

## 2026-05-17 Stage1 USMGT Takano 20250205 step-3000 补测
- 目标：
  - 在已有 Stage1 USMGT 评测目录中继续补测最新 `step-3000`。
- 训练主机：
  - `bfs6vaz4d6`
- 测试机器：
  - `6ikhpjzv3z`
  - 只使用 GPU `2,3`；实际只有一个 ckpt，因此推理占用 GPU `2`，GPU `3` 在测试期间临时占卡。
- 被测训练实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn`
- 新发现 ckpt：
  - `step-3000`
  - 训练到 `step=3000` 后结束，W&B 已同步。
- 推理设置：
  - Stage1 aligned inference。
  - `lq_proj_temporal_mode=nonstreaming_aligned`，不走 Stage2/streaming cache。
  - `input_bicubic_upscale=4.0`
  - `num_inference_steps=50`
  - `color_fix_method=adain`
  - `save_input_lq=true`
- 测试集：
  - 10 个 synthetic 89f 视频。
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503`
- 结果目录沿用上一轮：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
  - S3：`s3://lxh/artifacts/inference/stage1_usmgt_takano20250205_step1500_2500_20260517`
  - 本地：`/Users/lixiaohui/Desktop/stage1_usmgt_takano20250205_step1500_2500_20260517`
- 完成校验：
  - `usmgt_step1500`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step2000`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step2500`: `10` 个 SR，`10` 个 saved LQ。
  - `usmgt_step3000`: `10` 个 SR，`10` 个 saved LQ。
- 收尾：
  - `6ikhpjzv3z` GPU `2,3` 已重新启动占卡程序。
  - 最后检查 GPU `2,3` 利用率均为 `100%`，显存约 `166600 MiB`。

## 2026-05-17 Stage3 D3.2 loss 复查与后续保存间隔调整
- 背景：
  - 用户观察到当前 Stage3 D3.2 loss 曲线不够直观，要求对照 DMD/DMD2/OSEDiff 论文重新判断 loss 是否应该下降；
  - 当前正在运行的 D3.2 实验仍然使用启动时的 `save_steps=500`，不能通过本地改配置热修改该 run 的保存频率。
- 当前 D3.2 训练目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
- 当前 loss 统计结论（截至约 `step=999`）：
  - `stage3c_train/loss` 总体横盘，前 100 step 均值约 `1.036`，最近 100 step 均值约 `1.049`；
  - `student` / reconstruction 相关项有小幅下降，`student` 前 100 均值约 `0.636`，最近 100 均值约 `0.606`；
  - `fake_loss` 明显下降，前 100 均值约 `0.0697`，最近 100 均值约 `0.0461`；
  - `dmd_student` 没有下降，前 100 均值约 `0.330`，最近 100 均值约 `0.397`；
  - `dmd_grad` 基本横盘，前 100 均值约 `0.550`，最近 100 均值约 `0.566`；
  - `dmd_skip=0`，未见 DMD spike skip。
- 论文/代码复查结论：
  - DMD/DMD2 的核心是匹配分布，DMD loss 是由 `real score - fake score` 给出的梯度代理，不应期待像普通 MSE 一样单调下降；
  - DMD2 明确指出，去掉原 DMD regression loss 后，不稳定性主要来自 `G_fake` / fake score estimator 追不上 generator 的非平稳输出分布，因此推荐 TTUR：多次更新 fake critic 再更新一次 generator；
  - DMD2 SDv1.5 参考配置名中使用 `dfake10`，论文正文 ImageNet ablation 提到 5 fake score updates per generator update 可稳定无 regression loss 训练；
  - OSEDiff 的 VSD/DMD 风格 regularization 同样是分布正则，生成质量需要结合重建项、可视化 checkpoint、FID/LPIPS/NIQE 等评价，不能只看总 loss 是否下降。
- 对当前 D3.2 的判断：
  - 当前 run 没有爆炸，`fake_loss` 和 reconstruction/student 项有学习信号；
  - 但 DMD 项没有表现出更接近收敛的趋势，说明仅看 loss 不能确认 DMD 分布匹配已经变好；
  - 当前 `stage3_fake_update_every_n_steps=1` 等价于 student 每步只更新一次 fake critic，不是 DMD2 推荐的 dfake5/dfake10 TTUR，后续若视觉质量不随 checkpoint 改善，需要优先考虑提高 fake critic 更新频率或分阶段 warm-up。
- 后续保存策略：
  - 已将后续 D3.2 配置 `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml` 的 `train.save_steps` 从 `500` 改为 `100`；
  - 该修改只影响以后重启/新启动的训练，不影响当前已经运行中的 D3.2 run。

## 2026-05-17 Stage3 D3.2 step-1000 合成测试集补测
- 目标：
  - 用户确认当前 D3.2 已产出 `step-1000`，要求用正确 Stage3 一步推理代码补测，并下载到之前的桌面结果目录；测试完成后恢复 `6ikhpjzv3z` GPU `0,1` 占卡。
- 被测 checkpoint：
  - 训练目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
  - checkpoint：`output/step-1000.safetensors`
  - 中转 S3：`s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/step-1000.safetensors`
- 测试机器：
  - `6ikhpjzv3z`
  - 临时停止并复用 GPU `0,1` 的 `occupy01` 占卡；GPU `2,3` 原占卡未触碰。
- 推理脚本与设置：
  - `wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`
  - `MIN_STEP=1000`
  - `STEP_MOD=1000`
  - `TILED=0`
  - `GPU_LIST=0,1`
  - `MAX_PARALLEL=1`
  - `num_inference_steps=1`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_kv_ratio=3.0`
- 测试集：
  - 10 个 synthetic 89f 视频。
  - `/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503/lq`
- 结果位置：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
  - S3：`s3://lxh/data/test/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
  - 本地：`/Users/lixiaohui/Desktop/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
- 完成校验：
  - `step-1000`: `10` 个 mp4。
  - `summary_counts.txt` 已包含：`step-100 10`、`step-200 10`、`step-500 10`、`step-1000 10`。
  - `logs/step-1000.time`：`status=0`，`num_inputs=10`，`num_outputs=10`，`seconds=181`，`seconds_per_video=18.100`，`tiled=0`。
- 收尾：
  - 本地已从 S3 同步到桌面旧结果目录。
  - `6ikhpjzv3z` GPU `0,1` 已重新启动 `occupy01` 占卡；最后检查 GPU `0,1,2,3` 利用率均为 `100%`。
- dfake 语义备注：
  - DMD2 官方代码里的 `dfake_gen_update_ratio` 控制 generator/student 是否在当前 step 更新：`COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0`，而 fake/guidance model 基本每 step 更新。
  - 因此在 DMD2 语义里，数字越大，代表每次 generator 更新之间 fake critic 更新次数越多；`dfake=5` 约等价于 fake 更新 5 次、generator 更新 1 次。
  - DMD2 论文的 ImageNet ablation 重点推荐 `5 fake score updates per generator update`；官方实验脚本里 ImageNet/SDXL 多用 `--dfake_gen_update_ratio 5`，SDv1.5 参考脚本多用 `--dfake_gen_update_ratio 10`。
  - 当前 FlashVSR D3.2 的 `stage3_fake_update_every_n_steps=1` 不是 DMD2 的 `dfake_gen_update_ratio`，而是“每 N 个 student step 更新一次 fake”。把它设成 `5` 会让 fake 更少更新，方向与 DMD2 TTUR 相反。
  - 如果后续要按 DMD2 语义引入 `dfake=5`，需要新增真正的 fake 多步 / student 少步调度，不能直接改当前 `stage3_fake_update_every_n_steps`。

## 2026-05-17 Stage3 v7-D4 dfake=5 新训练线
- 目标：
  - 用户要求在不动正在运行的 v7-D3/D3.2 代码线的前提下，复制新线 `v7-D4`，实现真正 DMD2 语义的 `dfake=5`；
  - 明确不能把当前 `stage3_fake_update_every_n_steps` 直接改成 `5`，因为旧参数语义是 fake 每 N 个 student step 更新一次，方向与 DMD2 TTUR 相反。
- 新增文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_dfake5_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-Dfake5-OfflineWandb.sh`
- D4 调度语义：
  - 新参数：`stage3_dfake_gen_update_ratio: 5`
  - runner iteration 每步都用当前 student/generator 产出的 `z_pred.detach()` 更新 `G_fake`；
  - student/generator 只在 `runner_step % stage3_dfake_gen_update_ratio == 0` 时更新；
  - 因此 `dfake=5` 约等价于每 5 个 runner iteration 中 fake critic 更新 5 次、student/generator 更新 1 次；
  - checkpoint / `ModelLogger.num_steps` 只跟 student/generator update 计数走，不把 fake-only iteration 记成新的 student checkpoint step。
- 代码差异：
  - `train_flashvsr_stage3_v7_d4_lora.py` 新增 `_stage3d4_dfake_gen_update_ratio(args)` 与 `_stage3d4_is_generator_turn(args, runner_step)`；
  - `_stage3c_fake_fm_loss(...)` 在 D4 中不再用 `stage3_fake_update_every_n_steps` 跳过 fake update，fake update 每个 runner iteration 都执行；
  - fake-only iteration 中 student forward 用 `torch.no_grad()` 只生成当前 fake distribution 的 `z_pred`，不更新 student；
  - generator turn 中仍计算原有 reconstruction/student loss、DMD student loss、fake FM loss，并分别 step student optimizer 与 fake optimizer；
  - 日志新增 `runner_step`、`generator_update`、`dfake_gen_update_ratio`，wandb 同步 payload 也加入这些字段。
- D4 配置：
  - `save_steps: 100`
  - `stage3_fake_fm_weight: 1.0`
  - `stage3_fake_update_every_n_steps: 1` 仅保留 legacy/兼容，不作为 DMD2 dfake 语义使用；
  - `stage3_dfake_gen_update_ratio: 5`
  - wandb name：`flashvsr-stage3-48gpu-v7-d4-stage1teacher-aligned-authorweights-datafix-dfake5-offline`
  - output tag：`train_stage3_release_48gpu_v7_d4_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_dfake5_offlinewandb`
- D4 启动脚本：
  - 直接指向 D4 config 与 D4 train py；
  - offline wandb 打包 tmux session 改为 `wandb_package_v7d4_dfake5`；
  - `TRAIN_PROCESS_PATTERN` 改为 `train_flashvsr_stage3_v7_d4_lora.py`。
- 已完成本地校验：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_lora.py`
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-Dfake5-OfflineWandb.sh`
- 尚未做：
  - 还没有在远端做 2GPU smoke；正式启动前仍建议先 smoke 看日志是否呈现 `generator_update=1,0,0,0,0,1...`，并确认 fake optimizer 每个 runner iteration 都更新。

## 2026-05-17 Stage3 v7-D4 loss/optimizer 顺序审查与 2GPU smoke
- 背景：
  - 用户追问 pixel loss 绑定在哪个 optimizer、几个 optimizer、各 loss 的计算/回传/更新顺序，以及 FlashVSR/DMD2 论文把 loss 写在一起时工程实现是否也应混合回传。
- 对 DMD2 官方代码的复查：
  - DMD2 官方 `main/train_sd.py` 中有两个 optimizer：
    - `optimizer_generator`：更新 one-step generator / feedforward model；
    - `optimizer_guidance`：更新 fake score / guidance model。
  - 官方训练顺序不是把 `loss_dm + loss_fake_mean` 合成一个 total loss 一次 backward：
    - generator turn：先算 generator 输出和 `loss_dm`，当 `step % dfake_gen_update_ratio == 0` 时 `accelerator.backward(generator_loss)` 并 `optimizer_generator.step()`；
    - guidance/fake turn：再用 generator turn 产出的 detached fake sample 算 `loss_fake_mean`，`accelerator.backward(guidance_loss)` 并 `optimizer_guidance.step()`；
    - 两个 optimizer 之间显式 `zero_grad()`，避免梯度串线。
- FlashVSR D4 中各 loss 当前归属：
  - student optimizer：
    - `stage3_v7_b_loss` 内的 flow/MSE/LPIPS/first-frame pixel/first-frame LPIPS；
    - DMD student loss，即 real/fake score difference 形成的 `dmd_student_loss`；
  - fake optimizer：
    - `stage3_fake_fm_weight` 对应的 fake FM loss，用 `student_z_pred.detach()` 训练 `G_fake`；
  - no optimizer：
    - `G_real` / Stage1 teacher，全程 frozen；
    - DMD probe 中 `G_fake` 用于 score difference 时也走 no-grad probe，不让 DMD student loss 更新 fake。
- 尝试过的 D4 两段 backward：
  - 为贴近 DMD2 官方，曾临时把 D4 改成 generator turn 中先 `accelerator.backward(generator_loss)` / student optimizer step，再单独 fake backward / fake optimizer step。
  - 2GPU smoke 第一次在同一 iteration 内第二次使用 `accelerator.backward(fake_loss)` 时触发 DeepSpeed ZeRO2 `IndexError: list index out of range`。
  - 改成 fake loss 使用普通 `fake_loss.backward()` 后，能跑过：
    - `runner_step=0 generator_update=1 fake_update=1`
    - `runner_step=1/2/3/4 generator_update=0 fake_update=1`
  - 但到 `runner_step=5` 第二次 generator update 时，DeepSpeed/NCCL 报 `DistBackendError: NCCL communicator was aborted on rank 0`。
  - 结论：在当前 runner 结构里，student 被 DeepSpeed engine 管、`G_fake` 是手动 optimizer，不能直接把官方 DMD2 的物理两段 backward 照搬进同一个 ZeRO2 iteration；混排 DeepSpeed backward 和普通 fake backward 会破坏后续 generator turn。
- 当前 D4 处理：
  - 已回退到 ZeRO2 稳定的一次 backward 结构：
    - `total_loss = fake_loss`
    - generator turn 时再加 `student_loss` 和 `dmd_student_loss`
    - 每个 runner iteration 只调用一次 `accelerator.backward(total_loss)`
  - 梯度归属仍靠结构保证：
    - fake FM loss 的输入 `clean_latents` 使用 `detach()`，所以 fake loss 不反传 student；
    - DMD real/fake probes 在 `torch.no_grad()` 中运行，所以 DMD student loss 不更新 `G_real/G_fake`；
    - fake-only runner step 中 student forward 在 `torch.no_grad()` 中只生成当前 fake distribution 的 `z_pred`。
- smoke 记录：
  - 机器：`6ikhpjzv3z`
  - 使用 GPU：`0,1`
  - smoke window：`stage3_smoke:d4_dfake5_g01`
  - 长 smoke 目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_dfake5_20260517_d4dfake5_smoke_g01_fixbackward`
  - 长 smoke 观察到的有效日志：
    - `runner_step=0 generator_update=1 fake_update=1`
    - `runner_step=1 generator_update=0 fake_update=1`
    - `runner_step=2 generator_update=0 fake_update=1`
    - `runner_step=3 generator_update=0 fake_update=1`
    - `runner_step=4 generator_update=0 fake_update=1`
  - 长 smoke 最终失败点：
    - `runner_step=5` 第二次 generator update 期间 NCCL communicator abort；
    - 该失败对应临时两段 backward 版本，不是当前回退后的 D4 文件。
  - 回退后短 smoke 目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_dfake5_20260517_d4dfake5_smoke_g01_singlebackward_short`
  - 回退后短 smoke 结果：
    - `runner_step=0 generator_update=1 dfake_gen_update_ratio=5`
    - `loss=1.019969 student=0.975444 fake_loss=0.04452544 fake_update=1 real_probe=0.135365 fake_probe=0.171131 dmd_student=0.000000`
    - 该短 smoke 正常结束，未见 traceback。
- 当前边界：
  - D4 已具备真正 dfake=5 调度语义和短 smoke 通过；
  - 但还没有在当前“单 backward + detach 隔离”的最终 D4 文件上完整跑到 `runner_step=5`，因此正式 48 卡前仍建议再做一次更长 smoke，确认 `runner_step=5 generator_update=1` 也能正常结束；
  - 如果一定追求与 DMD2 官方物理两段 backward 完全一致，需要重构 fake model 的分布式包装/optimizer，使 fake turn 不混用 student DeepSpeed engine 状态。
- 收尾：
  - 失败 smoke 残留进程只按明确 PID 清理：`287955/288160/288161`；
  - `6ikhpjzv3z` GPU `0,1` 已恢复 `occupy01` 占卡；
  - 最后检查 GPU `0,1,2,3` 显存占用均恢复，占卡利用率均为 `100%`。
- 后续提醒：
  - 用户计划 v7-D4 正式训练时换用新的 Stage1 pretrain；
  - 当前 D4 launch 仍默认旧 Stage1 `train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501.../step-10000.safetensors`；
  - 正式启动 v7-D4 前必须替换 `STAGE3_REAL_CHECKPOINT` 与 `STAGE3_FAKE_CHECKPOINT` 到新的 Stage1 pretrain 结果。

## 2026-05-17 Stage3 v7-D3.2 checkpoint 复查与 v7-D4.1 turn-isolated 代码线
- 用户要求：
  - 先检查当前 7d3.2 是否有新 checkpoint；如果有就继续用正确 Stage3 测试代码评测并下载到之前桌面结果目录；
  - 同时新写一条完全隔离的 7d4.1 代码线，希望更接近 DMD2 官方两个模型轮换更新，而不是把 loss 合在一个 backward 里。
- 7d3.2 checkpoint 状态：
  - 远端主机：`t5qdtykjsw`
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2/output`
  - 只读检查结果仍只有：
    - `step-1.safetensors`
    - `step-2.safetensors`
    - `step-5.safetensors`
    - `step-10.safetensors`
    - `step-20.safetensors`
    - `step-50.safetensors`
    - `step-100.safetensors`
    - `step-200.safetensors`
    - `step-500.safetensors`
    - `step-1000.safetensors`
  - 没有发现 `step-1500` / `step-2000` 等新模型，因此本轮未启动新评测，也未改动之前桌面结果目录：
    - `/Users/lixiaohui/Desktop/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
- 新增 v7-D4.1 文件：
  - 训练代码：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - 48 卡配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_1_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_turnisolated_dfake5_offlinewandb.yaml`
  - 48 卡启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-TurnIsolated-Dfake5-OfflineWandb.sh`
- v7-D4.1 调度语义：
  - `stage3_dfake_gen_update_ratio: 5` 在 D4.1 中解释为 `dfake_fake_updates_per_generator=5`；
  - runner turn 序列为：
    - `runner_step=0`: generator/student turn，只更新 student optimizer；
    - `runner_step=1..5`: fake-only turn，只更新 `G_fake` optimizer；
    - `runner_step=6`: 下一次 generator/student turn；
  - 因此它是物理隔离的“1 次 student 更新 + 5 次 fake 更新”循环。
- v7-D4.1 梯度/optimizer 归属：
  - generator turn：
    - 正常跑 Stage3 student forward；
    - 计算 FlashVSR reconstruction loss 与 DMD student loss；
    - 只调用 `accelerator.backward(generator_total_loss)`；
    - 只执行 `optimizer.step()` 和 student scheduler；
    - 不计算/不回传 fake FM loss。
  - fake turn：
    - student forward 放在 `torch.no_grad()` 内，只用于生成当前 fake latent distribution；
    - 用 `student_z_pred.detach()` 训练 `G_fake` 的 FM loss；
    - 只调用普通 `fake_loss.backward()`；
    - 只执行 `_average_stage3c_fake_gradients(fake_model)`、`fake_optimizer.step()` 和 fake scheduler；
    - 不更新 student optimizer，不跑 DMD student loss。
  - wandb 记录用 `runner_step` 作为 log step，另外保留 `train/step` 表示 generator/global step，避免 fake-only turn 多次写同一个 wandb step。
- 与 DMD2 官方的关系：
  - D4.1 比 D3.2/D4 单 backward 更接近 DMD2 官方两个 optimizer 分开更新的组织方式；
  - 但它仍不是官方“同一 iteration 内先 generator 再 guidance”的完全同构实现，而是为了避开当前 DeepSpeed ZeRO2 runner 中混用 student DeepSpeed backward 与手动 fake backward 的稳定性问题，改成跨 runner turn 的物理隔离；
  - 实际效果上可以让 fake 以 5:1 的频率相对 student 加速更新。
- 本地校验：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-1-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-TurnIsolated-Dfake5-OfflineWandb.sh`
- 尚未完成：
  - D4.1 目前只完成本地语法/脚本检查，还没有远端 2GPU smoke；
  - 正式 48 卡启动前仍建议先 smoke 至少跑过 `runner_step=0..6`，确认第二个 generator turn 正常，并确认日志呈现 `generator, fake, fake, fake, fake, fake, generator`；
  - 用户计划 D4 系列正式训练时换新的 Stage1 pretrain，当前 D4.1 launch 默认仍是旧 Stage1 checkpoint，正式启动前必须替换。

## 2026-05-17 Stage3 v7-D4.2 pretrain 固定与 loss 验证计划中文化
- 用户新增要求：
  - `v7-D4.2` 以及后续 Stage3 验证/训练默认使用新的 Stage1 USMGT Takano 微调模型作为 `G_real/G_fake` 与 Stage1 teacher 初始化来源；
  - 指定 ckpt：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
  - 该文件在 BFS 机器上；其他机器如缺文件，默认从：
    - `s3://lxh/tmp/usmgt_stage1_takano20250205_step3000_20260517/step-3000.safetensors`
    拉取。
- 已新增 `v7-D4.2` 独立 release 线，不覆盖 `v7-D4.1` 稳定线：
  - 配置：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_turnisolated_dfake5_offlinewandb.yaml`
  - 启动脚本：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-TurnIsolated-Dfake5-OfflineWandb.sh`
  - 训练代码仍复用稳定：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - `OUTPUT_TAG` 改为：
    - `train_stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_turnisolated_dfake5_offlinewandb`
  - wandb name 改为：
    - `flashvsr-stage3-48gpu-v7-d4-2-usmgtpretrain-turnisolated-dfake5-offline`
- 已更新 GradCheck 默认 Stage1 路径：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-1.sh`
  - 后续 Grad-A 到 Grad-E 重新跑时会默认使用 USMGT `step-3000`，不再使用旧 48GPU Stage1 `step-10000`。
- 已重写中文验证计划：
  - `doc/flashvsr_stage3_loss_validation_plan_20260517.md`
  - 内容包括：
    - D4.1 / D4.2 的 turn-isolated 训练语义；
    - Grad-A 到 Grad-E 的参数归属验证；
    - Loss-1 到 Loss-7 的视觉 ablation；
    - Ghost / 残影专项计划，明确如何区分残影来自 DMD/Fake、pixel/LPIPS、one-step student、attention/mask 或 inference 路径。
- 关键工程约束：
  - 稳定训练源码 `train_flashvsr_stage3_v7_d4_1_lora.py` 不再塞临时 debug；
  - GradCheck 使用独立文件：
    - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_gradcheck_lora.py`
  - 后续验证如果需要更多 probe，也继续复制独立 `*_probe_*.py`，不要直接修改稳定训练源码。

## 2026-05-17 Stage3 ghost probe 第一轮
- 用户要求：
  - 16 卡机器先恢复占卡；
  - 用 23 卡探索之前 Stage3 v7.4/D 系列输出 ghost/残影的原因；
  - 从已有 1000 代 ckpt 开始做验证。
- 资源处理：
  - `bfs6vaz4d6` 与 `i6hf4scd4y` 均恢复 `gpu_stress_tc.sh`，确认 8 卡均约 `166604 MiB`、util 约 `100%`。
  - `6ikhpjzv3z` 临时释放 GPU 2/3 跑 ghost probe，测试完成后恢复 `lxh:occupy23`。
- 可用 ckpt 检查：
  - 在 `6ikhpjzv3z` 和 S3 上没有找到正式 D4/D4.1 的 1000 代 ckpt；
  - 当前可用的 ghost 相关 Stage3 ckpt 是：
    - `/mnt/task_wrapper/user_output/artifacts/ckpts/stage3_v7_d3_2_100plus_20260516/step-1000.safetensors`
  - Stage2 baseline 使用：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- 输出：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_ghost_probe_20260517/takano04`
  - S3：`s3://lxh/artifacts/inference/stage3_ghost_probe_20260517/takano04`
  - 本地：`/Users/lixiaohui/Desktop/stage3_ghost_probe_20260517/takano04`
  - 本地附带拆帧与 HTML 对比页：
    - `/Users/lixiaohui/Desktop/stage3_ghost_probe_20260517/takano04/boundary_contact_sheet.html`
    - `/Users/lixiaohui/Desktop/stage3_ghost_probe_20260517/takano04/ghost_probe_metrics.json`
- 对比输出：
  - `stage2_v641_step6000_sparse_adain.mp4`
  - `stage3_d32_step1000_sparse_adain.mp4`
  - `stage3_d32_step1000_sparse_nocolor.mp4`
  - `stage3_d32_step1000_dense_adain.mp4`
  - `stage3_d32_step1000_officialmask_adain.mp4`
- 结论：
  - `stage3_d32_step1000_sparse_adain.mp4`、`stage3_d32_step1000_dense_adain.mp4`、`stage3_d32_step1000_officialmask_adain.mp4` 的 mp4 hash 完全一致，关键帧 hash 也一致；
  - 这是一个无效的 dense/officialmask 对照：`infer_from_lq_streaming()` 固定走 `stage2_streaming_block_forward()`，该函数内部固定使用 streaming block sparse cache，不读取 `stage2_attention_mode`；
  - `sparse_nocolor` 与 `sparse_adain` 相邻帧指标接近，初步排除 color fix 是主要 ghost 来源；
  - 最大相邻帧变化集中在 `21->22`、`29->30`、`53->54` 等位置，和 chunk 边界跳变现象一致；
  - 目前只能确认“ghost 与 Stage3 输出/权重相关，且不是 color fix 主导”，还不能确认是 block sparse/mask 导致。
- 后续：
  - 若继续查 attention/mask，需要写真正可切换 streaming attention 的 probe，或写 full-DiT no-cache dense probe；
  - 更优先按 `doc/flashvsr_stage3_loss_validation_plan_20260517.md` 跑 loss ablation，先确认 ghost 来自 DMD/Fake、Pixel/LPIPS、one-step student，还是 inference 路径。

## 2026-05-17 Stage3 v7-D4.1 fake 同步从 DDP hook 改为显式 grad all_reduce
- 背景：
  - 用户要求 D4.1 不要停留在“简单版”，而要尽量靠近 DMD2 官方 runner 的两段式更新；
  - 远端 smoke 已证明 `accelerator.backward(fake_loss)` 在当前 FlashVSR Stage3 架构中不可用：student 被 DeepSpeed ZeRO2 管理，而 `G_fake` 是独立模型，fake loss 经 `accelerator.backward()` 会被错误路由进 student ZeRO2 engine；
  - DDP 包装 `G_fake` 虽然能让 fake phase 语义隔离，但在 89f 真模型 smoke 中反复卡在 `WorkNCCL(... ALLREDUCE, NumelIn=287838720, Timeout(ms)=600000)`，`dist.new_group(timeout=7200)` 和 fake backward 前 `dist.barrier()` 都没有可靠解除这个 watchdog。
- 代码修正：
  - 文件：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
  - `G_fake` 不再包 PyTorch DDP；
  - 初始化后对 `G_fake` 参数和 buffer 从 rank0 做一次 `dist.broadcast()`，保证多 rank 初值一致；
  - fake phase 保持独立 `fake_loss.backward()`；
  - autograd 完成后调用 `_average_stage3c_fake_gradients(fake_model)`，逐参数 `dist.all_reduce(SUM) / world_size`，再执行 `fake_optimizer.step()`；
  - 日志新增 `fake_manual_grad_sync=1`，同时 `fake_ddp=0`，用于区分该版本不是 DDP hook 同步。
- 语义结论：
  - FlashVSR 论文把多个 loss 写在一个 objective 里，不等于工程上必须把所有 loss 合成一次 backward；
  - DMD2 官方 `main/train_sd.py` 的 runner 是：可选 generator phase，然后每步 guidance/fake phase；
  - D4.1 当前实现也按这个顺序组织，只是因为 FlashVSR student 走 DeepSpeed ZeRO2，fake phase 不能用同一个 `accelerator.backward()`，所以用独立 backward + 显式梯度同步来实现同样的 optimizer ownership。
- 远端处理：
  - 旧 DDP/barrier smoke：`20260517_d41_smoke_g01_bar09`
  - `bar09` 在 `runner_step=4` 后仍长时间卡住，GPU0/1 表现为单 rank 长时间 100% / 0%，判定 DDP hook 路径不稳；
  - 已安全停止：先向 `stage3_smoke:d41bar_09` 发送 Ctrl-C，未响应后仅对只读确认出的精确 PID `1153163 1153373 1153374 1153151 1153154` 发送 `kill -TERM`，没有使用 `pkill` 或模糊匹配；
  - GPU0/1 已释放，GPU2/3 占卡程序保持不动。
- 新 smoke：
  - 机器：`6ikhpjzv3z`
  - tmux：`stage3_smoke:d41bar_10`
  - run tag：`20260517_d41_smoke_g01_bar10`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_bar10`
  - 目标：确认日志打印 `fake_ddp=0 fake_manual_grad_sync=1`，并至少跑过 `runner_step=0..5`，验证第二个 generator turn。
- `bar10` 结果：
  - 确认日志：`fake_ddp=0 fake_manual_grad_sync=1`；
  - runner 0-4 均通过，runner 0 为 `generator+fake`，runner 1-4 为 fake-only；
  - 失败仍为 NCCL watchdog：`WorkNCCL(... OpType=ALLREDUCE, NumelIn=287838720, Timeout(ms)=600000)`；
  - 这说明 DDP hook 不是唯一问题，手写 `_average_stage3c_fake_gradients()` 中对单个超大 fake 参数直接做整块 NCCL all_reduce 也会产生同样的 287M collective；
  - `fake_backward_optim=13.581s` 的日志不能完全代表 collective 已经安全结束，后续 watchdog 仍可在 NCCL work 上报错。
- 新代码修正：
  - `_average_stage3c_fake_gradients()` 改为分块同步；
  - 默认 `FLASHVSR_STAGE3_FAKE_GRAD_SYNC_CHUNK_NUMEL=4194304`，即每个 grad chunk 最多约 4M 元素；
  - 对每个 chunk 做 `dist.all_reduce(SUM)` 后除以 world size；
  - fake grad sync 后显式 `torch.cuda.synchronize()`，让 timing 和错误暴露点更接近真实同步耗时。
- `bar10` 清理：
  - 先对精确 PID `1260242 1260452 1260453 1260230 1260233` 发 `kill -TERM`；
  - 仍残留后对精确 PID `1260242 1260452 1260453` 发 `kill -KILL`；
  - 没有使用模糊 kill；
  - GPU0/1 已释放，GPU2/3 占卡保持不动。
- 新 smoke：
  - tmux：`stage3_smoke:d41bar_11`
  - run tag：`20260517_d41_smoke_g01_bar11`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_bar11`
  - 启动环境额外设置：`FLASHVSR_STAGE3_FAKE_GRAD_SYNC_CHUNK_NUMEL=4194304`
  - 目标：确认分块 fake grad sync 不再出现 287M 单 collective，并跑过 `runner_step=5`。
- `bar11` 结果：
  - runner 0-4 均通过；
  - fake grad sync 分块后没有再把 fake 路径暴露为 DDP hook 问题；
  - 但 runner 5 的 student/generator backward 失败，栈明确在：
    - `accelerator.backward(total_loss)`
    - DeepSpeed ZeRO2 `allreduce_bucket`
    - `torch.distributed.all_reduce`
  - 报错仍是 `WorkNCCL(... NumelIn=287838720, Timeout(ms)=600000)`，但这次属于 student DeepSpeed ZeRO2 gradient allreduce，不是 fake optimizer；
  - 结论：D4.1 两段式 loss/optimizer ownership 已经清楚，剩余 smoke 阻塞是 rank skew 进入 DeepSpeed ZeRO2 allreduce 前没有对齐。
- 新代码修正：
  - 在 generator/student phase 的 `accelerator.backward(total_loss)` 前增加 `dist.barrier()`；
  - 目的：先用默认主 process group 等慢 rank 对齐，再进入 DeepSpeed ZeRO2 的短 watchdog NCCL allreduce；
  - fake phase 仍保留分块 `_average_stage3c_fake_gradients()`。
- `bar11` 清理：
  - TERM 精确 PID：`1366294 1366481 1366482 1366282 1366285`
  - KILL 精确 PID：`1366294 1366481 1366482`
  - GPU0/1 已释放，GPU2/3 占卡保持。
- `bar12` 结果：
  - 使用 generator backward 前 barrier + fake grad 分块同步；
  - runner 0-4 正常通过；
  - runner 4 后长时间没有进入 runner 5，也没有触发 watchdog；
  - 进程保持 `R` 状态，GPU0 长时间 100%、GPU1 0%，判断为 barrier/上游 rank skew hang，而不是原先的 DeepSpeed allreduce 600s timeout；
  - 已安全停止：TERM 精确 PID `1467792 1467983 1467984 1467780 1467783`，GPU0/1 已释放。
- 新 debug 补丁：
  - 新增默认关闭的 `FLASHVSR_STAGE3_SYNC_DEBUG`；
  - 打印每个 rank 的 `before_next_data`、`after_next_data`、`enter_turn`、`before_generator_barrier`、`after_generator_barrier`；
  - 用于确认 rank 是否在 data、turn 入口或 generator barrier 前分叉/长尾。
- 新 smoke：
  - tmux：`stage3_smoke:d41bar_13`
  - run tag：`20260517_d41_smoke_g01_bar13_syncdebug`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_bar13_syncdebug`
  - 启动环境：`FLASHVSR_STAGE3_SYNC_DEBUG=1`、`FLASHVSR_STAGE3_FAKE_GRAD_SYNC_CHUNK_NUMEL=4194304`
  - 早期结果：
    - runner 0：rank1 先到 `before_generator_barrier`，rank0 后到；两 rank 均打印 `after_generator_barrier`，说明 runner 0 的 generator barrier 本身可以通过；
    - runner 1：两 rank 均完成 fake-only turn；
    - 这证明 barrier 不是立即死锁，rank skew 的关键点更可能发生在后续 runner 4 -> runner 5 的数据/前向长尾。

## 2026-05-17 Stage3 v7-D4.1 DMD2-style 两段更新修正
- 用户追问 FlashVSR 论文里的总 loss 公式到底是否意味着“所有 loss 放一起一次 backward”，以及 D4.1 是否仍是简单版。
- 当前判断：
  - FlashVSR 论文公式是 objective-level 写法，把 reconstruction / adversarial-DMD / fake critic 相关项写在同一训练目标下，并不等价于工程上必须一次 `total_loss.backward()`；
  - 论文没有给出足够细的 optimizer step 伪代码，因此具体 runner 调度应参考 DMD2 官方实现；
  - DMD2 官方 `main/train_sd.py` 是两段式：当 `step % dfake_gen_update_ratio == 0` 时先 generator phase `accelerator.backward(generator_loss)` + generator optimizer step，然后每个 runner step 都 guidance/fake phase `accelerator.backward(guidance_loss)` + guidance optimizer step。
- 已修正 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`：
  - student/generator phase：只组合 `student_loss + dmd_student_loss`，调用 `accelerator.backward(total_loss)`，只 step student optimizer；
  - fake/guidance phase：随后单独计算 fake FM loss，使用 detached `student_z_pred`，调用 `accelerator.backward(fake_loss)`，只 step fake optimizer；
  - `G_fake` 不再尝试放进同一个 DeepSpeed `Accelerator.prepare`，因为 Accelerate 1.12 + DeepSpeed 不允许同一 `Accelerator()` 实例 prepare 多个模型；当前做法是 student 走 DeepSpeed，`G_fake` 走 PyTorch DDP，同步由 DDP hook 完成；
  - 增加 `FLASHVSR_DIST_TIMEOUT_SECONDS`，默认 7200 秒，通过 `accelerate.InitProcessGroupKwargs(timeout=...)` 放宽 Stage3 真实 89f smoke 中长 data/offload 导致的 rank skew。
- 需要纠正旧文档里的 runner 序列：
  - `stage3_dfake_gen_update_ratio=5` 时，DMD2 官方语义是：
    - `runner_step=0`: generator+fake
    - `runner_step=1..4`: fake-only
    - `runner_step=5`: generator+fake
  - 不是 `0` 后等到 `6` 才下一个 generator。
- 本地校验：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py`
- 远端 smoke：
  - 机器：`6ikhpjzv3z`
  - tmux：`stage3_smoke:d41accel_06`
  - run tag：`20260517_d41_smoke_g01_accel06`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_1_turnisolated_dfake5_20260517_d41_smoke_g01_accel06`
  - 启动环境确认：`dfake_gen_update_ratio=5`、`fake_ddp=1`、`distributed_timeout_seconds=7200`；
  - 结果：失败在第一个 runner 的 fake phase；
  - 错误：`accelerator.backward(fake_loss)` 被 Accelerate 路由进 student 的 DeepSpeed ZeRO2 engine，ZeRO2 尝试 reduce 不属于 fake loss 图的 student optimizer bucket，触发 `IndexError: list index out of range`；
  - 结论：在当前架构里，`G_fake` 不能使用同一个 student DeepSpeed engine 的 `accelerator.backward()`；正确隔离方式是 student phase 继续 `accelerator.backward(total_loss)`，fake phase 使用 DDP 模型自己的 `fake_loss.backward()`，由 DDP hook 同步 fake 梯度；
  - 已把 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_1_lora.py` 改回 fake phase 的 DDP backward，并在代码中加注释解释原因。
- 后续 smoke `20260517_d41_smoke_g01_ddp07`：
  - runner 0 通过：`turn=generator+fake`，`loss=1.038355`，`fake_loss=0.04628955`；
  - runner 1-4 通过：均为 fake-only，loss 分别约 `0.037424 / 0.013124 / 0.006923 / 0.057620`；
  - 说明两段式调度、fake DDP backward、fake-only turn 都能跑；
  - 失败点：到 runner 5 前，fake DDP 的 process group 仍使用 PyTorch 默认 600s timeout，`WorkNCCL(... OpType=ALLREDUCE, NumelIn=287838720 ... Timeout(ms)=600000)` 超时；
  - 已修正：`_stage3d41_wrap_fake_ddp_if_needed()` 现在用 `dist.new_group(... timeout=timedelta(seconds=FLASHVSR_DIST_TIMEOUT_SECONDS))` 为 fake DDP 显式创建 7200s process group；
  - 新 smoke：`20260517_d41_smoke_g01_pg08`，用于验证 fake DDP process group timeout 后是否能跑到 runner 5 第二个 generator turn。
- `20260517_d41_smoke_g01_pg08` 结果：
  - runner 0-4 仍然正常通过；
  - 但 fake DDP allreduce 仍显示 `Timeout(ms)=600000`，说明 `dist.new_group(timeout=...)` 没有改变当前 PyTorch/NCCL split group 的 watchdog；
  - 已追加更直接的修法：fake phase 在 `fake_loss.backward()` 前，如果 `G_fake` 是 DDP，则先用默认主 process group 做 `dist.barrier()`，把 rank 对齐后再进入 fake DDP 的大梯度 allreduce；
  - 目的：避免一个 rank 因数据/decode/offload 长尾晚到，另一个 rank 先进入 `NumelIn=287838720` 的 fake gradient allreduce 并空等到 600s watchdog。

## 2026-05-18 Stage3 v7-D3.2 step-1500 正确三阶段测试
- 用户要求先测当前 D3.2 训练出的 step-1500，再继续写 D4.2。
- checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2/output/step-1500.safetensors`
  - 临时上传到 `s3://lxh/tmp/stage3_v7_d3_2_step1500_eval_20260518/step-1500.safetensors`
- 测试机器：`6ikhpjzv3z`，GPU0/1 分 5+5 跑 10 个合成测试视频。
- 使用正确 Stage3 一步 streaming/KV-cache batch inference。
- 输出：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7_d3_2_step1500_synthetic_20260518/step-1500`
  - S3：`s3://lxh/data/test/stage3_v7_d3_2_step1500_synthetic_20260518`
  - 本机桌面：`/Users/lixiaohui/Desktop/stage3_v7_d3_2_step1500_synthetic_20260518`
- 结果：10/10 mp4 输出完成，`timing_summary.txt` 记录 `status_gpu0=0`、`status_gpu1=0`、`seconds=105`、`outputs=10`。

## 2026-05-18 Stage3 v7-D4.2 single-runner dfake=5 实现与 smoke
- 用户要求写 D4.2：不再修 D4.1 的彻底 turn-isolated 版本，而是在单 runner 内做到更接近 DMD2 的梯度归属：
  - fake critic 每个 runner step 更新；
  - student/generator 只在 `runner_step % stage3_dfake_gen_update_ratio == 0` 更新；
  - fake loss 使用当前 `student_z_pred.detach()`；
  - student backward 图只包含 `student_loss + dmd_student_loss`；
  - fake backward 图只包含 fake FM loss，随后显式同步 fake grads 并只 step fake optimizer。
- 新文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_2_lora_89f_videoonly_usmgtpretrain_singlerunner_dfake5_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-2-Lora-89f-VideoOnly-USMGTPretrain-SingleRunner-Dfake5-OfflineWandb.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-2-SingleRunner-Dfake5.sh`
- Stage1 teacher/fake 初始化改用用户指定的新 USMGT step-3000：
  - 实际远端路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
  - S3 fallback：`s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors`
- 本地与远端校验：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
  - `bash -n` 两个 D4.2 launch script
  - 远端同步后重复上述检查通过。
- 第一次 smoke `20260518_d42_smoke_g01b`：
  - 机器：`6ikhpjzv3z`，tmux：`stage3_smoke:d42_g01`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5_20260518_d42_smoke_g01b`
  - runner 0-4 已通过，但 smoke config 仍是 `yubari_video_prob=0.5 / takano_video_prob=0.5`，`dataset_num_workers=0` 时抽到远端 conductor 样本会造成 `next(data)` 极慢；
  - 已用 tmux Ctrl-C 停止该 smoke，未使用 `pkill` 或模糊 kill。
- 已把 D4.2 smoke config 改为只走本地 Takano manifest：
  - `yubari_video_prob: 0.0`
  - `takano_video_prob: 1.0`
  - 正式 48 卡 release config 未改，仍保留训练数据比例。
- 第二次 smoke `20260518_d42_smoke_g01c_takano`：
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_2_singlerunner_dfake5_20260518_d42_smoke_g01c_takano`
  - 日志确认：`D4.2 single-runner separated optimizer`、`dfake_gen_update_ratio=5`、`fake_update_every_runner_step=1`、`fake_loss_uses_current_z_pred_detach=1`；
  - 日志确认 real/fake checkpoint 均为新的 Stage1 step-3000。
- smoke 结果：
  - runner 0：`generator_update=1`，`loss=1.084645`，`student=1.038355`，`fake_loss=0.04628955`，`dmd_student=0.000000`；
  - runner 1-4：均为 fake-only，fake loss 约 `0.021428 / 0.013311 / 0.006936 / 0.082231`；
  - runner 5：第二次 generator turn 通过，`generator_update=1`，`loss=1.557251`，`student=1.426915`，`fake_loss=0.10539128`，`dmd_student=0.024945`，`dmd_grad=0.171409`，`dmd_skip=0`；
  - 保存：`step-1.safetensors` 和 `step-2.safetensors` 均生成；
  - 进程正常退出，GPU0/1 释放，GPU2/3 占卡保持。
- 结论：
  - D4.2 单 runner 方案已经通过 2GPU smoke 到第二个 generator turn；
  - 该版本避免了 D4.1 turn-isolated/DDP fake 路径的 rank 对齐和 fake 大梯度 collective 问题；
  - 仍需诚实保留：完整 Stage1 teacher deterministic forward 数值等价没有在本轮重新做。

## 2026-05-18 Stage3 v7-D4.2 teacher 前 22 对齐修正与验证计划
- 用户明确指出：Stage3 student 的 22 个 latent 应对齐 Stage1 teacher 的前 22 个，而不是后 22 个。
- 背景：
  - Stage2 v6.4 的目标不是旧的 `GT 89 -> VAE 23 -> drop z0`；
  - v6.4 是取 `GT 前 85 帧 -> WAN VAE -> 22 latents`；
  - 因此 Stage1 `nonstreaming_aligned` teacher 89 帧得到 23 positions 后，Stage3 应保留 teacher `[0,22)` 对齐 student `[0,22)`，丢弃 teacher `[22,23)`。
- 代码修正：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
    - `_stage3d31_teacher_lq_alignment_mode()` 对 `nonstreaming_aligned` 返回 `trim_tail_to_match`。
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py`
    - `flashvsr_stage2_model_fn()` 新增 `trim_tail_to_match`；
    - debug temporal map 会显示 `keep_teacher_positions=[0,22)`、`drop_teacher_positions=[22,23)`。
- 本地校验：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage2_v6_4_lora.py wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
- 新增验证计划文档：
  - `doc/flashvsr_stage3_v7d42_validation_plan_20260518.md`
  - 包含 E0-E7 编号实验、结论页、deterministic forward 等价实验说明、48 卡短训 ablation 计划。
- 当前下一步：
  - 先做 E0：D4.2 前 22 对齐 2GPU smoke，确认 `trim_tail_to_match` 并跑到 runner 5。

## 2026-05-18 Stage3 v7-D4.3 dual DeepSpeed engine 尝试
- 用户要求尝试更干净的 dual-engine 版本：student/generator 一个 DeepSpeed engine，trainable `G_fake` 一个 DeepSpeed engine；两个 optimizer、两个 scheduler、两个 checkpoint/save/load 路径分开。
- 结论先写清楚：
  - D4.1 之前尝试的不是这个方案；D4.1 是 turn-isolated / 双 optimizer 路线，不是真正两个 DeepSpeed engine；
  - D4.3 已实现并通过基础 smoke 到至少 runner 1，证明 `G_fake` 可以被单独 DeepSpeed engine 管；
  - 但是速度目前不理想，不能直接作为 48 卡候选替换 D4.2。
- 新文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_3_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-3-DualEngine-Dfake5.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_3_lora_89f_videoonly_usmgtpretrain_dualengine_dfake5_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-3-Lora-89f-VideoOnly-USMGTPretrain-DualEngine-Dfake5-OfflineWandb.sh`
- 关键实现：
  - student 仍由 `accelerator.prepare(model, optimizer, ...)` 管；
  - `G_fake` 在 student prepare 后用 `deepspeed.initialize(...)` 单独初始化为 fake engine；
  - fake phase 使用 `fake_model.backward(fake_loss)` 和 `fake_model.step()`，不再手写 fake gradient all-reduce；
  - fake DeepSpeed state 单独保存到 `output/stage3_fake_deepspeed/`；
  - `G_fake` 初始化仍使用用户指定 Stage1 USMGT step-3000：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- 第一轮 smoke `20260518_d43_dualengine_e0`：
  - 失败点：DeepSpeed 拒绝 `ZeRO-Offload + client-provided torch.optim.AdamW`；
  - 修正：fake DeepSpeed config 加 `zero_force_ds_cpu_optimizer=False`。
- 第二轮 smoke `20260518_d43_dualengine_e1`：
  - dual-engine 初始化成功，`fake_ds_zero_stage=2`；
  - 但发现 `fake_loss=0 / fake_update=0`；
  - 原因：`_stage3c_fake_fm_loss()` 仍直接判断 `isinstance(fake_model, FlashVSRStage3BTrainingModule)`，包成 DeepSpeedEngine 后该判断为 false；
  - 修正：先 `_unwrap_stage3c_model(fake_model)`，再判断底层 module 类型。
- 第三轮 smoke `20260518_d43_dualengine_e2`：
  - 机器：`6ikhpjzv3z`，GPU0/1；
  - 日志确认：`D4.3 dual DeepSpeed engine`、`fake_backward=deepspeed_engine`、`fake_ds_zero_stage=2`；
  - teacher 对齐确认：`keep_teacher_positions=[0,22)`、`drop_teacher_positions=[22,23)`；
  - runner 0：
    - `generator_update=1`
    - `loss=1.069140`
    - `student=1.038355`
    - `fake_loss=0.03078421`
    - `fake_update=1`
    - `real_probe=0.115747`
    - `fake_probe=0.149991`
    - `dmd_student=0.000000`
  - timing：
    - `data=123.142s`
    - `student=9.083s`
    - `probe=17.542s`
    - `dmd=17.591s`
    - `fake=8.778s`
    - `fake_backward_sync=175.037s`
    - `student_backward=7.823s`
    - `optim=2.943s`
    - `save_sched=20.438s`
  - runner 1 fake-only：
    - `generator_update=0`
    - `fake_loss=0.00752554`
    - `fake_update=1`
    - timing：`fake=8.750s`、`fake_backward_sync=46.805s`、`optim=2.293s`
  - 已用 tmux Ctrl-C 停止 smoke，未使用模糊 `pkill`；GPU0/1 已释放。
- 当前判断：
  - D4.3 的数学/工程结构比 D4.2 更“干净”：`G_fake` 真正由 DeepSpeed 管，fake loss 通过 fake engine backward/step；
  - 但 2GPU smoke 显示 fake backward 仍是主要耗时，首个 generator turn 约 175s，fake-only turn 约 47s；
  - 因此 D4.3 现在是可继续优化的验证线，不建议直接上 48 卡长训；
  - D4.2 仍是当前更稳的候选，D4.3 需要继续查 fake engine ZeRO-Offload/optimizer/scheduler 配置和 rank 等待成本。

## 2026-05-18 Stage3 v7-D4.3 e2/e3 与双 DeepSpeed 参考复查
- 用户询问 e2/e3 是什么，以及是否能找一个双 DeepSpeed model 参考。
- 先澄清命名：
  - `e2` / `e3` 只是 D4.3 的 smoke 实验后缀，不是代码版本名；
  - D4.3 的代码版本仍是 `train_flashvsr_stage3_v7_d4_3_lora.py`。
- `e2`：
  - run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5_20260518_d43_dualengine_e2`
  - fake DeepSpeed 复用原 ZeRO2 config，包含 CPU/param offload；
  - 结果证明 fake engine 真实更新：runner 0 `fake_loss=0.03078421`、`fake_update=1`；
  - 但 runner 0 `fake_backward_sync=175.037s`，runner 1 fake-only `fake_backward_sync=46.805s`。
- `e3`：
  - run：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_3_dualengine_dfake5_20260518_d43_dualengine_nooffload_e3`
  - 修改：fake DeepSpeed 默认不再启用 CPU/param offload，日志标记 `fake_ds_offload=0`；
  - 结果：runner 0 仍为 `fake_loss=0.03078421`、`fake_update=1`；
  - timing：`fake_backward_sync=174.472s`、`student_backward=7.754s`、`optim=0.034s`、`save_sched=23.522s`；
  - 结论：关 fake offload 明显降低 optimizer step 时间，但没有解决核心的 fake backward/sync 长耗时。
- 已用 tmux Ctrl-C 安全停止 e3；未使用 `pkill`、未使用模糊 kill；停止后 GPU0/1 释放，GPU2/3 保持原占用。
- 官方参考复查：
  - DeepSpeed 官方支持 multiple models 的核心形式是为每个模型创建独立 `DeepSpeedEngine`，分别调用 `engine.backward(loss)` / `engine.step()`；
  - HuggingFace Accelerate 官方对“多个 disjoint DeepSpeed 模型”的建议是多个 `DeepSpeedPlugin`，且由于一个 `Accelerator` 同时只能携带一个 engine/plugin，训练多个 disjoint 模型时需要第二个 `Accelerator`；
  - 本地 DMD2 官方代码不是双 DeepSpeed engine。它是 `feedforward_model` / `guidance_model` 两个子模型，两套 optimizer/scheduler，经 Accelerate/DDP/FSDP 管理。
- DMD2 官方代码对应点：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py`
    - `optimizer_generator` / `optimizer_guidance` 两套 optimizer；
    - `COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0`；
    - generator turn 先更新 generator，再更新 guidance/fake；
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_unified_model.py`
    - generator turn 会临时 `guidance_model.requires_grad_(False)`；
    - guidance/fake 使用 `generated_image.detach()`，不把 guidance loss 回传到 generator。
- 当前判断：
  - D4.3 direct `deepspeed.initialize()` 路线不是凭空写的，符合 DeepSpeed 官方 multiple-engine API；
  - 但它不是从 DMD2 原样抄来的，因为 DMD2 本身没有 DeepSpeed 双 engine；
  - 如果继续追求更官方的 Accelerate 写法，下一条线应考虑“两个 `DeepSpeedPlugin` + 两个 `Accelerator`”的 D4.4 风格实验，但这是比 D4.3 更大的重构；
  - 目前 e2/e3 说明 D4.3 的慢点不在 fake optimizer offload，主要仍在 fake full-attention backward/sync 或 rank collective 等待。

## 2026-05-18 Stage3 v7-D4.4 dual Accelerator + dual DeepSpeedPlugin 实现与 smoke
- 用户要求不要停在 D4.3 raw DeepSpeed engine，继续按 Accelerate 官方 multiple DeepSpeed models 方式写 D4.4。
- 参考依据：
  - HuggingFace Accelerate 官方 multiple DeepSpeed models 文档说明：训练多个 disjoint models 时需要两个 `DeepSpeedPlugin`，并需要第二个 `Accelerator`，因为一个 `Accelerator` 同时只能携带一个 engine/plugin；
  - DeepSpeed 官方 training API 支持 multiple models/engines，D4.3 属于该路线；
  - 本地 DMD2 官方代码本身不是 DeepSpeed 双 engine，而是 Accelerate/DDP/FSDP 双子模型、双 optimizer/scheduler；D4.4 只借鉴 DMD2 的更新语义，不可能直接照抄 DMD2 的 DeepSpeed 写法。
- 新文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- 关键实现：
  - `main()` 构造两个 `DeepSpeedPlugin`：`student` 和 `fake`；
  - `accelerator = Accelerator(..., deepspeed_plugins=plugins)` 后选择 `student`；
  - `fake_accelerator = Accelerator()` 后选择 `fake`；
  - student 仍走 `accelerator.prepare(model, optimizer, scheduler/dataloader)`；
  - fake 走 `fake_accelerator.prepare(fake_model, fake_optimizer, fake_scheduler)`；
  - fake phase 改成 `fake_accelerator.backward(fake_loss)`，随后只 step fake optimizer/scheduler；
  - fake DeepSpeed checkpoint 继续单独写到 `output/stage3_fake_deepspeed/`；
  - fake DeepSpeed 默认关闭 CPU/param offload，日志显示 `fake_ds_offload=0`。
- 本地检查：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
  - `bash -n` 两个 D4.4 launch script
- 远端环境检查：
  - 机器：`6ikhpjzv3z`
  - `/mnt/conda_envs/flashvsr/bin/python` 确认 `accelerate==1.12.0`
  - 确认 `Accelerator` 支持 `deepspeed_plugins` 参数；
  - 确认 `AcceleratorState.select_deepspeed_plugin` 存在；
  - 远端 `py_compile` 与 smoke 脚本 `bash -n` 通过。
- smoke：
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_dualaccel_e0`
  - GPU：`6ikhpjzv3z` GPU0/1
  - 日志确认：`D4.4 dual Accelerate DeepSpeed engine`、`fake_backward=fake_accelerator_deepspeed`、`fake_ds_zero_stage=2`、`fake_ds_offload=0`
  - runner 0：
    - `generator_update=1`
    - `loss=1.069140`
    - `student=1.038355`
    - `fake_loss=0.03078421`
    - `fake_update=1`
    - `real_probe=0.115747`
    - `fake_probe=0.149991`
    - `dmd_student=0.000000`
    - timing：`data=127.236s`、`student=9.485s`、`probe=18.013s`、`dmd=18.062s`、`fake=9.017s`、`fake_backward_sync=172.442s`、`student_backward=9.438s`、`optim=0.000s`、`save_sched=22.881s`
  - runner 1 fake-only：
    - `generator_update=0`
    - `loss=0.007519`
    - `student=1.144972`
    - `fake_loss=0.00751893`
    - `fake_update=1`
    - timing：`data=57.597s`、`student=7.800s`、`fake=9.002s`、`fake_backward_sync=46.520s`、`optim=0.004s`
  - 已用 tmux Ctrl-C 停止 smoke，未使用 `pkill`；停止后 GPU0/1 释放，GPU2/3 保持原占用。
- 结论：
  - D4.4 完成了“两个 Accelerator + 两个 DeepSpeedPlugin”的官方 Accelerate 形态；
  - 数值/loss 与 D4.3 基本一致，说明 fake loss、detach、optimizer ownership 没有被新包装破坏；
  - 但性能没有改善：runner 0 fake backward 仍约 172s，runner 1 fake-only 仍约 46s，与 D4.3 同量级；
  - 因此瓶颈不是 raw DeepSpeed vs Accelerate wrapper，也不是 fake optimizer offload，而更可能是 trainable full-attention `G_fake` 的大 backward/ZeRO2 gradient sync 本身，或 Stage3 每步重复跑 heavy fake FM 的结构成本。

## 2026-05-18 Stage3 v7-D4.4 fake 参数范围与 flash-attn 复查
- 用户怀疑 D4.4 fake backward 过慢可能是把 full WAN body 也放进了 ZeRO sync，或 fake full-attention 没有走 flash-attn。
- 本地修改：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
    - fake DeepSpeed 默认 config 改为 Stage1 v5.3.5 同款 `deepspeed_zero2_flashvsr_nooffload.json`，仍可用 `FLASHVSR_STAGE3_DS_CONFIG` 覆盖；
    - 增加 `_summarize_trainable_param_groups()`，在 runner 启动日志打印 fake 可训练参数分组，显式暴露是否有 `dit_base_unexpected`。
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Smoke-2GPU-v7-D4-4-DualAccelerator-Dfake5.sh`
    - 增加 `FLASHVSR_STAGE3_DS_CONFIG`、`FLASHVSR_TRAIN_DEBUG`、`FLASHVSR_DEBUG_DIR` 输出，方便确认 DeepSpeed/flash-attn 分支。
- 检查：
  - 本地 `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py` 通过；
  - 本地/远端 smoke 脚本 `bash -n` 通过；
  - 远端 `/mnt/task_runtime/lucidvsr` 已同步到包含 `fake_trainable_groups` 与 no-offload 默认 config 的版本。
- smoke：
  - 机器：`6ikhpjzv3z` GPU0/1
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_nooffload_actckpt_e1`
  - 启动时确认：
    - `stage3_ds_config=/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload.json`
    - student/fake DeepSpeed 都是 `zero_stage=2`、`offload_optimizer_device=None`、`offload_param_device=None`、`has_activation_checkpointing=True`
  - fake 可训练参数：
    - `fake_trainable_params=570961408`
    - `fake_trainable_groups={"lora": 283115520, "lq_proj_in": 287845888}`
    - 没有 `dit_base_unexpected`，说明 full WAN/DiT base body 没有被打开训练。
  - flash-attn：
    - debug 记录 `branch=flash_attn_2`
    - shape 为 `(1, 84480, 12, 128)`，dtype 为 `torch.bfloat16`
    - 说明 fake dense_full attention 不是 SDPA fallback。
  - timing：
    - runner 0：`fake_backward_sync=172.897s`、`student_backward=6.129s`
    - runner 1 fake-only：`fake_backward_sync=45.945s`
- 结论：
  - 当前 D4.4 fake 不存在“误把整个 WAN body 放进 optimizer/ZeRO sync”的问题；
  - fake 也确实走了 flash-attn 2；
  - 使用 Stage1 同款 no-offload + activation checkpointing 后，fake backward/sync 仍与 e0 同量级，核心慢点没有改善；
  - 下一步若继续定位性能，应做同机同 2GPU 的 Stage1 v5.3.5 backward timing 对照，或进一步拆 fake backward 内部 profile/ZeRO2 gradient reduction，而不是再围绕“是否全模型训练/是否缺 flash-attn”排查。
- 清理：
  - D4.4 e1 smoke 已安全停止；
  - 只在明确确认目标 smoke rank 残留后处理了精确 PID，未使用 `pkill` 或模糊 kill；
  - GPU0/1 已恢复占卡程序，GPU2/3 原占用保持不动。

## 2026-05-18 Stage3 v7-D4.4 48 卡 fresh1 启动
- 用户要求先把正在跑的 48 卡旧实验停掉，换成 D4.4 48 卡正式训练，先让早上有结果。
- 旧实验：
  - `train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
  - 主机 session：`v7d32_48gpu`
  - 停止方式：
    - 先在主机 `t5qdtykjsw` 的 tmux session 发 Ctrl-C；
    - 从节点未及时退出时，先用完整 run tag 做只读进程确认，再只处理确认属于该 run 的进程；
    - 未使用 `pkill`，未使用空变量/模糊 kill。
- D4.4 release 脚本修正：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
  - 修正 wandb package tmux 名：`wandb_package_v7d44_dfake5`
  - 显式打印 `stage3_ds_config`
  - 说明文字修正为 `dual_accelerator_deepspeed` 和 `teacher_lq_trim_tail_to_match`
  - 默认 `FLASHVSR_STAGE3_DS_CONFIG` 指向 Stage1 同款 no-offload DeepSpeed config。
- 新 48 卡 run：
  - run name：`train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - tmux：6 节点均为 `v7d44_48gpu`
  - master：`t5qdtykjsw` / `240.12.138.137:29547`
  - stage2 checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
  - stage1 real/fake checkpoint：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
  - wandb offline S3：`s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1.tar.gz`
  - 初始 wandb package tmux 过早退出，原因是 loop 启动时训练进程尚未出现，误判 training process not found；
  - 已手动重启 `wandb_package_v7d44_dfake5`，interval 改为 900s；
  - S3 当前已有更新包，大小约 `76020` bytes，后续会继续覆盖刷新。
- 启动结果：
  - 日志确认 `D4.4 dual Accelerate DeepSpeed engine`
  - `fake_trainable_params=570961408`
  - `fake_trainable_groups={"lora": 283115520, "lq_proj_in": 287845888}`
  - `fake_ds_zero_stage=2`
  - `fake_ds_offload=0`
  - `step-1.safetensors` 已保存。
- 48 卡 timing：
  - runner 0：
    - `loss=0.857881`
    - `student=0.800874`
    - `fake_loss=0.05700695`
    - `real_probe=0.054306`
    - `fake_probe=0.167886`
    - `fake_backward_sync=68.595s`
    - `student_backward=15.511s`
    - `save_sched=106.844s`
  - runner 1 fake-only：
    - `fake_loss=0.00931268`
    - `fake_backward_sync=19.763s`
  - runner 2 fake-only：
    - `fake_loss=0.01540674`
    - `fake_backward_sync=15.244s`
  - runner 3 fake-only：
    - `fake_loss=0.01425055`
    - `fake_backward_sync=45.479s`
  - runner 4 fake-only：
    - `fake_loss=0.05309864`
    - `fake_backward_sync=55.245s`
- 初步结论：
  - D4.4 48 卡已真正跑起来，不是停在初始化；
  - 48 卡 fake backward/sync 明显低于 2GPU smoke 的 172s，但仍有 15-55s 波动；
  - 首步较慢主要叠加了 probe/DMD 和保存开销。
- 额外验证：
  - 新建文档：`doc/flashvsr_stage3_v7d44_validation_plan_20260518.md`
  - 在 16 卡机器 `bfs6vaz4d6` 上启动 D4.4 2GPU timing/runner smoke：
    - tmux：`d44_validate_2gpu`
    - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_validate_bfs_2gpu`
    - GPU0/1 用于验证；
    - GPU2-7 已挂占卡，另一台 `i6hf4scd4y` 保持占卡。

## 2026-05-18 Stage3 v7-D4.4 48 卡 fresh1 继续观察与 D44 验证刷新
- 48 卡 D4.4 正式 run 继续正常：
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - `step-1.safetensors` 与 `step-2.safetensors` 已保存。
  - 主机 `t5qdtykjsw` 抽查时 8 张 GPU 约 100% 利用率。
  - 从节点 `a9suya6gxe`、`67dxkwcb7m`、`ui9n6p293s`、`g48bd6x4h7`、`gx2intv5rk` 均抽查到 8 张 GPU 高显存、高利用率，说明 48 卡不是只在主机跑。
- 新增 48 卡 timing：
  - runner 5，即第二个 generator turn：
    - `loss=1.197141`
    - `student=0.923289`
    - `fake_loss=0.03292919`
    - `dmd_student=0.240923`
    - `dmd_grad=0.531402`
    - `fake_backward_sync=63.379s`
    - `student_backward=15.372s`
    - `save_sched=105.583s`
  - runner 6-8 fake-only：
    - runner 6：`fake_loss=0.07938955`、`fake_backward_sync=16.102s`
    - runner 7：`fake_loss=0.09516689`、`fake_backward_sync=21.146s`
    - runner 8：`fake_loss=0.02226195`、`fake_backward_sync=34.984s`
- wandb：
  - `wandb_package_v7d44_dfake5` tmux 仍存在。
  - S3 包已更新：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1.tar.gz`
    - 当前大小约 `76020` bytes。
  - 注意：打包目录里还镜像了一个旧 transient offline run，后续如果用户要导入 wandb 时可清理；当前不阻塞自动 offload。
- D44 验证文档已刷新：
  - `doc/flashvsr_stage3_v7d44_validation_plan_20260518.md`
  - 已记录 D44-0/D44-1 的当前结论，并保留 D44-2 到 D44-6 的后续验证计划。
- 16 卡验证：
  - `bfs6vaz4d6` GPU0/1 正在跑 D4.4 2GPU timing/runner smoke：
    - tmux：`d44_validate_2gpu`
    - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5_20260518_d44_validate_bfs_2gpu`
  - 已完成 runner 0-2：
    - runner 0：`fake_backward_sync=173.689s`、`student_backward=6.132s`
    - runner 1 fake-only：`fake_backward_sync=45.155s`
    - runner 2 fake-only：`fake_backward_sync=64.202s`
  - `bfs6vaz4d6` GPU2-7 仍为占卡程序；`i6hf4scd4y` 保持占卡。
- 当前判断：
  - D4.4 48 卡已跑上，且跨节点利用率健康；
  - fake 不是 full model 误训，只训 `lq_proj_in` 和 LoRA；
  - fake dense_full 之前已确认走 flash-attn；
  - fake backward/sync 在 48 卡比 2GPU 明显缓解，但仍是主要成本；这更像 trainable Stage1-like fake critic 的计算/同步成本，而不是参数误放或缺少 flash-attn。

## 2026-05-18 Stage3 v7-D4.4 16 卡占卡恢复与未确认项
- 用户要求如果 16 卡验证排不上或验证低效，必须补占卡，不能让卡长时间空着。
- `bfs6vaz4d6` 2GPU D4.4 timing smoke 后续观察：
  - GPU0 持续 100%；
  - GPU1 虽有验证进程显存占用，但长时间 `0%`，判断为 rank 等待/同步导致的低效状态；
  - 该 smoke 已经提供 D44-1 所需 timing 证据，因此继续运行收益不高。
- 处理：
  - 对 `d44_validate_2gpu` tmux session 发 Ctrl-C，未使用 `pkill` 或模糊 kill；
  - 第一次 Ctrl-C 后 GPU1 释放、GPU0 仍有残留进程；第二次 Ctrl-C 后验证进程全部退出；
  - 启动 `occupy01`：
    - `CUDA_VISIBLE_DEVICES=0,1 bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`
  - 当前 `bfs6vaz4d6` GPU0-7 均为 100% 利用率；
  - `i6hf4scd4y` 仍保持原 `lxh` 占卡，GPU0-7 均为 100%。
- 未确认项，留待用户醒后决策：
  - D44-2：完整 fake/student ownership checksum/grad 检查尚未做；
  - D44-3：pixel、DMD student、fake FM 三路 loss 的 grad norm 细查尚未做；
  - D44-4：同机同卡 Stage1 v5.3.5 backward timing 对照尚未做；
  - D44-5：fake 参数消融，即只训 LoRA / 只训 `lq_proj_in` 的 timing 对照尚未做；
  - 是否继续做这些验证，需要在“别拉低 16 卡利用率”和“继续定位 D4.4 fake 慢点”之间取舍。

## 2026-05-18 Stage3 v7-D4.4 48 卡 fresh1 runner 15 观察
- 48 卡 D4.4 继续推进：
  - 已到 runner 15 / step 4。
  - 主机 `t5qdtykjsw` 连续 6 次 10 秒 GPU 采样均为 97-100%，没有复现长期 0%。
- 新增 timing：
  - runner 9 fake-only：`fake_loss=0.00695447`、`fake_backward_sync=94.174s`
  - runner 10 generator：`loss=0.789458`、`student=0.525690`、`fake_loss=0.01465193`、`dmd_student=0.249116`、`dmd_grad=0.542519`、`fake_backward_sync=41.757s`、`student_backward=15.470s`
  - runner 11 fake-only：`fake_loss=0.02635277`、`fake_backward_sync=99.130s`
  - runner 12 fake-only：`fake_loss=0.05087015`、`fake_backward_sync=20.903s`
  - runner 13 fake-only：`fake_loss=0.03671652`、`fake_backward_sync=42.620s`
  - runner 14 fake-only：`fake_loss=0.01504827`、`fake_backward_sync=19.093s`
  - runner 15 generator：`loss=0.860282`、`student=0.646449`、`fake_loss=0.00000000`、`dmd_student=0.213833`、`dmd_grad=0.502822`、`fake_backward_sync=17.647s`、`student_backward=15.101s`
- 不确定项：
  - runner 15 的 `fake_loss=0.00000000` 需要继续观察；
  - 暂时不判定为 bug，因为 `fake_update=1`、`dmd_student`、`dmd_grad` 同条正常，且前后 fake-only runner 的 fake loss 非零；
  - 后续如果连续多个 generator/fake-only step 都出现 fake loss 为 0，需要检查 fake FM loss 的输入 batch、mask/weight、以及是否某些条件下被置零。
- wandb offload：
  - `wandb_package_v7d44_dfake5` 在 14:50:50 再次成功打包上传；
  - S3 包更新时间约 14:50:52，大小约 `107535` bytes；
  - 说明 15 分钟周期自动刷新已恢复，不是只上传了一次。
- 追加观察：
  - runner 16 fake-only：`fake_loss=0.00718069`、`fake_backward_sync=43.509s`
  - runner 17 fake-only：`fake_loss=0.09564392`、`fake_backward_sync=14.346s`
  - 因 runner 16/17 fake loss 恢复非零，runner 15 的 `fake_loss=0.00000000` 暂时更像单 batch 偶发/格式化极小值，不作为立即阻塞项。

## 2026-05-18 Stage3 v7-D4.4 本机 wandb 同步修复
- 用户反馈网页上没看到 D4.4 wandb loss。
- 排查：
  - t5 训练端 D4.4 本地 offline wandb 正常增长：
    - `offline-run-20260517_142448-yid6lzvt/run-yid6lzvt.wandb`
  - t5 的 package loop 正常每 15 分钟上传到：
    - `s3://lxh/tmp/wandb_offline/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1.tar.gz`
  - t5 直连 `api.wandb.ai` 超时，且默认 `wandb status` 没有 api_key；因此远端只能离线打包，不能直接在线 sync。
- 处理：
  - 用户在本机 `tmux lxh` 窗口 2 登录了 wandb 并要求用 `proxy_on` 同步。
  - 已在 `tmux lxh:2` 启动循环：
    - `proxy_off` 后用 conductor 下载 D4.4 offline tar；
    - 解压到 `/tmp/v7d44_wandb_sync`；
    - `proxy_on` 后运行 `wandb sync --include-offline /tmp/v7d44_wandb_sync/wandb/offline-run-20260517_142448-yid6lzvt`；
    - 每 900 秒重复一次。
  - 第一轮同步完成：
    - `Syncing: https://wandb.ai/veralee/flashvsr/runs/yid6lzvt ... done.`
- 当前结论：
  - D4.4 loss 不是没写，而是之前只完成了远端 offline 打包，没有在线同步到 wandb 网页；
  - 现在本机同步 loop 已挂上，run id 为 `yid6lzvt`。

## 2026-05-18 Stage3 v7-D3.2 step-2000 正确三阶段测试补测
- 用户要求补测 D3.2 `step-2000`，下载到已有桌面目录：
  - `/Users/lixiaohui/Desktop/stage3/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
- checkpoint：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2/output/step-2000.safetensors`
  - 已上传到旧中转：
    - `s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/step-2000.safetensors`
- 测试机器：
  - `6ikhpjzv3z`
  - 只释放 GPU0 做评测；GPU1/2/3 保持占卡。
  - 释放 GPU0 时发现 `occupy01` 留下孤儿 worker，先用 `nvidia-smi` 明确确认 GPU0 PID 为 `2380215`，再只对该 PID 发 `TERM`；未使用模糊 `pkill`。
- 推理设置：
  - 脚本：`wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`
  - `MIN_STEP=2000`
  - `STEP_MOD=2000`
  - `TILED=0`
  - `GPU_LIST=0`
  - `MAX_PARALLEL=1`
  - `num_inference_steps=1`
  - `input_bicubic_upscale=4.0`
  - `color_fix_method=adain`
  - `stage2_attention_mode=block_sparse_chunk_causal`
  - `stage2_topk_ratio=2.0`
  - `stage2_local_num=-1`
  - `stage2_kv_ratio=3.0`
- 输出：
  - 远端：
    - `/mnt/task_wrapper/user_output/artifacts/inference/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01/step-2000`
  - S3：
    - `s3://lxh/data/test/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01/step-2000`
  - 本机：
    - `/Users/lixiaohui/Desktop/stage3/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01/step-2000`
- 完成校验：
  - `step-2000`: `10/10` mp4。
- `logs/step-2000.time`：

## 2026-05-18 Stage3 D3.2 1800 步后 loss 走势与 D4.4 对比复查
- 背景：
  - 用户观察到旧 D3.2 虽然后来被认为 teacher temporal alignment 不符合“前 22”目标，但 `step=1800` 以后 loss 开始下降、视觉效果变好，要求分析是否可以复刻 D3.2，以及如果把 D3.2 的 GT/teacher 都改成前 22 是否可能去掉鬼影。
- D3.2 启动记录确认仍完整：
  - wrapper：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D3-2-Lora-89f-VideoOnly-AuthorWeights-Stage1TeacherAligned-DataFix-OfflineWandb.sh`
  - config：`wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb.yaml`
  - 训练入口实际 exec 到 D3.1 base script，即 `train_flashvsr_stage3_v7_d3_1_lora.py` 代码线。
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
  - `RUN_TS_OVERRIDE=20260516_v7d32_datafix_48gpu_fresh2`。
- D3.2 代码语义复查：
  - `train_flashvsr_stage3_v7_d3_1_lora.py` 中 `nonstreaming_aligned` 对齐返回 `trim_front_to_match`；
  - 也就是 Stage1 teacher 23 positions 对齐 student 22 positions 时，丢弃 teacher position 0，保留 teacher `[1,23)`；
  - 这与用户后来明确的目标不一致。当前目标应为 teacher 前 22，即保留 `[0,22)`、丢弃 `[22,23)`，D4.4 已是 `trim_tail_to_match`。
- D3.2 `run.log` 分段统计到 `step=2084`，未见 DMD spike skip：
  - `1-100`：`loss=1.0361`、`student=0.6391`、`fake_loss=0.0691`、`dmd_student=0.3279`、`dmd_grad=0.5459`
  - `901-1000`：`loss=1.0451`、`student=0.6010`、`fake_loss=0.0462`、`dmd_student=0.3978`、`dmd_grad=0.5655`
  - `1401-1500`：`loss=1.0004`、`student=0.5843`、`fake_loss=0.0461`、`dmd_student=0.3700`、`dmd_grad=0.5537`
  - `1501-1600`：`loss=1.0977`、`student=0.6518`、`fake_loss=0.0593`、`dmd_student=0.3866`、`dmd_grad=0.5535`
  - `1601-1700`：`loss=1.0682`、`student=0.6315`、`fake_loss=0.0547`、`dmd_student=0.3820`、`dmd_grad=0.5754`
  - `1701-1800`：`loss=0.9927`、`student=0.5685`、`fake_loss=0.0439`、`dmd_student=0.3803`、`dmd_grad=0.5626`
  - `1801-1900`：`loss=1.0090`、`student=0.5958`、`fake_loss=0.0465`、`dmd_student=0.3667`、`dmd_grad=0.5605`
  - `1901-2000`：`loss=1.0230`、`student=0.5889`、`fake_loss=0.0473`、`dmd_student=0.3867`、`dmd_grad=0.5704`
  - `2001-2084`：`loss=1.0104`、`student=0.5865`、`fake_loss=0.0505`、`dmd_student=0.3734`、`dmd_grad=0.5607`
- 解释：
  - 从 100-step 均值看，D3.2 并不是在 `1800` 后突然单调下降；更准确是 `1701-1800` 开始出现一段较低均值，`student/reconstruction` 和 `fake_loss` 比早期更低，之后 `1801-2084` 维持在略低于早期的水平。
  - D3.2 视觉在 1800 后变好，可能来自 reconstruction/student 项逐渐稳住、fake critic 学到更贴近当前 student 分布的局部 score，以及长训后 DMD 梯度不再只表现为早期噪声；但 DMD loss 本身仍横盘，不应解读成“DMD loss 单调收敛”。
  - D3.2 的错误 teacher 后 22 对齐可能引入 temporal smoothing/错位正则，所以它能变好不矛盾；但这不等于它的目标正确。
- D4.4 当前对比：
  - run dir：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - 当前只到约 `runner_step=193`、`generator step=39`，因此不能和 D3.2 `step=1800+` 直接比较。
  - D4.4 generator turn `step=1-39`：`loss=0.8678`、`student=0.6834`、`fake_loss=0.0337`、`dmd_student=0.1508`、`dmd_grad=0.4091`。
  - D4.4 fake-only turn `N=155`：`fake_loss=0.0374`，只用于更新 G_fake，不代表 student loss。
  - D4.4 generator turn timing：`student_backward` 均值约 `15.37s`，`fake_backward_sync` 均值约 `49.56s`；
  - D4.4 fake-only timing：`fake_backward_sync` 均值约 `44.65s`，仍是主要耗时。
- 当前判断：
  - D3.2 可以复刻，启动脚本/config/run 记录都在；但如果要“严格复刻”，需要固定同一 D3.1 代码线、同一 `RUN_TS_OVERRIDE` 风格、同一 Stage2/Stage1 权重和旧 `trim_front_to_match` 语义。
  - 如果新开一个“D3.2-like + teacher 前 22 + GT 前 85/前 22 对齐”的版本，它更可能减少鬼影，因为它去掉了 D3.2 最大的 temporal target mismatch；但不能保证单独解决全部鬼影，残影还可能来自 one-step DMD 压力、teacher/fake 分布、pixel/DMD 权重和 Stage2 streaming student 的时序行为。

## 2026-05-18：PPT 用 20 synthetic + 11 real 标准 benchmark 完成

- 目标：
  - 为 PPT 重新生成 20 个 Takano20250205 4K 来源的轻退化合成测试视频；
  - 同时在 20 synthetic 和 11 real 上测试 FlashVSR official、SeedVR3B、SeedVR2-3B、Stage1 535、Stage1 USMGT、Stage2 641、Stage3 v7-D3.2；
  - 单 GPU 单任务，不叠模型，保留 timing。
- 新测试集：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset20_89f_takano20250205_light_x4_lq_20260518`
  - S3：`s3://lxh/data/test/testset20_89f_takano20250205_light_x4_lq_20260518`
  - 生成脚本：`wanvideo/data/flashvsr/tests/export_inference_testset20_takano20250205_light_x4_lq.py`
  - 源数据：`s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/`
- benchmark 脚本：
  - `wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_20synthetic_11real_20260518.sh`
  - Stage1 使用 `run_stage1_v5_3_aligned_dir.sh`，`LQ_PROJ_TEMPORAL_MODE=nonstreaming_aligned`，`NUM_INFERENCE_STEPS=50`，无 streaming/KV cache。
  - Stage2 641 使用 `infer_flashvsr_stage2_v6_1_batch.py`，`NUM_INFERENCE_STEPS=50`，`stage2_attention_mode=block_sparse_chunk_causal`，`stage2_kv_ratio=3.0`。
  - Stage3 v7-D3.2 使用 `infer_flashvsr_stage3_v7_d_batch.py`，同 Stage2 streaming/KV-cache 路径，`NUM_INFERENCE_STEPS=1`。
- 关键 checkpoint：
  - Stage1 535：`stage1_v535_step10000.safetensors`
  - Stage1 USMGT：`stage1_usmgt_takano20250205_step3000.safetensors`
  - Stage2 641：`stage2_v641_step6000.safetensors`
  - Stage3 v7-D3.2：`stage3_v7d32_step2000.safetensors`
  - 本地 critical cache：`/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518`
  - S3 critical backup：`s3://lxh/models/flashvsr/critical_ppt_20260518`
- 输出：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_20synthetic_11real_20260518`
  - S3：`s3://lxh/data/test/ppt_benchmark_20synthetic_11real_20260518`
  - 本机：`/Users/lixiaohui/Desktop/ppt_benchmark_20synthetic_11real_20260518`
- 完整性：
  - synthetic：`gt/lq/flashvsr_official/seedvr2_3b/seedvr3b/stage1_535_step10000/stage1_usmgt_step3000/stage2_v641_step6000/stage3_v7d32_step2000` 均为 20 个 mp4。
  - real：`lq/flashvsr_official/seedvr2_3b/seedvr3b/stage1_535_step10000/stage1_usmgt_step3000/stage2_v641_step6000/stage3_v7d32_step2000` 均为 11 个 mp4。
- timing 摘要：
  - synthetic seconds/video：FlashVSR official `42.800`，SeedVR2-3B `50.650`，SeedVR3B `228.650`，Stage1 535 `209.450`，Stage1 USMGT `209.850`，Stage2 641 `149.300`，Stage3 v7-D3.2 `18.250`。
  - real seconds/video：FlashVSR official `41.909`，SeedVR2-3B `53.273`，SeedVR3B `230.182`，Stage1 535 `210.455`，Stage1 USMGT `210.273`，Stage2 641 `141.727`，Stage3 v7-D3.2 `18.455`。
- 收尾：
  - 结果已下载到桌面目录；
  - `6ikhpjzv3z` 上测试结束后已重新启动正常占卡，最后检查 GPU0-3 均为 `100%`，约 `166602 MiB`。

    - `status=0`
    - `num_inputs=10`
    - `num_outputs=10`
    - `seconds=184`
    - `seconds_per_video=18.400`
    - `tiled=0`
- 收尾：
  - 本地已从 S3 同步到用户指定桌面目录。
  - `6ikhpjzv3z` GPU0 已重新启动 `occupy0`，GPU1 为 `occupy1`，GPU2/3 为 `occupy23`；最后检查 GPU0-3 均为 100% 利用率。

## 2026-05-18 D4.4 当前状态确认与组会文档
- 用户要求确认 D4.4 是否仍在跑，并参考 2026-05-11 / 2026-05-14 两份组会文档，整理 5 月 14 日之后的组会汇报。
- D4.4 48 卡当前状态只读检查：
  - 主机：`t5qdtykjsw`
  - tmux：`v7d44_48gpu` 仍存在。
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
  - 训练进程仍在，`nvidia-smi` 抽查 t5 GPU0-7 均为 `100%` 利用率，高显存占用。
  - 最新日志已到约 `runner_step=678` / `generator step=136`：
    - runner 675 generator：`loss=0.498236`、`student=0.414720`、`fake_loss=0.00979650`、`dmd_student=0.073720`、`dmd_grad=0.297983`
    - runner 678 fake-only：`fake_loss=0.00706585`、`fake_backward_sync=174.750s`
  - output 目录已看到：`step-1/2/5/10/20/50/100.safetensors`。
  - W&B 远端直接同步 tmux：`wandb_sync_v7d44_direct` 仍存在，run id：`yid6lzvt`。
- 新增组会文档：
  - `doc/flashvsr_training_group_meeting_20260518.md`
  - 内容覆盖：
    - Stage1 USMGT teacher；
    - Stage2 v6.4.1 作为 Stage3 student；
    - D3.2、D4.1、D4.2、D4.3、D4.4 的目的、设置、结果和问题；
    - D3.2 step-1500/2000 与 20 synthetic + 11 real benchmark；
    - fake backward/sync、rank skew、wandb、远程安全停实验等工程问题；
    - 当前 D4.4 正在跑、D4.2 作为备用标准线的结论。

## 2026-05-18 D4.4 验证重点调整：从慢点/残影转向代码正确性
- 用户反馈：
  - D6 残影已经消失，D44-6 不再优先做；
  - D44-4/D44-5 主要解释 fake 慢点，但 48 卡 D4.4 当前约 60s/step，慢点暂时不是首要阻塞；
  - 当前更希望验证 D4.4 代码有没有语义错误，例如 fake loss 是否错误回传 student、DMD student 是否误更新 fake/real、功能是否只是“跑通”但实现不严谨。
- DMD 首个 generator turn 为 0 的解释：
  - 16 卡 ownership run 的 runner 0 出现 `dmd_student=0` / `dmd_grad=0`；
  - 重新审查后判断这不是异常：runner 0 计算 DMD 时，`G_real` 和 `G_fake` 都刚从同一个 Stage1 USMGT checkpoint 初始化，fake 还没完成第一次 optimizer step；
  - DMD student loss 使用 real/fake score difference，初值相同则 `real_x0 - fake_x0` 可以为 0；
  - 正式 48 卡 run 中 runner 5、runner 10 等已经出现非零 `dmd_student` / `dmd_grad`，说明 fake 更新后 DMD 项开始生效。
- 4 卡验证机状态：
  - 机器：`6ikhpjzv3z`
  - 当前 GPU0-3 均由 `occupy_pptbench` 占卡，约 `166602 MiB`、`100%`；
  - 本轮没有释放 GPU，因为先做的是代码级 correctness 审查，不需要动卡。
- 在 4 卡机远端 `/mnt/task_runtime/lucidvsr` 与本地均做了 D4.4 静态 correctness 检查：
  - fake FM loss 内部使用 `clean_latents.detach()`；
  - DMD real/fake probe 在 `torch.no_grad()` 中执行；
  - DMD target 使用 `.detach()`，梯度只回到 student `clean_latents`；
  - fake 使用 `fake_accelerator.backward(fake_loss)`；
  - student 使用 `accelerator.backward(student_total_loss)`；
  - fake/student optimizer step 分离；
  - `fake_probe_model` 在 fake DeepSpeed prepare 后重新绑定为 prepared `fake_model`，DMD 使用当前 trainable fake critic；
  - dfake 调度为 `runner_step % stage3_dfake_gen_update_ratio == 0`；
  - 远端代码确认存在 `trim_tail_to_match`，即 teacher 前 22 对齐。
- 当前结论：
  - 未发现 fake loss 回传 student、DMD student loss 更新 fake/real、或 teacher 对齐仍为后 22 的代码路径；
  - 还没有动态拿到 fake-only runner 参数 delta，因此“fake-only 只改 fake”仍不是完全实验证明，但从代码图看没有明显串线；
  - D44-4/D44-5/D44-6 暂缓/取消当前优先级，后续如性能或残影再次成为问题再恢复。

## 2026-05-18 Stage3 v7-D4.4-DMDOnly 16 卡对照启动
- 用户要求：
  - 不要修改正在跑的 D4.4 正式代码；
  - 如果需要改逻辑，必须复制新文件；
  - 先做一个只有 DMD、没有 pixel/recon 的对照，观察 DMD 单独优化方向。
- 命名：
  - 不叫 D4.5，因为不是新主线实现；
  - 记录为 `v7-D4.4-DMDOnly` / `v7d44_dmdonly_16gpu` ablation。
- 本地新增文件，未修改 `train_flashvsr_stage3_v7_d4_4_lora.py`：
  - config：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5.yaml`
  - launch：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-16GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-DMDOnly-Dfake5.sh`
- loss 设置：
  - `stage3_flow_weight=0`
  - `stage3_mse_weight=0`
  - `stage3_lpips_weight=0`
  - `stage3_dmd_weight=1`
  - `stage3_fake_fm_weight=1`
  - `stage3_dfake_gen_update_ratio=5`
  - 保留 fake FM 是必要的，否则 fake critic 不更新，后续 DMD score difference 没法形成。
- 本地校验：
  - yaml 可读，确认 loss 权重为 `0/0/0/1/1`，`max_train_steps=20`；
  - launch `bash -n` 通过。
- 远端同步与校验：
  - 机器：`bfs6vaz4d6` + `i6hf4scd4y`
  - 两端 `/mnt/task_runtime/lucidvsr` 均看到新增 config/launch；
  - 两端 launch `bash -n` 通过。
- 启动前资源处理：
  - 两台机器原 `occupy_all` session 已停止；
  - 初次停止后显存仍显示占用，随后确认无 running processes，显存释放为 0；
  - 未使用模糊 `pkill`。
- 启动信息：
  - tmux：两台均为 `d44_dmdonly16`
  - master：`240.6.132.130:29645`
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5_20260518_v7d44_dmdonly_16gpu`
  - stage2 checkpoint：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
  - Stage1 real/fake checkpoint：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- 初始结果：
  - 日志确认 `D4.4 dual Accelerate DeepSpeed engine`；
  - fake 参数仍为 `lora=283115520`、`lq_proj_in=287845888`；
  - student 主分支打印：
    - `loss=0.000000`
    - `mse=0.000000`
    - `lpips=0.000000`
    - `need_reconstruction=False`
    - 说明 pixel/recon 没有参与 student 目标；
  - probe 内部打印的 `compute_z_pred=False flow=...` 是 G_real/G_fake probe 自己的 forward，不是 student reconstruction loss。
- runner 0 / 1 早期结果：
  - runner 0 generator：
    - `loss=0.006974`
    - `student=0.000000`
    - `fake_loss=0.00697420`
    - `real_probe=0.056691`
    - `fake_probe=0.116816`
    - `dmd_student=0.000000`
    - `dmd_grad=0.000000`
    - `fake_backward_sync=38.738s`
    - `student_backward=4.782s`
  - runner 1 fake-only：
    - `loss=0.002623`
    - `student=0.000000`
    - `fake_loss=0.00262338`
    - `fake_backward_sync=14.225s`
- 解释：
  - runner 0 DMD 为 0 仍符合预期，因为 `G_real/G_fake` 初始相同；
  - DMD-only 对照要看 fake 更新若干步后第二个 generator turn，即 runner 5 是否出现非零 `dmd_student/dmd_grad`，以及后续 checkpoint 的视觉效果。
- watch：
  - 6 节点、16 卡两台 watch 已恢复；
  - 4 卡 `6ikhpjzv3z` watch 窗口单独重新接入。

## 2026-05-18 Stage3 v7-D4.4-DMDOnly runner5 结论
- 继续观察同一 16 卡对照：
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5_20260518_v7d44_dmdonly_16gpu`
  - tmux：`d44_dmdonly16`
  - 机器：`bfs6vaz4d6` + `i6hf4scd4y`
- runner 2-4 fake-only：
  - runner 2：`student=0.000000`、`fake_loss=0.01299118`、`fake_backward_sync=196.618s`
  - runner 3：`student=0.000000`、`fake_loss=0.03037244`、`fake_backward_sync=174.086s`
  - runner 4：`student=0.000000`、`fake_loss=0.00000000`、`fake_backward_sync=24.888s`
  - 说明 pixel/recon 在该对照中确实没有参与 student loss，但 16 卡 fake sync 波动很大。
- runner 5 第二个 generator turn：
  - `loss=0.161390`
  - `student=0.000000`
  - `fake_loss=0.01611994`
  - `dmd_student=0.145271`
  - `dmd_grad=0.411943`
  - `dmd_skip=0`
  - timing：`dmd=18.121s`、`fake_backward_sync=100.645s`、`student_backward=4.834s`
- 结论：
  - DMD-only 16 卡对照证明：在 fake critic 经过 runner 1-4 更新后，第二个 generator turn 已经出现非零 DMD student loss 和 DMD gradient；
  - 因为该 config 中 `flow/mse/lpips=0` 且日志 `student=0.000000`，runner 5 的 student-side 非零目标来自 DMD，而不是重建损失；
  - 这支持 D4.4 正式训练里 DMD 分支确实生效，不是只靠 pixel/recon 在训练。
- 边界：
  - 这个对照不证明 DMD-only 视觉效果会好，只证明 loss/gradient 语义；
  - 该对照没有修改正式 D4.4 代码，只新增独立 config/launch；
  - 16 卡 fake-only `fake_backward_sync` 不稳定，不适合拿来评估正式 48 卡吞吐。

## 2026-05-18 Stage3 v7-D4.4-DMDOnly 16 卡对照停止
- 用户询问是否继续训练该 16 卡 DMD-only 对照，以及是否有 validation / wandb。
- 配置确认：
  - `validation_num_samples=0`，没有 validation；
  - `use_wandb=false`，不会自动 W&B 同步；
  - `max_train_steps=20`，`save_steps=5`，`extra_save_steps=1,2,5,10,20`。
- 继续观察到 runner 10：
  - `loss=0.170929`
  - `student=0.000000`
  - `fake_loss=0.01019344`
  - `dmd_student=0.160736`
  - `dmd_grad=0.434257`
  - `dmd_skip=0`
  - 这进一步确认 DMD-only 的非零 student-side 目标来自 DMD，而不是 pixel/recon。
- 尝试等到 `step-5.safetensors`：
  - 已保存 checkpoint：`step-1.safetensors`、`step-2.safetensors`；
  - 训练到 runner 15 开始后长时间停在 generator turn 内部，未输出 runner 15 完整 `stage3c_train`，未生成 `step-5.safetensors`；
  - 由于 DMD 语义验证已经完成，继续占用 16 卡收益较低，因此停止。
- 停止方式：
  - 仅对两台机器明确 tmux session `d44_dmdonly16` 发送 Ctrl-C；
  - 没有使用 `pkill` 或模糊 kill。
- 收尾：
  - 机器：`bfs6vaz4d6` + `i6hf4scd4y`
  - 两台均重新启动 `occupy_all`：
    - `bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`
  - 最后检查两台 8 卡均约 `166604 MiB`、`100%`。

## 2026-05-18 组会文档逻辑重写
- 用户指出 `doc/flashvsr_training_group_meeting_20260518.md` 的叙述顺序有问题：
  - 5 月 14 日汇报时 `v7-C` 只是“下一步计划”，不是已经完成；
  - 需要先讲 C0-C6 如何逐步把 `G_real/G_fake` 接进去，以及 C6 的问题；
  - 再讲为什么从 C6 过渡到 D / D1 / D2 / D3.2；
  - D3.2 长训期间并行推进 Stage1 USMGT teacher，应在这个位置插入 Stage1 `step-3000` 的结果；
  - 然后再解释为什么进入 D4、D4.2 / D4.4 当前状态。
- 已重写：
  - `doc/flashvsr_training_group_meeting_20260518.md`
- 新版结构：
  - 5.14 后的主线；
  - `v7-C` 从计划到 C6 trainable `G_fake`；
  - C6 暴露的 validation / W&B / 快照问题；
  - C6 到 D stable / D1 / D2 / D3.2 的逻辑；
  - D3.2 的慢、loss 波动、鬼影和 step-2000 改善；
  - 并行 Stage1 USMGT `step-3000`；
  - D4 的 front-22、dfake=5、fake loss detach、D4.2 / D4.4；
  - benchmark、工程问题和下一步。

## 2026-05-18 PPT close-up synthetic benchmark 重选启动
- 用户反馈上一版 20 个 synthetic PPT 测试视频不够适合展示，要求从 Stage3/D4.4 同源 Takano 训练数据里重新挑选近景、大主体、细节多的视频。
- 新增 close-up 测试集生成脚本：
  - `wanvideo/data/flashvsr/tests/export_inference_testset20_takano20250205_closeup_light_x4_lq.py`
  - 逻辑：从 `takano_video_20250205_test_4k_tar_manifest.txt` 随机抽候选，不和上一版 20 个 sample_id 重复；对候选按中心区域边缘/细节密度、中心主体 proxy、曝光和饱和度打分，取 top 20；再用轻量 Aliyun x4 degradation 生成 LQ。
  - 新测试集远端路径：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset20_89f_takano20250205_closeup_light_x4_lq_20260518`
  - 新测试集 S3 备份：`s3://lxh/data/test/testset20_89f_takano20250205_closeup_light_x4_lq_20260518`
- 新增只测 synthetic 的 7 方法 benchmark 脚本：
  - `wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_20closeup_synthetic_20260518.sh`
  - 输出远端路径：`/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_20closeup_synthetic_20260518`
  - 输出 S3：`s3://lxh/data/test/ppt_benchmark_20closeup_synthetic_20260518`
  - 方法：FlashVSR official、SeedVR3B、SeedVR2-3B、Stage1 v5.3.5 step10000、Stage1 USMGT step3000、Stage2 v6.4.1 step6000、Stage3 v7-D3.2 step2000。
  - 运行机器：`6ikhpjzv3z`，GPU `0,1,2,3`；同一张 GPU 不叠多个方法。
- 当前状态：
  - 已通过本地 `sync` 同步新增脚本到 6ikh。
  - 已在远端 tmux `pptcloseup_run` 启动生成+benchmark 长任务。
  - close-up 测试集已完成 20 个 GT / 20 个 LQ 写入，benchmark 已进入第一批方法并行。
  - 脚本结束后会自动 `conductor s3 sync` 输出目录并恢复 GPU `0,1,2,3` 占卡。

## 2026-05-18 PPT random50 中度退化 benchmark
- 用户反馈 close-up 20 个筛出来仍以自然景观为主，不适合 PPT 展示，要求放弃该测试，改为随机抽 50 个视频：
  - Takano 25 个；
  - Yubari 25 个；
  - 退化强度改为“中度退化”，介于之前 light 测试退化和完整 Stage1 训练退化之间。
- 已停止 4 卡测试机 `6ikhpjzv3z` 上的 close-up benchmark，并恢复 `occupy_pptbench_closeup` 占卡。
- 新增中度退化配置：
  - `wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_medium_x4test.yaml`
  - 设计：单阶段 x4 Aliyun/video compression，关闭 second stage；比 light 更强的 resize/noise/jpeg/video compression，但不使用完整训练级二阶段退化。
- 新增 / 更新随机 50 导出与 benchmark：
  - `wanvideo/data/flashvsr/tests/export_inference_testset50_random_takano_yubari_light_x4_lq.py`
    - 复用脚本名，但通过传入 `medium_x4test` config 生成中度退化；
    - Takano 使用 `takano_video_20250205_test_4k_tar_manifest.txt`；
    - Yubari 修正为 `FlashVSRStreamingDataset` tar/streaming 路径，避免远端 parquet `ref_big conductor client unavailable`。
  - `wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_50random_medium_synthetic_20260518.sh`
  - `wanvideo/model_inference/flashvsr/history/run_generate_and_benchmark_50random_medium_synthetic_20260518.sh`
- 路径：
  - 测试集远端：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518`
  - 测试集 S3：`s3://lxh/data/test/testset50_89f_random25takano25yubari_medium_x4_lq_20260518`
  - benchmark 远端：`/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_50random_medium_synthetic_20260518`
  - benchmark S3：`s3://lxh/data/test/ppt_benchmark_50random_medium_synthetic_20260518`
  - 桌面下载目标：`/Users/lixiaohui/Desktop/ppt_benchmark_50random_medium_synthetic_20260518`
- 运行机器：
  - `bfs6vaz4d6` 作为 8 卡 benchmark 节点；
  - `i6hf4scd4y` 保持占卡不动；
  - benchmark 规则：GPU `0-6` 分别跑 7 个方法，GPU `7` 占卡；同一张卡不叠多个方法，保证时间统计相对公平。
- 方法：
  - `flashvsr_official`
  - `seedvr3b`
  - `seedvr2_3b`
  - `stage1_535_step10000`
  - `stage1_usmgt_step3000`
  - `stage2_v641_step6000`
  - `stage3_v7d32_step2000`
- 当前状态：
  - 测试集已生成完成：50 个 GT + 50 个 LQ；
  - benchmark 已启动并在 `bench50_medium` 远端 tmux 中运行；
  - 已挂本地 `download_bench50_medium` tmux，等待远端最终同步到 S3 后自动下载到桌面；
  - 远端脚本结束后会启动 `occupy_pptbench50`，恢复 bfs `0-7` 卡占卡。

## 2026-05-19 v7-D4.4 48 卡掉线后 resume
- 用户反馈 48 卡 v7-D4.4 实验掉了，要求恢复同一个 48 卡训练。
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1`
- 启动脚本：
  - `/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- config：
  - `/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- 代码：
  - `/mnt/task_runtime/lucidvsr/wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- 6 节点：
  - rank0 `t5qdtykjsw`
  - rank1 `a9suya6gxe`
  - rank2 `67dxkwcb7m`
  - rank3 `ui9n6p293s`
  - rank4 `g48bd6x4h7`
  - rank5 `gx2intv5rk`
- 原始掉线原因：
  - 主日志显示 NCCL ALLREDUCE watchdog timeout，`Timeout(ms)=600000`；
  - 当时训练实际已到约 `runner_step=1118 / step=224`，但完整 training state 只保存到 `step-200`。
- resume 处理：
  - 初始尝试从 `training_state/step-200` 恢复失败，从节点缺少 Accelerate/DeepSpeed common files；
  - 修复 `step-200` 过程中误把从节点局部 optimizer shard 做成了错误 symlink，因此本次不再使用 `step-200` training state；
  - 切到干净的 `training_state/step-100` 恢复。
- 补齐的 state：
  - 从 rank0 上传并分发 student/Accelerate common files：
    - S3：`s3://lxh/tmp/d44_resume_step100_common_20260519`
    - `latest`
    - `scheduler.bin`
    - `flashvsr_training_state.json`
    - `zero_to_fp32.py`
    - `pytorch_model/mp_rank_00_model_states.pt`
  - 从 rank0 上传并分发 G_fake DeepSpeed common model state：
    - S3：`s3://lxh/tmp/d44_resume_step100_fake_common_20260519/mp_rank_00_model_states.pt`
    - 写入各从节点：
      - `output/stage3_fake_deepspeed/global_step100/mp_rank_00_model_states.pt`
  - 各节点原本已有自己的 optimizer/random shard：
    - student `training_state/step-100`：每节点 8 个 rank shard；
    - fake `output/stage3_fake_deepspeed/global_step100`：每节点 8 个 fake optimizer shard。
- 启动参数：
  - `RUN_TS_OVERRIDE=20260518_v7d44_48gpu_fresh1`
  - `WANDB_RUN_ID=yid6lzvt`
  - `WANDB_RESUME=allow`
  - `NCCL_TIMEOUT=1800`
  - `MASTER_ADDR=240.12.138.137`
  - 最终成功启动端口：`MASTER_PORT=29564`
  - `EXTRA_ARGS='--resume_training_state_dir .../output/training_state/step-100'`
- 安全操作记录：
  - 停残留进程只使用精确 tmux session：`tmux send-keys -t v7d44_48gpu:0 C-c`；
  - 未使用任何模糊 `pkill` / `pgrep | kill`。
- 当前确认：
  - 2026-05-19 13:31 CST 左右，6 台均有 9 个训练相关进程；
  - 主节点显存约 `147828-148466 MiB`，GPU 利用率约 `92-100%`；
  - 日志已确认：
    - `[stage3d43_fake_resume] loaded G_fake DeepSpeed checkpoint`
    - `[stage3c_resume] loaded student/fake state student_step=100 runner_step=500 epoch_id=0`
    - `[stage3c_train] epoch=0 step=101 runner_step=500 generator_update=1 ...`
    - `[stage3c_train] epoch=0 step=101 runner_step=501 generator_update=0 ...`
- 注意：
  - 本次是从 `step-100` optimizer/training state 恢复，不是从 `step-200` 恢复；
  - `step-200.safetensors` 仍可作为权重文件使用，但 `training_state/step-200` 当前不要再直接拿来做 optimizer-state resume，除非重新从可靠备份修复。

## 2026-05-19 v7-D4.4 W&B loss 分项记录修正
- 用户反馈当前 W&B 里主要看到总 loss，D4.4 的 loss 曲线难以判断。
- 问题判断：
  - D4.4 是 `dfake=5` 调度；
  - `runner_step` 每个 batch 增加，G_fake 每个 `runner_step` 更新；
  - `student_step/global_step` 只有 generator/student 更新时才增加；
  - 旧代码用 `wandb_run.log(..., step=global_step)`，会让 5 个 runner step 中的 fake-only 更新挤在同一个 W&B step 上，曲线容易被合并/覆盖或看起来只有总 loss。
- 已修改：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- 新 W&B 记录方式：
  - 用 `runner_step` 作为 W&B x-axis；
  - 额外记录 `train/student_step`；
  - 保留 `train/loss`，并新增/显式记录：
    - `train/total_loss`
    - `train/student_recon_loss`
    - `train/student_total_loss`
    - `train/fake_loss`
    - `train/fake_fm_loss`
    - `train/dmd_student_loss`
    - `train/dmd_grad_norm`
    - `train/flow_loss`
    - `train/mse_loss`
    - `train/lpips_loss`
    - `train/weighted_flow_loss`
    - `train/weighted_mse_loss`
    - `train/weighted_lpips_loss`
    - `train/generator_did_update`
    - `train/fake_did_update`
    - `train/recon_latents`
    - `train/decoded_frames`
    - sampled latent/frame window metadata。
- 验证：
  - 本机通过 `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`。
- 注意：
  - 这个修改不会影响当前已经在内存中运行的 48 卡 D4.4 进程；
  - 需要下一次启动或 resume 后才会生效；
  - 训练数学未改，只改 W&B logging。

## 2026-05-19 v7-D4.4 W&B logging 收窄并重启
- 用户反馈上一版 W&B 分项过多，只需要盯总 loss 和核心四类 loss。
- 已把 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py` 的 W&B logging 收窄为：
  - `train/loss`
  - `train/dmd_student_loss`
  - `train/flow_loss`
  - `train/mse_loss`
  - `train/lpips_loss`
  - `train/fake_fm_loss`
  - `train/generator_did_update`
  - `train/runner_step` 仅作为 W&B x-axis 保留，不作为需要人工盯的 loss。
- 本机验证：
  - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
- 远端同步验证：
  - t5/a9 的代码 hash 均为 `62775a821be4afab4df97c844b11ea8a35a5d77254cf90b6fe2afbf906dda121`；
  - t5 远端 `/mnt/conda_envs/flashvsr/bin/python -m py_compile ...` 通过。
- 重启记录：
  - 先尝试 `--wandb_mode online`，但 t5 日志出现 `wandb: Network error (ConnectTimeout), entering retry loop.`；
  - 为避免 48 卡被 W&B 网络阻塞，已精确 `tmux send-keys -t v7d44_48gpu:0 C-c` 停止 online 尝试；
  - 重新用 `--wandb_mode offline` 启动，继续同一 `RUN_TS_OVERRIDE=20260518_v7d44_48gpu_fresh1` 和 `WANDB_RUN_ID=yid6lzvt`；
  - 最终成功端口：`MASTER_PORT=29566`；
  - resume state：`.../output/training_state/step-100`。
- 当前确认：
  - 6 台均有 9 个训练相关进程；
  - 显存约 `147-148GB/GPU`；
  - 新日志已确认：
    - `wandb: W&B syncing is set to offline`
    - 新 offline run：`wandb/offline-run-20260518_234749-yid6lzvt`
    - `[stage3c_train] epoch=0 step=101 runner_step=500 ...`
    - `[stage3c_train] epoch=0 step=101 runner_step=501 ...`
- W&B 同步状态：
  - t5 训练环境 `/mnt/conda_envs/flashvsr/bin/wandb login --verify` 可确认账号；
  - 但使用 `/mnt/conda_envs/flashvsr/bin/wandb sync` 会报 `ConnectTimeout`；
  - 按 `远程机器wandb` skill 重新检查后，确认 t5 的 b200 环境有可用 W&B：
    - `/miniforge/envs/b200/bin/wandb`
  - 使用 b200 wandb 同步最新 run 成功：
    - `/miniforge/envs/b200/bin/wandb sync --include-offline .../wandb/offline-run-20260518_234749-yid6lzvt`
    - 输出：`Syncing: https://wandb.ai/veralee/flashvsr/runs/yid6lzvt ... done.`
  - 已重新挂远端自动同步 tmux：
    - `wandb_sync_v7d44_direct`
    - 每 900 秒扫描 `.../wandb/offline-run-*`
    - 使用 `zsh -lc "/miniforge/envs/b200/bin/wandb sync --include-offline $d"`
  - 训练脚本自带 `wandb_package_v7d44_dfake5` offline package 循环仍在，作为 S3/本机同步兜底。
- 解释：
  - 当前不是 W&B API key 缺失；
  - 这条记录里的“flashvsr env 的 wandb 同步会超时”后来被证明不准确，见下一条 2026-05-19 复查；
  - 后续 D4.4/DMD 类实验优先在 t5 远端做离线 run 同步，不要再误判为 t5 不能 W&B。

## 2026-05-19 v7-D4.4 W&B 环境复查和同步修正
- 用户指出：t5 标准交互 zsh 中，`conda activate flashvsr` 后外部 `wandb` 也能成功。
- 已在 t5 同一个 D4.4 run 目录重新做 A/B：
  - A：交互式 `conda activate flashvsr`，实际 `wandb=/mnt/conda_envs/flashvsr/bin/wandb`，`wandb sync --include-offline ...-yid6lzvt` 成功；
  - B：模拟训练脚本环境：`PATH=/mnt/conda_envs/flashvsr/bin:...`、`PYTHONNOUSERSITE=1`、`AWS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`，同一个 flashvsr wandb 同步也成功；
  - C：b200 wandb 也成功。
- 关键结论：
  - `flashvsr` env 本身不是问题；
  - 之前 online W&B `ConnectTimeout` 的直接差异是当前训练父进程环境没有代理变量：
    - 没有 `HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy`；
    - 交互 zsh 中这些变量是 `http://proxy.config.pcp.local:3128`。
  - `wandb status` 不足以证明训练内 online W&B 一定能通，必须检查真实 `wandb sync` 或训练进程 `/proc/<pid>/environ`。
- 已修正：
  - 更新 skill：`/Users/lixiaohui/.codex/skills/远程机器wandb/SKILL.md`，记录“先比较训练进程环境和交互 zsh 代理”的规则；
  - D4.4 48 卡启动脚本新增默认代理导出，影响后续启动：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
    - `HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy=http://proxy.config.pcp.local:3128`
  - `bash -n` 通过。
- 当前运行中的 48 卡训练没有重启，仍是 offline W&B；已重挂 t5 远端自动同步 tmux：
  - tmux：`wandb_sync_v7d44_direct`
  - 环境：显式导出代理，`source /miniforge/etc/profile.d/conda.sh && conda activate flashvsr`
  - 只同步当前 run id：`wandb/offline-run-*-yid6lzvt`
  - 间隔：`900s`
  - 已看到 `Syncing: https://wandb.ai/veralee/flashvsr/runs/yid6lzvt ... done.`
- 备注：
  - 之前临时启动同步 tmux 时曾误用本地展开的 `$RUN_DIR`，导致远端 `cd` 到 home；已立即用硬编码绝对路径重挂，避免同类错误。
  - 当前继续保留 offline 训练 + 远端 direct sync；如果以后要直接 `wandb_mode=online`，需要用带代理的启动脚本重新启动。

## 2026-05-20 89f 单视频推理时间拆分
- 目的：
  - 用户需要解释 PPT benchmark 中 Stage3 v7d32 单视频约 `18s`、FlashVSR official 约 `50-77s` 的差异来源；
  - 重新只测一个 89f 视频，按模型加载、读写、LQ projector、DiT/sampling、VAE decode、color fix、保存等阶段拆时间，并给出按 89 帧均摊表。
- 新增脚本：
  - `wanvideo/model_inference/flashvsr/profile_single_video_timing_20260520.py`
  - `wanvideo/model_inference/flashvsr/history/run_single_video_timing_profile_20260520.sh`
- 输入：
  - `takano_00_lq.mp4`
  - 数据集：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518`
  - S3：`s3://lxh/data/test/testset50_89f_random25takano25yubari_medium_x4_lq_20260518`
- 方法：
  - `flashvsr_official`
  - `seedvr3b`
  - `seedvr2_3b`
  - `stage1_535_step10000`
  - `stage1_usmgt_step3000`
  - `stage2_v641_step6000`
  - `stage3_v7d32_step2000`
- 重要修正：
  - Stage1 profiling 初版漏掉原始 `infer_from_lq` 中的 `latents=noise` 兜底，导致缺 `latents`，已修复；
  - Stage1/Stage2 手写计时路径初版漏了 `torch.no_grad()`，导致显存异常到 `177GB` 并 OOM，已修复；
  - 6ikh 的占卡脚本 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh` 不读取 `GPUS` 环境变量，传 `0,1,3` 也会占满 4 卡；本次正式计时期间不叠占卡，结束后恢复全卡占卡。
- 输出：
  - 远端：`/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_single_video_timing_20260520`
  - S3：`s3://lxh/data/test/ppt_benchmark_single_video_timing_20260520`
  - 桌面：`/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520`
  - 主要报告：`timing_analysis_table_20260520.md`
- 关键结论：
  - Stage3 v7d32 总时间 `18.415s`，其中 DiT/sampling 只有 `1.758s`；主要耗时变成 VAE decode `8.238s`、color fix `2.237s`、模型加载 `3.198s`；
  - Stage2 v641 50-step 总时间 `143.539s`，其中 LQ projector streaming `36.458s`、DiT/sampling `90.504s`；
  - Stage1 535/USMGT 总时间约 `210s`，核心瓶颈是 dense 50-step DiT，约 `192.7s`；
  - FlashVSR official 当前只作为 upstream subprocess 黑盒计时：总 `76.868s`，subprocess `74.223s`；若要拆 model load / official decode / cache，需要继续在 official repo 脚本内加 timer；
  - SeedVR3B/SeedVR2-3B 的日志能拆出 DiT/VAE configure 与 sampler 估计，但总时间仍包含分布式初始化、视频读写/编码等 wrapper overhead。

## 2026-05-20 89f 单视频推理耗时统一归并与官方 FlashVSR 精细计时
- 背景：
  - 用户指出上一版耗时表按各模型私有字段展开，空项太多，无法横向解释；
  - 新要求不是照搬别组截图，而是按 LucidVR / SeedVR / FlashVSR 的共性推理链路归并。
- 新增官方 FlashVSR profile 脚本：
  - `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/FlashVSR/examples/WanVSR/infer_flashvsr_full_cloud_profile_20260520.py`
  - 复制官方 `infer_flashvsr_full_cloud.py` 后加计时，不改原脚本；
  - 通过 monkey patch 拆出：
    - `official_lq_projector_stream_s`
    - `official_dit_model_fn_s`
    - `official_vae_decode_s`
    - `official_color_fix_s`
    - `official_prepare_input_s`
    - `official_tensor2video_s`
    - `official_save_video_s`
    - 加载/初始化分项。
- 远端执行：
  - 机器：`6ikhpjzv3z`
  - 输入：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset50_89f_random25takano25yubari_medium_x4_lq_20260518/lq/takano_00_lq.mp4`
  - 模型：`/mnt/models/FlashVSR-v1.1`
  - 运行时需从 `/mnt/task_runtime/FlashVSR/examples/WanVSR` 启动，否则官方 `../../examples/WanVSR/prompt_tensor/posi_prompt.pth` 相对路径会找不到；
  - 测试期间停止 6ikh 占卡，完成后已恢复全卡占卡。
- 官方 FlashVSR 单视频 profile 结果：
  - `official_script_total_wall_s=22.837s`
  - 单视频不含全脚本外层的 `official_total_wall_s=15.642s`
  - `pipe(...) = 11.017s`
  - LQ projector `0.701s`
  - DiT/model_fn `1.828s`
  - VAE decode `8.475s`
  - color fix `0.005s`
  - tensor2video `1.912s`
  - save `1.018s`
  - 89 输入按官方流式规则输出 85 帧。
- 统一归并报告：
  - 桌面目录：`/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520`
  - 主要新表：`timing_unified_common_phases_20260520.md`
  - JSON：`timing_unified_common_phases_20260520.json`
  - 归并行：
    - 模型加载 / 初始化
    - 输入读取 / resize / 准备
    - LQ conditioning / projector
    - Denoise / DiT / sampler
    - Scheduler step
    - VAE decode
    - Tensor -> video frames
    - Color fix
    - 视频保存 / trim
    - 模型运算合计
    - 其他开销 / 未拆分
    - 端到端冷启动总时长。
- 关键结论：
  - 之前官方 FlashVSR 黑盒统计约 `76.9s`，主要是外部子进程冷启动/加载/包装口径，不等于模型核心推理；
  - 官方 FlashVSR 精细计时后，核心 `pipe(...)` 约 `11.0s`；
  - LucidVR Stage3 D3.2 核心运算约 `10.68s`，和官方 FlashVSR 同量级；
  - LucidVR Stage3 端到端 `18.41s`，主要额外来自加载、读视频、VAE decode、color fix、保存；
  - Stage1 慢点仍是 50-step dense/offline DiT，约 `192.7s`；
  - Stage2 通过流式 chunk / KV-cache 语义 / block-sparse 路径把 DiT 降到约 `90.5s`；
  - SeedVR 系列只能从日志稳定拆出 configure 和 sampler，其 VAE/IO/视频循环仍归入“其他开销 / 未拆分”。

## 2026-05-20 PPT 单视频推理时间统计口径重写
- 背景：
  - 用户指出 `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/timing_unified_common_phases_20260520.md` 旧版表格仍然不适合汇报：SeedVR 没有拆分、全 0/无效项太多、没有按“启动/加载”和“模型运算”分组，也不能直接看出 PPT 应使用的 `/frame` 模型时间。
- 修正：
  - 覆盖重写：
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/timing_unified_common_phases_20260520.md`
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/timing_unified_common_phases_20260520.json`
  - 同步生成别名版本：
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/timing_grouped_startup_model_total_20260520.md`
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/timing_grouped_startup_model_total_20260520.json`
  - 写入 SeedVR profile 原始拆分记录：
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/seedvr3b_profile_20260520.json`
    - `/Users/lixiaohui/Desktop/ppt_benchmark_single_video_timing_20260520/seedvr2_3b_profile_20260520.json`
- 新口径：
  - `启动/加载`：配置初始化、模型权重加载、pipeline/scheduler/prompt 等一次性准备。
  - `模型运算`：模型部署完成后的每视频推理链路，包括 LQ projector、DiT/sampler、VAE encode/decode、scheduler/noise/condition、必要的模型 device transfer。
  - `I/O 与后处理`：视频读取、resize、tensor 转视频、color fix、保存、cleanup。
  - PPT 只使用 `模型运算合计 / 输出帧数`。
- 新 PPT 计时数字：
  - FlashVSR official：`0.13 s/frame`
  - SeedVR-3B：`2.60 s/frame`
  - SeedVR2-3B：`0.62 s/frame`
  - LucidVR Stage1 535 10k：`2.27 s/frame`
  - LucidVR Stage1 USM-GT 3k：`2.27 s/frame`
  - LucidVR Stage2 641 6k：`1.60 s/frame`
  - LucidVR Stage3 D3.2 2k：`0.13 s/frame`

## 2026-05-19 v7-D4.4 重新 resume 到干净 W&B 时间线
- 背景：
  - 旧 run `yid6lzvt` 在同一个 W&B run 中经历过失败/残缺的 156-200 区间，W&B history 不会像 ckpt 文件一样自动覆盖旧点；
  - 用户要求重新 resume 一次，换新的实验名字和新的 W&B run，避免曲线混在同一个颜色/同一条 history 里。
- 旧残缺 ckpt 处理：
  - 在旧目录中把残缺 `step-200.safetensors` 改名备份：
    - 原路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1/output/step-200.safetensors`
    - 备份后：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1/output/step-200.incomplete_before_resume_20260519.safetensors`
    - 大小约 `1.1G`，时间 `May 18 11:48`。
- 第一次新名尝试：
  - run ts：`20260519_v7d44_resume_clean_from_step100`
  - W&B run id：`v7d44clean2`
  - resume state：旧目录 `output/training_state/step-100`
  - 失败原因：
    - 之前为了尝试训练内 online W&B，把 `HTTP_PROXY/HTTPS_PROXY/http_proxy/https_proxy=http://proxy.config.pcp.local:3128` 写进了 D4.4 训练启动脚本；
    - 训练数据读取的 conductor/notary/s3fs 也继承了代理，在 notary 临时凭证请求处报 `requests.exceptions.ProxyError`；
    - 进程退出，显存归零。
  - 结论：
    - W&B 同步需要代理，但训练进程不能默认带这组代理，否则会影响 conductor 数据读取；
    - 正确方式是训练保持 `wandb_mode=offline` 且无代理，外部 `wandb sync` tmux 单独带代理。
- 修正：
  - 撤销 D4.4 训练启动脚本里的代理导出：
    - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-48GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
  - 本地 `bash -n` 通过；
  - 6 台远端脚本 hash 均确认同步为：
    - `deb4eeb03f9a55d6c7b335a8833e23e9ab0dfe49b07e7e512370fc2784095734`
  - 更新 skill：`/Users/lixiaohui/.codex/skills/远程机器wandb/SKILL.md`，补充“训练代理和 conductor 的边界”。
- 第二次干净 resume，当前有效 run：
  - run name：
    - `train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260519_v7d44_resume_clean2_from_step100`
  - run dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260519_v7d44_resume_clean2_from_step100`
  - W&B run id：
    - `v7d44clean3`
  - W&B offline dir：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260519_v7d44_resume_clean2_from_step100/wandb/offline-run-20260519_004746-v7d44clean3`
  - resume state：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260518_v7d44_48gpu_fresh1/output/training_state/step-100`
  - 启动端口：
    - `MASTER_PORT=29568`
  - 启动时显式 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy`，避免 conductor/notary 走代理。
- 当前确认：
  - 6 节点 `v7d44_48gpu` tmux 已重新启动；
  - t5 进程数：`9`；
  - 已成功加载 state：
    - `[stage3c_resume] loaded student/fake state student_step=100 runner_step=500 epoch_id=0`
  - 已出现第一条训练 loss：
    - `[stage3c_train] epoch=0 step=101 runner_step=500 generator_update=1 dfake_gen_update_ratio=5 loss=0.933237 student=0.775047 fake_loss=0.03802867 fake_update=1 fake_scale=0.000000 dmd_student=0.120161 dmd_grad=0.367815 dmd_skip=0`
  - t5 显存回到训练态，约 `147-148GB/GPU`。
- W&B 同步：
  - 远端 t5 自动同步 tmux：
    - `wandb_sync_v7d44_clean2`
  - 当前使用硬编码 offline run 绝对路径，避免 glob/变量误同步旧 run：
    - `wandb sync --include-offline .../wandb/offline-run-20260519_004746-v7d44clean3`
  - 已确认：
    - `Syncing: https://wandb.ai/veralee/flashvsr/runs/v7d44clean3 ... done.`
- 安全记录：
  - 停旧训练只使用精确 tmux session：`tmux send-keys -t v7d44_48gpu C-c`；
  - 没有使用 `pkill`、空变量 kill 或模糊匹配 kill。

## 2026-05-20 v7-D4.4 clean2 再次退出的原因与 fake 侧同步修复
- 事故目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260519_v7d44_resume_clean2_from_step100`
  - 主日志：`run.log`
- 退出位置：
  - 最后一条正常训练日志停在 `runner_step=743, step=149`；
  - 之后没有进入 validation，也没有生成 step-200 ckpt；
  - 报错集中在 `fake_accelerator.backward(fake_loss)`；
  - DeepSpeed/ZeRO 在 `scaled_global_norm -> dist.all_reduce` 处触发 NCCL watchdog：
    - `PG ID 1`；
    - `OpType=ALLREDUCE`；
    - `NumelIn=287838720`；
    - `Timeout(ms)=600000`。
- 与 2026-05-18 旧失败不同：
  - 本次日志未发现 `SlowDown`、`OSError`、`DataLoader worker` 或 conductor/notary 数据读取异常；
  - 多台机器相互报告 `Connection closed by remote peer`，但沿 t5 -> gx2 -> ui9 -> g48 查下来，都是同一个 fake-side all-reduce timeout 的连锁后果。
- 关键定位：
  - D4.4 启动日志显示：
    - `fake_trainable_params=570961408`
    - `fake_trainable_groups={"lora": 283115520, "lq_proj_in": 287845888}`
  - NCCL timeout 的 `NumelIn=287838720` 与 fake 侧 `lq_proj_in` 参数组规模基本吻合；
  - 因为 D4.4 按 DMD2 语义每个 runner step 都更新 `G_fake`，fake 侧 `lq_proj_in` 每步都被 DeepSpeed/ZeRO 同步，造成 48 卡长训的高风险大集合通信。
- 修复：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
    - 新增参数 `--stage3_fake_train_lq_proj_in / --no-stage3_fake_train_lq_proj_in`；
    - 构造 trainable `G_fake` 时，如果该参数为 false，则冻结 `pipe.lq_proj_in.*`；
    - 启动日志新增：
      - `train_lq_proj_in=...`
      - `lq_proj_param_count=...`
  - 以下 config 设为 `stage3_fake_train_lq_proj_in: false`：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_48gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_smoke_2gpu_v7_d4_4_dualaccelerator_dfake5.yaml`
    - `wanvideo/model_training/flashvsr/configs/history/stage3_release_16gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dmdonly_dfake5.yaml`
- 语义边界：
  - student/generator 仍按原设定训练 LoRA + `lq_proj_in`；
  - frozen `G_real` 不训练；
  - trainable `G_fake` 改为 LoRA-only critic，仍使用当前 `z_pred.detach()` 做 FM loss，不回传 student；
  - 这比“完整 fake 也训练 lq_proj_in”少一部分 fake critic 自适应能力，但能去掉每个 runner step 上最重的一组 fake-side ZeRO 同步，是针对本次死锁/超时的必要稳定性修复。
- 验证：
  - 本地：
    - `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py` 通过；
- t5 远端：
    - 已确认代码和 config 同步到 `/mnt/task_runtime/lucidvsr`；
    - `/mnt/conda_envs/flashvsr/bin/python -m py_compile .../train_flashvsr_stage3_v7_d4_4_lora.py` 通过。

## 2026-05-20 更正：D4.4 最终不冻结 fake lq_proj_in，改为分频更新并启动 40 卡 fresh
- 上一节中曾记录“冻结 `G_fake.lq_proj_in` / fake LoRA-only critic”，这是中间方案，不是最终采用方案。
- 用户明确要求不要用“关掉 fake lq_proj_in”这种有语义风险的方式修稳定性问题；后续选择是：
  - `stage3_fake_train_lq_proj_in: true`
  - `stage3_fake_lq_proj_update_every_n_runner_steps: 5`
  - 即 `G_fake` 的 LoRA 每个 fake step 更新，`G_fake.lq_proj_in` 每 5 个 runner step 更新一次；
  - fake FM loss 仍使用当前 `z_pred.detach()`，不回传 student。
- 保留的工程优化：
  - fake 侧单独 DeepSpeed config：
    `wanvideo/model_training/flashvsr/lora/history/deepspeed_zero2_flashvsr_nooffload_fake_stable.json`
  - fake 侧 ZeRO bucket 下调到 `50000000`，并启用 fake engine 的 DeepSpeed activation checkpointing；
  - 启动脚本通过 `FLASHVSR_STAGE3_FAKE_DS_CONFIG` 指向该 fake stable config。
- 46/47 卡判断：
  - 当前 `accelerate launch --num_machines/--num_processes` 脚本假设每机 8 个本地进程；
  - a9 的 GPU3 已确认 CUDA 不可用，不能可靠作为 8 卡节点；
  - bfs 不在同一卡群，跨 bfs 凑 48 风险过高；
  - 因此放弃 48/bfs 组合，按 5 台原卡群机器启动 40 卡 fresh。
- 新增文件：
  - `wanvideo/model_training/flashvsr/lora/history/accelerate_zero2_flashvsr_5node40gpu_nooffload.template.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-40GPU-v7-D4-4-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- 新 40 卡 run：
  - run name：
    `train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
  - run dir：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
  - 节点：
    `t5qdtykjsw(rank0), 67dxkwcb7m(rank1), ui9n6p293s(rank2), g48bd6x4h7(rank3), gx2intv5rk(rank4)`
  - `MASTER_PORT=29573`
  - `WANDB_RUN_ID=v7d44_40g_freq5_1`
  - fresh 启动，不带 `--resume_training_state_dir`。
- 启动确认：
  - t5 启动日志确认：
    - `distributed_shape=5node40gpu`
    - `fake_lq_proj_trainable=True`
    - `fake_lq_proj_update_every_runner_steps=5`
    - `fake_ds_reduce_bucket=50000000`
    - `fake_ds_allgather_bucket=50000000`
  - W&B offline dir：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5/wandb/offline-run-20260519_131156-v7d44_40g_freq5_1`
  - t5 远端 W&B 同步 tmux：
    `wandb_sync_v7d44_40g_freq5`
  - 已确认同步成功：
    `Syncing: https://wandb.ai/veralee/flashvsr/runs/v7d44_40g_freq5_1 ... done.`
  - 40 卡启动脚本已补内置远端 W&B 自动同步逻辑：
    - `FLASHVSR_REMOTE_WANDB_SYNC=1` 默认开启；
    - 训练进程仍保持 offline 且无代理；
    - rank0 额外启动单独 tmux，只给 `wandb sync --include-offline` 导出代理；
    - 默认按 `WANDB_RUN_ID` 查找当前 run 的 offline 目录，避免误同步旧 run。
  - 已出训练日志：
    - `runner_step=0 step=1 loss=1.056189 student=0.991037 fake_loss=0.06515235 fake_lq_proj_update=1`
    - `runner_step=1 step=1 loss=0.031077 fake_loss=0.03107698 fake_lq_proj_update=0`
    - `runner_step=2 step=1 loss=0.046796 fake_loss=0.04679577 fake_lq_proj_update=0`
  - t5 训练态显存：
    - GPU1-7 约 `147.8GB`，利用率 `100%`；
    - GPU0 在第一步不同阶段约 `63GB-102GB`，利用率 `100%`。
- a9 状态：
  - GPU3 仍然 CUDA unavailable；
  - a9 不参与训练；
  - 已确认只占可用 GPU `0,1,2,4,5,6,7`，不触碰 GPU3；
  - `lxh_occupy_a9_good7` 中 7 张可用卡均约 `166604 MiB`、`100%`，GPU3 保持约 `4 MiB` 空闲。
  - 按用户要求又单独尝试 GPU3 占卡：
    - tmux：`lxh_occupy_a9_gpu3_try`
    - 命令：`gpu_stress_tc.py --gpus 3 --size 4096`
    - 结果：在 `torch.cuda.set_device(3)` 直接失败，报 `CUDA-capable device(s) is/are busy or unavailable`；
    - 结论：GPU3 当前不是普通空卡，而是 CUDA runtime 不可用，无法通过占卡程序拉起利用率。

## 2026-05-20 D4.2 W&B 日志字段对齐 D4.4
- 用户询问 D4.2 与跑到 2000 代的 D3.2 除 teacher 前/后 22 latent 对齐和 teacher checkpoint 之外还有什么差异，并要求把 D4.4 当前更清晰的 W&B loss 记录同步到 D4.2。
- D3.2 与 D4.2 的关键差异：
  - D3.2 是每个训练 step 同时做 student/generator 与 fake critic 更新，`total_loss = student_loss + fake_loss + dmd_student_loss` 后一起走 `accelerator.backward(total_loss)`；
  - D4.2 改为 dfake=5 的 single-runner 组织：每个 runner step 都用当前 `z_pred.detach()` 更新 `G_fake`，student/generator 只在每 5 个 runner step 更新一次；
  - D4.2 的 fake FM backward 与 student/generator backward 被拆开，fake loss 不回传 student；
  - D4.2 的 checkpoint/save/validation 跟随 generator turn，也就是按 student 更新步计数；W&B x 轴应使用 runner step，否则 5 个 runner 子步会挤在同一个 global step 上。
- 已修改：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py`
- 修改内容：
  - W&B payload 改为和 D4.4 一致的清爽字段：
    - `train/loss`
    - `train/fake_fm_loss`
    - `train/generator_did_update`
    - `train/runner_step`
    - `train/flow_loss`
    - `train/mse_loss`
    - `train/lpips_loss`
    - `train/dmd_student_loss`
  - W&B step 从 `global_step` 改成 `current_runner_step`，避免 dfake=5 下同一 global step 被多个 fake-only runner step 覆盖。
- 验证：
  - 本地 `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_2_lora.py` 通过。

## 2026-05-22 v7-D4.4 Flow0.1 新 checkpoint 补测
- 目标实验：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_flow0p1_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260521_111700`
- 已有本地结果：
  - `/Users/lixiaohui/Desktop/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1/step-50`
  - `/Users/lixiaohui/Desktop/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1/step-100`
- 新增 checkpoint：
  - `step-150.safetensors`
  - `step-200.safetensors`
  - `step-250.safetensors`
- 操作：
  - 在训练机 `bfs6vaz4d6` 将新增 ckpt 上传到：
    `s3://lxh/tmp/stage3_v7d44_flow0p1_ckpts_20260522/`
  - 在 4 卡测试机 `6ikhpjzv3z` 使用 Stage3 one-step streaming/KV-cache 推理脚本补测新增 ckpt；
  - 已测过的 `step-50/100` 自动跳过；
  - 测试期间 GPU0-2 跑 `step-150/200/250`，GPU3 单独占卡；
  - 测试结束后已启动 `occupy_after_flow0p1_eval`，4 卡恢复占卡。
- 远端输出：
  - `/mnt/task_wrapper/user_output/artifacts/inference/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
  - S3：`s3://lxh/data/test/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
- 本地下载：
  - `/Users/lixiaohui/Desktop/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
- 本地核对：
  - `step-50/100/150/200/250` 各 10 个视频；
  - `flow0p1` 下共 50 个 mp4。

## 2026-05-22 Stage3 D3.2 / D4.4 / Flow0.1 10 synthetic 对比目录整理
- 用户要求：
  - 将 D4.4 主线、D4.4 flow0.1 对照、D3.2 历史结果整理到同一个干净目录；
  - 文件名直接带模型标识，便于肉眼对比；
  - 检查 D3.2 是否已经测到 `step-2000`，如缺失则补测。
- 检查结果：
  - D3.2 本地已有完整 10 synthetic 结果：
    `/Users/lixiaohui/Desktop/stage3/stage3_v7_d3_2_scan89_100plus_synthetic_20260516_6ikh_nontiled_g01`
  - D3.2 已包含 `step-100/200/500/1000/1500/2000`，每个 step 10 个视频；
  - D4.4 主线桌面原有 `step-100/200/300`，远端最新多出 `step-400`。
- D4.4 `step-400` 补测：
  - 远端实验：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
  - 从 `t5qdtykjsw` 上传 ckpt 到：
    `s3://lxh/tmp/stage3_v7_d4_4_40gpu_fresh_ckpts_20260522/step-400.safetensors`
  - 在 4 卡测试机 `6ikhpjzv3z` 使用 Stage3 one-step streaming/KV-cache 推理脚本补测 `step-400`；
  - GPU0 跑测试，GPU1-3 单独占卡；
  - 测完后已恢复 `occupy_after_d44_step400_eval`，4 卡 100% 占用。
- D4.4 `step-400` 本地同步：
  - `/Users/lixiaohui/Desktop/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu/step-400`
  - 已核对 `step-100/200/300/400` 各 10 个视频。
- 新整理目录：
  - `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522`
- 整理规则：
  - 按视频名分组到 `by_video/<video>/`；
  - 文件名带模型标识和 step，例如：
    `d44_flow1p0_40gpu__step-0400__takano_00.mp4`
  - `index.tsv` 记录每个重命名文件对应的原始来源。
- 模型标识：
  - `d32_48gpu`：v7-D3.2 48GPU baseline，`step-100/200/500/1000/1500/2000`；
  - `d44_flow1p0_40gpu`：v7-D4.4 40GPU 主线，`step-100/200/300/400`；
  - `d44_flow0p1_16gpu`：v7-D4.4 flow0.1 对照，`step-50/100/150/200/250`。
- 本地核对：
  - 10 个视频目录；
  - 每个视频 15 个 mp4；
  - 总计 150 个 mp4。

## 2026-05-20 D44版本2 DMD student loss 尖刺复查
- 这里的 D44版本2 指 2026-05-20 启动的 40 卡 fresh run：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
- 只读解析 t5 主节点 `run.log` 中 88 条含 `dmd_student` 的 generator turn：
  - `dmd_student`：max `19.620518`，mean `0.662964`，median `0.2221595`；
  - `dmd_student` p90 `0.9045195`，p95 `2.0932245`；
  - `dmd_student > 2` 共 4 次，`>5` 共 1 次，`>10` 共 1 次；
  - `dmd_grad`：max `4.870988`，mean `0.682862`，median `0.494514`；
  - `dmd_skip` 总和为 `0`。
- 最大尖刺发生在 runner 295 / student step 60：
  - `loss=20.099821`
  - `student=0.376212`
  - `fake_loss=0.10309093`
  - `dmd_student=19.620518`
  - `dmd_grad=4.870988`
  - `dmd_skip=0`
- 结论：
  - 现有防护只看 `dmd_grad.abs().mean()` 是否超过 `stage3_dmd_grad_norm_max=5.0`；
  - 最大尖刺的 `dmd_grad=4.870988` 没过 5.0，因此没有触发 skip；
  - 但 DMD loss 是平方项，`0.5 * mean(dmd_grad^2)`，少数位置较大的 DMD grad 会让 loss 尖刺明显高于 mean-abs norm。
- 建议下一版优先加“DMD loss-level clamp”，而不是只把 mean-abs grad 阈值从 5 调低：
  - 保留现有 `stage3_dmd_grad_norm_max=5.0` 作为硬异常/NaN 防线；
  - 新增例如 `stage3_dmd_loss_max=2.0` 或 `3.0`；
  - 当 `dmd_loss > loss_max` 时，按 `sqrt(loss_max / dmd_loss)` 缩放 `dmd_grad`，保留方向但限制 DMD 对 student 的单步贡献；
  - 这样只影响当前 run 中约 4 个异常点，不会压掉 median/p90 附近的正常 DMD 信号。

## 2026-05-20 D4.4 加入 DMD loss-level clamp
- 按上面对 D44版本2 的 run.log 复查结论，已在 D4.4 训练代码中加入 DMD loss-level clamp。
- 修改文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py`
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- 新增参数：
  - `--stage3_dmd_loss_max`
  - 默认 `0.0`，不影响未显式开启的旧配置；
  - D44版本2 后续/重启用的 40 卡 config 显式设置为 `stage3_dmd_loss_max: 3.0`。
- 实现逻辑：
  - 保留原有 `stage3_dmd_grad_norm_max=5.0` 与 `stage3_dmd_spike_policy=skip`；
  - 在构造 DMD target 前计算未加权 DMD loss：`0.5 * mean(dmd_grad^2)`；
  - 若该值超过 `stage3_dmd_loss_max`，按 `sqrt(loss_max / dmd_loss)` 缩放 `dmd_grad`；
  - 这样保留 DMD 方向，但限制异常 batch 对 student 的单步梯度冲击。
- 日志：
  - run.log 新增 `dmd_loss_clamp=0/1`；
  - W&B 不新增该线，避免监控曲线变复杂，仍主要看 5 个 loss。
- 验证：
  - 本地 `python -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py` 通过。

## 2026-05-20 v7-D4.4 16卡 Ownership 梯度归属验证
- 使用 `bfs6vaz4d6` + `i6hf4scd4y` 共 16 卡跑现有 gradcheck：
  - 代码：`wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py`
  - 脚本：`wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh`
  - 配置基线：`wanvideo/model_training/flashvsr/configs/history/stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
- 第一次失败原因：
  - `i6hf4scd4y` 仍有旧 `occupy_all/gpu_stress` 占满显存；
  - DeepSpeed ZeRO2 初始化 optimizer buffer 时 OOM；
  - 清理真实 GPU PID 后重跑成功。
- 成功实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradcheck_16gpu_v7_d4_4_Ownership_20260520_v1b_ownership_16gpu_clean`
- 验证结果：
  - 首帧 window 强制生效：`latent_window=[0,2)`、`frame_window=[0,5)`；
  - `first_frame_pixel_weight=4.0` 和 `first_frame_lpips_weight=4.0` 生效；
  - fake-only turn 没有更新 student：`student_delta=0`；
  - `G_real` 没有更新：`real_delta=0`；
  - generator turn 里出现 `fake_delta`，复核源码后确认是 D4.4 当前设计：每个 runner step 都更新 fake，student/generator 每 5 个 runner step 更新一次。
- 命名修正：
  - 当前日志里的 `generator_turn/generator_update` 更准确应理解为 `student_update_turn`；
  - 它表示“这一轮 student 也更新”，不是“这一轮 fake 不更新”。
- 配置补充：
  - `stage3_fake_lq_proj_update_every_n_runner_steps=5`，所以 fake 侧 `lq_proj_in` 正好也在 runner 0/5/10 这些 student update turn 更新；
  - 这会让这些 turn 的 `fake_delta` 更明显，属于预期。
- 收尾：
  - 验证结束后已在两台 16 卡机器上恢复正常占卡，`bfs6vaz4d6` 与 `i6hf4scd4y` 均为 8/8 卡 100%。
- 详细分析已追加到：
  - `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`

## 2026-05-20 Gemini Stage3 审阅意见复核
- 复核文件：`/Users/lixiaohui/Desktop/gemini_FlashVSR_Stage3_Review.md`
- 结论：
  - LPIPS / DMD / Flow 梯度量级失衡：有道理，不能只看 loss 标量，需要新增同 batch 分 loss backward，直接量 `z_pred.grad`、LoRA grad、LQ projector grad。
  - G_fake 梯度累加：当前 `gradient_accumulation_steps=1`，所以不影响当前 D4.4；未来如果开 accumulation，需要给 fake 分支加 `fake_accelerator.accumulate(fake_model)` 并重写 zero_grad/step 条件。
  - LPIPS 输入值域：Gemini 这条基本误判。Wan VAE decode 输出 `[-1,1]`，GT 经过 `preprocess_video` 也映射到 `[-1,1]`；debug dump clamp 到 `[0,1]` 只是保存可视化。
  - 三模型共存显存压力：属实，但这是已知工程压力，不是新 bug。
- 详细分类和新增验证计划已写入：
  - `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`

## 2026-05-20 v7-D4.4 GradScale 梯度量级验证
- 为复核 Gemini 提出的 `LPIPS / DMD / Flow` 梯度量级失衡问题，新增独立 gradcheck case，不改正式 D4.4 训练文件。
- 修改文件：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh`
- 本地验证：
  - `python3 -m py_compile wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_gradcheck_lora.py` 通过；
  - `bash -n wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-GradCheck-16GPU-v7-D4-4.sh` 通过。
- 远端验证：
  - 机器：`bfs6vaz4d6` + `i6hf4scd4y`，共 16 卡；
  - 第一次 `max_train_steps=1` 只看到 runner 0，DMD 为 0，因为初始 `G_real/G_fake` 尚未分化；
  - 第二次改为 `max_train_steps=2`，让 fake 先更新到 runner 5 后再读 DMD 梯度；
  - 有效目录：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_gradscale_16gpu_v7_d4_4_20260520_1640_gradscale_16gpu_step2`。
- 有效结果 runner 5：
  - `flow_to_noise_pred_mean_abs=7.528200e-09`；
  - `mse_to_z_pred_mean_abs=6.975461e-08`；
  - `lpips_to_z_pred_mean_abs=6.729997e-06`；
  - `dmd_to_z_pred_mean_abs=9.535980e-08`；
  - `dmd_student=0.23425940`，`dmd_raw_grad_mean_abs=0.515619`，说明 DMD 分支已非零。
- 结论：
  - 当前权重下 LPIPS 对 `z_pred` 的 mean-abs 梯度约为 MSE 的 `~96x`、DMD 的 `~71x`；
  - Gemini 关于梯度量级失衡的担心成立；
  - 下一步建议做 `lpips_weight=2/1/0.5/0` 的短训 ablation，看 ghost、运动残影和纹理稳定性。
- 收尾：
  - 验证完成后已在 `bfs6vaz4d6` 与 `i6hf4scd4y` 恢复 16 卡占卡，均为 8/8 卡 100%。
  - 详细记录已写入 `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`。

## 2026-05-20 Gemini 新问题复核：LPIPS λ、值域、accumulation
- 复核 FlashVSR 论文：
  - Stage3 目标明确写 `λ = 2`；
  - 当前 D4.4 的 `stage3_lpips_weight: 2.0` 与论文一致。
- 复核 OSEDiff：
  - `OSEDiff/osediff.py` 中 VAE decode 输出 `clamp(-1, 1)`；
  - `OSEDiff/train_osediff.py` 中 `lambda_lpips` 默认也是 `2.0`。
- 复核当前 D4.4：
  - `x_pred` 由 Wan VAE decode 返回 `[-1,1]`；
  - `x_gt` 由 `pipe.preprocess_video(..., min_value=-1, max_value=1)` 得到 `[-1,1]`；
  - 因此 Gemini 关于 `[0,1]` LPIPS 输入错误的担心不适用于当前 D4.4。
- 复核 gradient accumulation：
  - 当前 40GPU YAML 是 `gradient_accumulation_steps: 1`；
  - 与 DMD2 `assert gradient_accumulation_steps == 1` 的约束一致；
  - 当前不存在 accumulation 污染 G_fake/student 更新的问题。
- 更新结论：
  - LPIPS 梯度大是真现象，但不是值域 bug；
  - GradScale 是强制首帧 window，且首帧 LPIPS x4，因此是偏 worst-case 的量级；
  - 用户反馈 ghost 已基本解决，当前不建议立刻偏离论文降低 `λ=2`；
  - 如后续 ghost/残影复现，再做 `lpips_weight=2/1/0.5/0` 短训 ablation 或引入 LPIPS gradient cap。
- 详细记录追加到：
  - `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`

## 2026-05-20 DMD mean reduction 复核
- 复核 DMD2 本地代码 `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/sd_guidance.py`：
  - DMD2 同样使用 `(p_real - p_fake) / mean(abs(p_real))`；
  - DMD2 同样使用 `0.5 * F.mse_loss(..., reduction="mean")`。
- 结论：
  - 当前 D4.4 的 DMD 除法归一化和 mean reduction 不是实现 bug；
  - GradScale 中 LPIPS 比 DMD 大很多，说明首帧 window 下 LPIPS 会主导 `z_pred` 局部梯度，但不说明 DMD 公式错；
  - 当前如果 ghost 已解决，不建议马上偏离论文改 `stage3_lpips_weight=2.0` 或 DMD reduction；
  - 若后续要判断 DMD 是否真正影响 trainable 参数，下一步应做 per-loss parameter grad 验证，分别统计 LoRA / LQ projector 的 flow、MSE、LPIPS、DMD grad norm。
- 详细记录追加到：
  - `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`

## 2026-05-21 v7-D4.4 Per-Loss Parameter Grad 验证
- 目的：
  - 继续复核 Gemini 提出的 `LPIPS / DMD / Flow` 梯度量级失衡；
  - 这次不只看 `z_pred.grad`，而是直接量可训练参数上的 per-loss grad norm。
- 实现方式：
  - 未改正式训练代码；
  - 复制独立验证脚本 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_paramgrad_lora.py`；
  - 复制独立启动脚本 `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-ParamGrad-2GPU-v7-D4-4.sh`；
  - 诊断模式使用 `1GPU + no DeepSpeed + small shape`，因为 ZeRO2 下直接读 `.grad` 不可靠，`safe_get_full_grad` 太慢。
- 有效远端 case：
  - Flow：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_paramgrad_1gpu_v7_d4_4_20260521_0035_paramflow_1gpu_small_nods_fixedpy`
  - MSE：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_paramgrad_1gpu_v7_d4_4_20260521_0045_ParamMSE_1gpu_small_nods_fixedpy`
  - LPIPS：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_paramgrad_1gpu_v7_d4_4_20260521_0045_ParamLPIPS_1gpu_small_nods_fixedpy`
  - DMD：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_paramgrad_1gpu_v7_d4_4_20260521_0045_ParamDMD_1gpu_small_nods_fixedpy`
- 关键结果：
  - all sampled params L2：Flow `1.7389`，MSE `0.0602`，LPIPS `1.5213`，DMD `0.0267`；
  - LoRA L2：Flow `1.3424`，MSE `0.0382`，LPIPS `0.6436`，DMD `0.0117`；
  - LQ projector L2：Flow `1.1053`，MSE `0.0466`，LPIPS `1.3785`，DMD `0.0240`。
- 结论：
  - LPIPS / DMD 在参数层面也明显失衡，LPIPS/DMD 约 `55-57x`；
  - Flow 也明显强于 DMD，LoRA 上 Flow/DMD 约 `114x`；
  - 普通 `clip_grad_norm_` 只能防爆，不能自动平衡各 loss 的贡献；
  - 不建议直接上 GradNorm algorithm，因其会偏离论文设定并可能削弱 LPIPS 的细节约束。
- 下一步建议：
  - 做短训 ablation，而不是直接改正式 D4.4：
  - 优先试 `stage3_dmd_weight=5/10/20` 或 `stage3_flow_weight=0.1/0.3`；
  - 只有 ghost/残影复现时再试 `stage3_lpips_weight=2/1/0.5/0`。
- 收尾：
  - `bfs6vaz4d6` 与 `i6hf4scd4y` 均已恢复 8/8 卡 100% 占卡；
  - 详细记录已追加到 `doc/flashvsr_stage3_v7d44_fresh_review_20260520.md`。

## 2026-05-21 v7-D4.4 Flow 降权 16GPU 对照实验启动
- 目的：
  - 针对 D4.4 中 Flow / LPIPS 对 trainable 参数梯度明显强于 DMD 的现象，先做最小干预对照；
  - 只把 `stage3_flow_weight` 从 `1.0` 降到 `0.1`；
  - 其余主要权重保持：`stage3_mse_weight=1.0`，`stage3_lpips_weight=2.0`，`stage3_dmd_weight=1.0`，`stage3_fake_fm_weight=1.0`。
- 公平对齐说明：
  - 当前只有 16 卡可用，原 D4.4 是 40 卡；
  - 在 `batch_size=1`、`gradient_accumulation_steps=1` 下，40 卡 `student step=100/200` 约等价 16 卡 `student step=250/500`；
  - 因此本轮设置 `max_train_steps=520`，保存 `250/500` 作为主要对比点，同时保留 `100/200` 观察同更新次数动态。
- 新增文件：
  - `wanvideo/model_training/flashvsr/configs/history/stage3_release_16gpu_v7_d4_4_flow0p1_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb.yaml`
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-Release-16GPU-v7-D4-4-Flow0p1-Lora-89f-VideoOnly-USMGTPretrain-DualAccelerator-Dfake5-OfflineWandb.sh`
- 远端机器：
  - rank0：`bfs6vaz4d6`
  - rank1：`i6hf4scd4y`
- 远端实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_16gpu_v7_d4_4_flow0p1_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260521_111700`
- 远端启动确认：
  - 两台机器同步检查通过，远端文件均存在；
  - 已停止两台机器的占卡程序后启动训练；
  - 已出第一条完整训练日志：
    - `step=1 runner_step=0 generator_update=1 loss=0.200127 student=0.182379 fake_loss=0.01774874`
    - `stage3_v7_b_loss loss=0.182379 flow=0.103846 mse=0.000221 lpips=0.085886`
  - 该数值验证 `flow0p1` 已生效：`0.1 * 0.103846 + 1.0 * 0.000221 + 2.0 * 0.085886 ~= 0.182379`。
- 输出：
  - 已保存 `output/step-1.safetensors`；
  - wandb offline package tmux：`wandb_package_v7d44_flow0p1_16gpu`。
- 空卡保护：
  - 已在 `bfs6vaz4d6` 和 `i6hf4scd4y` 各启动 `tmux occupy_after_flow0p1`；
  - 该 monitor 会在 `train_flashvsr_stage3_v7_d4_4_lora.py` 进程结束后自动启动 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`，避免 520 step 完成后 16 卡空置。

## 2026-05-22 Stage3 三组模型 10synthetic 补测与桌面 clean 目录重排
- 按用户要求补测最新 checkpoint：
  - D4.4 40GPU 主实验补测 `step-500`；
  - D4.4 flow0p1 16GPU 对照补测 `step-300`、`step-350`；
  - D3.2 已有 `step-100/200/500/1000/1500/2000`，无需重复测试。
- 4 卡测试机：
  - `6ikhpjzv3z`；
  - 测试时 D4.4 `step-500` 使用 GPU0，flow0p1 `step-300/350` 使用 GPU1/2，GPU3 保持占卡；
  - 测试完成后启动 `tmux occupy_after_stage3_latest_eval`，4 张卡均恢复 100% 占卡。
- 远端输出与 S3：
  - D4.4：`s3://lxh/data/test/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`
  - flow0p1：`s3://lxh/data/test/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
- 本机同步完成：
  - `/Users/lixiaohui/Desktop/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`
  - `/Users/lixiaohui/Desktop/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
- 重新整理 clean 对比目录：
  - `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522`
  - 新结构改为按模型分目录，再按 `step + 模型标识` 分子目录；
  - 视频文件名保持原始名字，不再额外加模型前缀。
- clean 目录计数：
  - `d32_48gpu`: `step100_d32/step200_d32/step500_d32/step1000_d32/step1500_d32/step2000_d32`，共 60 个 mp4；
  - `d44_flow1p0_40gpu`: `step100_d44/step200_d44/step300_d44/step400_d44/step500_d44`，共 50 个 mp4；
  - `d44_flow0p1_16gpu`: `step50_flow01/step100_flow01/step150_flow01/step200_flow01/step250_flow01/step300_flow01/step350_flow01`，共 70 个 mp4；
  - 总计 180 个 mp4。

## 2026-05-22 Stage3 最新 checkpoint 继续补测
- 按用户要求继续补测最新 checkpoint，并追加到同一个桌面 clean 目录：
  - D4.4 40GPU 主实验补测 `step-600`；
  - D4.4 flow0p1 16GPU 对照补测 `step-400`、`step-450`。
- 4 卡测试机：
  - `6ikhpjzv3z`；
  - 新容器缺少 `conductor`、`flashvsr` 环境依赖、`/mnt/models/Wan2.1-T2V-1.3B`、`imageio-ffmpeg`、`block_sparse_attn` CUDA 扩展；
  - 已在远端补齐 `conductor`、最小 Python 依赖、Wan 1.3B 基座模型、`Block-Sparse-Attention` cutlass 子模块，并编译安装 `block_sparse_attn`；
  - 推理启动时显式设置 `LD_LIBRARY_PATH=/miniforge/envs/b200/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64`，保证 `block_sparse_attn_cuda` 能找到 `libc10.so`。
- 测试设置：
  - 使用 `wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`；
  - D4.4 `step-600` 使用 GPU0；
  - flow0p1 `step-400/450` 使用 GPU1/2；
  - 推理仍为 Stage3 one-step + Stage2 streaming/KV-cache path，`stage2_attention_mode=block_sparse_chunk_causal`，`topk_ratio=2.0`，`kv_ratio=3.0`，`tiled=0`。
- 远端输出与 S3：
  - D4.4：`s3://lxh/data/test/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`
  - flow0p1：`s3://lxh/data/test/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`
- 本机下载与 clean 目录：
  - 本机 staging：`/Users/lixiaohui/Desktop/_stage3_latest_sync_20260522`
  - clean 目录：`/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522`
  - 新增 `d44_flow1p0_40gpu/step600_d44`，10 个 mp4；
  - 新增 `d44_flow0p1_16gpu/step400_flow01`，10 个 mp4；
  - 新增 `d44_flow0p1_16gpu/step450_flow01`，10 个 mp4；
  - clean 总计更新为 210 个 mp4。
- 空卡保护：
  - 测试结束后远端自动启动 `tmux occupy_after_stage3_latest_eval2`；
  - `6ikhpjzv3z` 4 张卡均恢复 100% 占卡。

## 2026-05-23 4 卡测试机恢复与 flow0p1 step-500 补测
- 按用户要求先恢复本机远程工作流：
  - 本机 tmux 仅保留 `lxh`、`lxh1`、`sync`、`watch`、`gemini`；
  - `sync` 中补回 `6ikhpjzv3z`，命令为 `bolt task sync 6ikhpjzv3z`；
  - `watch` 中保留 48 卡 6 节点、16 卡 2 节点和 `6ikhpjzv3z` 的 `nvidia-smi` 监控窗口。
- 4 卡测试机 `6ikhpjzv3z` 状态：
  - task 仍在，但 pod 重建导致本地 `/mnt/task_wrapper/user_output/artifacts` 和部分环境状态丢失；
  - 在远端可见 tmux 窗口中执行 `bash /mnt/task_runtime/bolt_lxh/setup_after_docker1.sh`，脚本正常退出 `SETUP_EXIT_0`；
  - setup 恢复 `/mnt/conda_envs/flashvsr`、`/mnt/conda_envs/seedvr`、`/mnt/conda_envs/DiffVSR_b200`，并从 `s3://lxh/models/SR/` 同步模型到 `/mnt/models/`。
- artifacts 恢复来源：
  - 从 `s3://bolt-prod-2320845741/tasks/6ikhpjzv3z/artifacts` 拉回常用 10 synthetic 测试集和历史输出；
  - 恢复后自检通过：
    - `testset10_89f_aliyun_light_x4_lq_20260503/lq`: 10 个 mp4；
    - `stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1`: 90 个历史 mp4；
    - `stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`: 60 个历史 mp4；
    - `/mnt/conda_envs/flashvsr/bin/python` 中 `cv2`、`imageio`、`block_sparse_attn` 均可 import。
- 最新 checkpoint 扫描：
  - D4.4 40GPU 主实验仍只保存到 `step-600.safetensors`，当前 run log 约 `student step=666`，暂无新保存 ckpt；
  - flow0p1 16GPU 对照已保存到 `step-500.safetensors`，需要补测。
- flow0p1 `step-500` 补测：
  - ckpt 来源：`s3://lxh/tmp/stage3_v7d44_flow0p1_ckpts_20260522/step-500.safetensors`；
  - 测试脚本：`wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`；
  - 推理设置保持 Stage3 one-step + Stage2 streaming/KV-cache path：
    - `num_inference_steps=1`；
    - `stage2_attention_mode=block_sparse_chunk_causal`；
    - `stage2_topk_ratio=2.0`；
    - `stage2_kv_ratio=3.0`；
    - `tiled=0`；
    - `color_fix_method=adain`；
    - 输入 89 帧，输出 85 帧。
  - 测试时 GPU1 跑推理，GPU0/2/3 临时占卡；测试完成后恢复全 4 卡占卡。
- 输出与下载：
  - 远端输出：`/mnt/task_wrapper/user_output/artifacts/inference/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1/step-500`，10 个 mp4；
  - S3 输出：`s3://lxh/data/test/stage3_flow0p1_vs_v7d32_10synthetic_20260521/flow0p1/step-500`；
  - 本机 clean 目录新增：`/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow0p1_16gpu/step500_flow01`，10 个 mp4；
  - clean 目录总计更新为 220 个 mp4。
- 空卡保护：
  - 测试结束后远端自动启动 `tmux occupy_after_test`；
  - `6ikhpjzv3z` 4 张卡确认均为 100% GPU 利用率，占用约 125GB 显存。

## 2026-05-23 Stage3 OF-fast 过拟合验证缓存修复

- 问题：
  - OF-A/B/C/D v4 过拟合验证虽然固定了 4 个视频 manifest 和随机 seed，但每个 runner step 仍从 DataLoader 重新读取/退化/准备视频；
  - 日志显示 `data` 经常为几十到几百秒，导致“过拟合 debug”本身过慢，无法快速判断三阶段 loss 方向。
- 修复：
  - 只在 overfit 专用文件 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py` 中加入 `FLASHVSR_STAGE3_OVERFIT_CACHE_FIRST_BATCH=1`；
  - 每个 rank 第一次取到训练 batch 后缓存到内存，后续 runner step 直接复用；
  - 通用 D4.4 源码 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_lora.py` 未修改。
- 启动：
  - bfs `0-3`: `stage3_OF_A_fast_4gpu_overfit4_full_v7_d4_4_20260523_of_a_fast_v5`
  - bfs `4-7`: `stage3_OF_B_fast_4gpu_overfit4_recononly_v7_d4_4_20260523_of_b_fast_v5`
  - i6 `0-3`: `stage3_OF_C_fast_4gpu_overfit4_flowonly_v7_d4_4_20260523_of_c_fast_v5`
  - i6 `4-7`: `stage3_OF_D_fast_4gpu_overfit4_dmdfakeonly_v7_d4_4_20260523_of_d_fast_v5`
- 结果：
  - 四组均正常进入训练并输出 `stage3c_train` / `stage3_timing`；
  - 首个 runner step 的 `data` 仍高，因为要读首 batch；
  - 第二个 runner step 起 `data` 稳定约 `0.10-0.15s`，说明重复数据读取瓶颈已消除；
  - 当前四组继续运行，退出后脚本会在对应 GPU 范围内自动启动占卡。

## 2026-05-23 GPU 空卡自动占卡 guard

- 新增通用脚本：
  `wanvideo/model_training/flashvsr/scripts/gpu_empty_guard_start_occupy.sh`
- 功能：
  - 监控指定 `GPU_IDS`；
  - 只有当某张 GPU 同时满足 `memory.used <= 1024MB` 且 `utilization <= 5%`，并连续空闲 `180s` 后，才启动正常占卡；
  - 占卡按空闲 GPU 子集启动，不会因为某个 rank0 短暂低 util 误判训练卡为空；
  - 默认使用 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`，不使用低显存占卡。
- 接入：
  - `FlashVSR-Stage3-OF-Fast-4GPU-v7-D4-4.sh` 新增 `START_EMPTY_GPU_GUARD=1`；
  - 每个 OF 实验启动时自动创建一个远端 `gpu_empty_guard_<OF_ID>` tmux；
  - 原来的 `trap finish EXIT` 仍保留，程序正常退出或报错退出时会立即在该实验的 `CUDA_VISIBLE_DEVICES` 范围启动占卡。
- 当前运行中的 OF v5 是接入前启动的，因此已手动补挂 guard：
  - bfs: `gpu_empty_guard_OF_A_fast_manual`, `gpu_empty_guard_OF_B_fast_manual`
  - i6: `gpu_empty_guard_OF_C_fast_manual`, `gpu_empty_guard_OF_D_fast_manual`
  - guard 日志写入各自实验目录的 `gpu_empty_guard.manual.log`。

## 2026-05-24 16 卡节点重初始化与 D4.4 step700/800 补测

- 背景：
  - 16 卡节点 `bfs6vaz4d6`、`i6hf4scd4y` 被抢占后重启，远端 artifacts 被清空；
  - 4 卡测试机 `6ikhpjzv3z` 暂时不可用，后续 validation/test 暂时改在 `bfs6vaz4d6` 上执行。
- 初始化：
  - 对 `bfs6vaz4d6`、`i6hf4scd4y` 执行 `bolt task patchsshconfig`；
  - 在远端运行 `bash /mnt/task_runtime/bolt_lxh/setup_after_docker1.sh`，两台均确认 `SETUP_EXIT_0`；
  - 验证 `/mnt/conda_envs/flashvsr/bin/python`、`torch 2.9.1+cu128`、`block_sparse_attn` 可用。
- sync/watch：
  - `sync` tmux 保留 `bfs6vaz4d6`、`i6hf4scd4y`、`6ikhpjzv3z` 和 48 卡 6 节点；
  - 重建 `watch` tmux，窗口顺序固定为：
    `0-5` 为 48 卡节点，`6` 为 `bfs6vaz4d6`，`7` 为 `i6hf4scd4y`。
- artifacts 恢复：
  - 从 `s3://bolt-prod-2320845741/tasks/6ikhpjzv3z/artifacts` 恢复到 `bfs6vaz4d6`：
    - 常用 10 synthetic 测试集 `testset10_89f_aliyun_light_x4_lq_20260503`；
    - 历史 D4.4 测试输出 `stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521`。
  - 从 `s3://bolt-prod-2320845741/tasks/bfs6vaz4d6/artifacts` 轻量恢复：
    - `train_stage3_release_16gpu_v7_d4_4_flow0p1_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260521_111700`；
    - `stage3_OF_A_fast_4gpu_overfit4_full_v7_d4_4_20260523_of_a_fast_v5`；
    - `stage3_OF_B_fast_4gpu_overfit4_recononly_v7_d4_4_20260523_of_b_fast_v5`。
  - 从 `s3://bolt-prod-2320845741/tasks/i6hf4scd4y/artifacts` 轻量恢复：
    - `stage3_OF_C_fast_4gpu_overfit4_flowonly_v7_d4_4_20260523_of_c_fast_v5`；
    - `stage3_OF_D_fast_4gpu_overfit4_dmdfakeonly_v7_d4_4_20260523_of_d_fast_v5`。
  - 恢复策略：只拉 logs、snapshot yaml/sh、`output/step-*.safetensors`、validation/inference mp4/json；跳过 `training_state`、`stage3_fake_deepspeed`、optimizer `.pt`、random state `.pkl` 等大断点。
- D4.4 40GPU step700/800 补测：
  - checkpoint 来源：`s3://lxh/tmp/stage3_v7_d4_4_40gpu_fresh_ckpts_20260522`；
  - 测试机器：`bfs6vaz4d6`，GPU0/1 跑测试，GPU2-7 临时占卡；
  - 测试脚本：`wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`；
  - 推理设置保持 Stage3 one-step + Stage2 streaming/KV-cache path，`tiled=0`；
  - `step-700`、`step-800` 均完成 10 个 synthetic 视频输出，并同步到：
    `s3://lxh/data/test/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`。
- 本机下载：
  - 下载到：
    `/Users/lixiaohui/Desktop/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`；
  - 同步整理到 clean 对比目录：
    `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow1p0_40gpu/step700_d44`；
    `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow1p0_40gpu/step800_d44`；
  - 两个目录均确认各 10 个 mp4。
- 空卡保护：
  - D4.4 step700/800 测试结束后，`bfs6vaz4d6` 自动启动全卡正常占卡；
  - `i6hf4scd4y` 也重启全卡正常占卡；
  - 当前 watch 可见两台 16 卡节点均 8 卡 100% 利用率。

## 2026-05-25 D4.4 40GPU step900/1000/1100 补测

- 目标：
  - 按用户要求继续补测 D4.4 40GPU 主实验最新 checkpoint。
- checkpoint 状态：
  - 远端实验目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
  - t5 上扫描到最新 ckpt 已到 `step-1100.safetensors`；
  - 日志中 `fake_loss` 和总 `loss` 已为 `nan`，但 `stage3_v7_b_loss` 的 student 重建分支仍在输出有限值，因此继续测 `step-900/1000/1100` 用于视觉判断。
- checkpoint 中转：
  - 将 `step-900.safetensors`、`step-1000.safetensors`、`step-1100.safetensors` 上传到：
    `s3://lxh/tmp/stage3_v7_d4_4_40gpu_fresh_ckpts_20260522`。
- 测试：
  - 测试机器：`bfs6vaz4d6`；
  - GPU0/1 跑测试，GPU2-7 临时正常占卡；
  - 测试脚本：
    `wanvideo/model_inference/flashvsr/history/run_stage3_v7_d3_2_scan89_100plus_synthetic_safe.sh`；
  - 推理设置保持 Stage3 one-step + Stage2 streaming/KV-cache path，`tiled=0`；
  - `step-900`、`step-1000`、`step-1100` 均完成 10 个 synthetic 视频输出。
- 输出：
  - S3 输出：
    `s3://lxh/data/test/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`；
  - 本机下载：
    `/Users/lixiaohui/Desktop/stage3_v7d44_40gpu_step100_200_300_10synthetic_20260521/d44_40gpu`；
  - clean 对比目录新增：
    `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow1p0_40gpu/step900_d44`；
    `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow1p0_40gpu/step1000_d44`；
    `/Users/lixiaohui/Desktop/stage3_d44_flow0p1_d32_10synthetic_clean_20260522/d44_flow1p0_40gpu/step1100_d44`；
  - 三个目录均确认各 10 个 mp4。
- 空卡保护：
  - 测试结束后 `bfs6vaz4d6` 自动启动 `occupy_after_d44_eval` 全卡正常占卡；
  - watch 中 `bfs6vaz4d6` 8 卡确认均为 100% 利用率。

## 2026-05-25 48 卡机器重要 Stage3 实验云端备份

- 背景：
  - 用户准备释放 48 卡机器，要求先备份两个重要 Stage3 实验目录；
  - 明确约束：不得执行 `bolt task cancel` 或任何等价取消远程任务命令。
- 备份源与目标：
  - D4.4 40GPU：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
    -> `s3://lxh/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
  - D3.2 48GPU：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
    -> `s3://lxh/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
- 执行情况：
  - D4.4 全量备份 tmux：`backup_stage3_important`，日志显示 `BACKUP_DONE`；
  - D3.2 全量备份 tmux：`backup_stage3_d32_parallel`，日志显示 `BACKUP_DONE`；
  - 另外补跑了关键文件优先备份，确保 `run.log` 和 `output/step-*.safetensors` 先落到 `s3://lxh/artifacts/exp/`。
- 核验：
  - D4.4 云端存在 `run.log`、`output/step-1100.safetensors`，并能列出 `step-100` 到 `step-1100` 等 safetensors；
  - D3.2 云端存在 `run.log`、`output/step-2000.safetensors`，并能列出 `step-1` 到 `step-2000` 等 safetensors。
- 后续规则更新：
  - 已写入本机 Codex skill `/Users/lixiaohui/.codex/skills/用远程集群工作流/SKILL.md`：
    - 以后备份实验目录默认排除 `wandb/`、`wandb*`、`output/training_state/`、`output/*deepspeed*/`、optimizer/rng/scheduler 等用于 resume 的状态；
    - 除非用户明确要求完整 resume 状态，否则只备份 ckpt、log、脚本/config、必要 meta 和评估结果。
  - 注意：这条规则是在本次备份过程中补充的，因此本次云端目标里已经有部分 W&B / resume 状态对象；按新规则，不擅自删除云端对象，需用户明确同意后再清理。

## 2026-05-25 新 48 卡任务 3ec 恢复 D4.4 / D3.2 轻量 artifacts

- 背景：
  - 用户指定新 48 卡任务 `3ec6pb9art` 后续作为 48 卡训练 rank0 使用；
  - 需要从旧 t5 任务云端 artifacts 恢复两个重要 Stage3 实验到新机器本地 artifacts，方便后续继续测试/引用。
- 机器确认：
  - 按初始化新规则先执行 `bolt task show 3ec6pb9art` / `bolt task ls`；
  - `3ec6pb9art` 状态为 `RUNNING`，任务名 `6Node_teafortwo`，父任务 `s2rieabwpq`；
  - 已确认 `bolt task patchsshconfig 3ec6pb9art` 配置存在且匹配。
- 恢复源：
  - `s3://bolt-prod-2320845741/tasks/t5qdtykjsw/artifacts/exp`
- 恢复目标：
  - `/mnt/task_wrapper/user_output/artifacts/exp` on `3ec6pb9art`
- 恢复策略：
  - 只恢复轻量有用文件：`run.log`、`*.yaml/*.yml/*.sh/*.py/*.json/*.txt/*.md/*.csv`、validation/test 图片视频、`output/step-*.safetensors`；
  - 明确排除：`wandb/`、`wandb*`、`output/training_state/`、`output/*deepspeed*/`、optimizer/rng/scheduler/zero_to_fp32 等 resume 状态。
- 恢复结果：
  - D4.4：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5`
    - 文件数：96；
    - ckpt：`step-1/2/5/10/20/50/100/200/300/400/500/600/700/800/900/1000/1100.safetensors`。
  - D3.2：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_48gpu_v7_d3_2_lora_89f_videoonly_authorweights_stage1teacher_aligned_datafix_offlinewandb_20260516_v7d32_datafix_48gpu_fresh2`
    - 文件数：71；
    - ckpt：`step-1/2/5/10/20/50/100/200/500/1000/1500/2000.safetensors`。
- 约定：
  - 后续如果使用这批新的 48 卡机器启动训练，优先把 `3ec6pb9art` 作为 rank0 / 主节点处理。

## 2026-05-25 D4.4 NaN 追踪结论

- 分析对象：
  `/mnt/task_wrapper/user_output/artifacts/exp/train_stage3_release_40gpu_v7_d4_4_lora_89f_videoonly_usmgtpretrain_dualaccelerator_dfake5_offlinewandb_20260520_v7d44_40gpu_fresh_dfake5_lqprojfreq5/run.log`
  on `3ec6pb9art` restored artifacts。
- 第一次 NaN：
  - 首次 `fake_loss=nan` / `loss=nan` 出现在 `runner_step=3946, step=790`；
  - 这一轮是 `generator_update=0` 的 fake-only turn；
  - 前一轮 `runner_step=3945, step=790` 仍然有限：`loss=0.919913, student=0.683698, fake_loss=0.033308, dmd_student=0.202907, dmd_grad=0.492469`；
  - 同一时刻 student branch 仍有限，`stage3_v7_b_loss loss=0.478474 flow=0.030410 mse=0.002588 lpips=0.222738`。
- 早期风险信号：
  - `runner_step=295, step=60` 出现一次明显 DMD spike：`dmd_student=19.620518, dmd_grad=4.870988, total loss=20.099821`；
  - 配置里有 `stage3_dmd_grad_norm_max=5.0` 和 `stage3_dmd_spike_policy=skip`，但当时 `dmd_grad=4.870988 < 5.0`，所以没有触发 `dmd_skip=1`；
  - 整个日志里没有发现 `dmd_skip=1`。
- 判断：
  - D4.4 的 NaN 直接触发点更像是 fake critic 分支数值失稳，不是 reconstruction / LPIPS / flow branch 首先 NaN；
  - 但早期 DMD spike 说明 DMD guidance 方向曾经给过一次很大的 student update，可能导致后续 fake critic 追逐一个不稳定的 student distribution；
  - `gradient_accumulation_steps=1` 已经满足 DMD2 对交替训练的基本安全要求，因此本次 NaN 不能简单归因于 gradient accumulation；
  - 当前 guard 只按 mean absolute DMD grad 做阈值，没有按 DMD loss magnitude / 非有限 fake loss 做硬跳过，保护不够。
- 后续规避建议：
  - fake branch 增加 non-finite guard：`fake_pred` / `fake_loss` 非有限时跳过 fake backward 和 fake optimizer step；
  - DMD guard 同时检查 loss magnitude，例如启用 `stage3_dmd_loss_max` 或降低 `stage3_dmd_grad_norm_max`；
  - DMD / fake 可做 warmup 或 ramp，避免一开始就用完整权重；
  - fake lr 可低于 student lr，或初期冻结 / 降频更新 fake `lq_proj_in`；
  - 一旦 fake loss NaN，应停止并回滚到上一个健康 ckpt，不应继续训练。

## 2026-05-25 Stage3 OF-medium-long 过拟合重做

- 背景：
  - 用户复查 2026-05-23 的 OF-fast，指出只跑 `20` student steps，且退化过重，无法判断固定样本是否真的能被 Stage3 代码记住；
  - 原 OF-fast 只能证明代码跑通，不能解释 D4.4 的“早期清晰、后期变糊/震荡/NaN”。
- 本轮改动：
  - 新增 4 个独立 config，不修改通用 D4.4 训练源码：
    - `wanvideo/model_training/flashvsr/configs/history/stage3_of_a_medium_long_4gpu_overfit4_full_v7_d4_4.yaml`
    - `wanvideo/model_training/flashvsr/configs/history/stage3_of_b_medium_long_4gpu_overfit4_recononly_v7_d4_4.yaml`
    - `wanvideo/model_training/flashvsr/configs/history/stage3_of_c_medium_long_4gpu_overfit4_flowonly_v7_d4_4.yaml`
    - `wanvideo/model_training/flashvsr/configs/history/stage3_of_d_medium_long_4gpu_overfit4_dmdfakeonly_v7_d4_4.yaml`
  - 退化从完整双阶段 `params_aliyun_video_compression_v1.yaml` 改为中度单阶段：
    `wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_medium_x4test.yaml`
  - `max_train_steps=220`；
  - 保存点：`1,2,5,10,20,50,100,150,200,220`；
  - 仍使用固定 4 个 Takano 视频、首 batch cache、validation from train batch。
- 4 卡而不是 16 卡的原因：
  - overfit 的核心是快速判断固定样本能否被记住；
  - 16 卡 bs=1 至少需要 16 个视频，样本更多、通信更重，反而拖慢定位；
  - 因此本轮优先 4 卡 4 视频，每组单独看清楚后再决定是否 16 卡复验。
- 启动机器：
  - `3ec6pb9art`
    - `OF-A-medium-long` on GPU `0,1,2,3`
    - `OF-B-medium-long` on GPU `4,5,6,7`
  - `yagex8unf4`
    - `OF-C-medium-long` on GPU `0,1,2,3`
    - `OF-D-medium-long` on GPU `4,5,6,7`
- 远端 tmux：
  - 两台机器均使用 `tmux of_medium_20260525`；
  - 训练结束后对应脚本会在各自 `CUDA_VISIBLE_DEVICES` 范围内启动正常占卡。
- 初始确认：
  - 四组均已进入训练并输出 `stage3c_train`；
  - 首 batch 后 `data` 降到约 `0.1s`，说明不再卡重复 dataset scan；
  - C/flow-only 核心 runner step 约 `7.4s`；
  - A/D 带 G_real/G_fake/DMD，首个 generator step 明显更慢，符合预期。

### 2026-05-25 固定 LQ/GT 与远端 conductor 传输修正

- 用户指出 OF 对比不应在线随机退化；本轮改为先生成固定中度退化 LQ/GT，再让 OF-A/B/C/D 读取同一批 `.pt`，彻底消除不同实验退化不一致的问题。
- 固定数据位置：
  `/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`
  - 包含 `sample_00.pt` 到 `sample_03.pt`；
  - 同时包含 `gt/` 和 `lq/` mp4 预览；
  - 总大小约 `4.0G`。
- 代码实现只放在 overfit 专用文件：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_valbatch_lora.py`
  - 通过环境变量 `FLASHVSR_STAGE3_FIXED_LQGT_ROOT` 启用；
  - 通用 D4.4 训练文件 `train_flashvsr_stage3_v7_d4_4_lora.py` 未修改。
- 传输方式修正：
  - 最初尝试过本机 `ssh | tar | ssh` 桥接，4G 数据明显过慢；
  - 已按用户要求改为远端 tmux 正常 zsh 内直接使用 conductor：
    - `3ec6pb9art` 上传到 `s3://lxh/tmp/stage3_overfit4_medium_fixed_lqgt_20260525.tar`；
    - `yagex8unf4` 从该 S3 路径下载并解压；
  - 该规则已写入 `~/.codex/skills/用远程集群工作流/SKILL.md`：多 GB 远端到远端传输必须优先走远端 tmux zsh + conductor，不走本机 `/tmp` 桥接。
- 当前实际 tmux：
  - `3ec6pb9art`: `of_fixed_lqgt_20260525`
    - `OF_A`: full loss on GPU `0,1,2,3`
    - `OF_B`: recon-only on GPU `4,5,6,7`
  - `yagex8unf4`: `of_fixed_lqgt_20260525`
    - `OF_C`: flow-only on GPU `0,1,2,3`
    - `OF_D`: DMD/fake-only on GPU `4,5,6,7`
- 状态：
  - A/B 已输出多条 `stage3c_train` loss；
  - C/D 已确认打印 `Using pre-generated fixed LQ/GT tensors` 和 `Loaded fixed LQ/GT batch for reuse`，并有训练进程占用 GPU；
  - yagex 旧占卡残留已清理，避免和 C/D 抢显存。

## 2026-05-26 Stage3 DMD debug plan 与 Stage2 v7E pixel-loss 对照

- 新增 DMD/fake 分支 debug 计划：
  `doc/flashvsr_stage3_dmd_fake_branch_debug_plan_20260526.md`
  - 目的：把 `OF-D = DMD/fake-only` 的绿色/灰屏问题拆成固定输入、G_real/G_fake forward、fake critic 单训、DMD gradient 方向、DMD-only overfit、full loss 组合六层验证；
  - 当前判断：DMD/fake-only 不能单独稳定训练 student，但不能直接等价为“DMD 理论错误”；优先怀疑 fake branch、teacher/fake wrapper、timestep/noisy point、grad guard 和更新频率。
- 新增 7E 对照实验，不修改任何已有 Stage2/Stage3 正式训练代码：
  - 训练代码：
    `wanvideo/model_training/flashvsr/train_flashvsr_stage2_v7_e_pixelloss_lora.py`
  - config：
    `wanvideo/model_training/flashvsr/configs/history/stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val.yaml`
  - 启动脚本：
    `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage2-Release-48GPU-v7-E-Lora-89f-VideoOnly-bs1-lr5e6-FlowPixelLPIPS-BlockSparse-Worker2-Val.sh`
- 7E 设计：
  - 从 Stage2 `v6.4.1 step-6000` 初始化；
  - 数据、尺寸、退化、worker、Stage2 block-sparse attention、50-step streaming validation 均沿用 641；
  - 不引入 DMD/G_real/G_fake；
  - 在原 Stage2 flow loss 外增加 selected-window `z_pred -> Wan decoder -> MSE/LPIPS`；
  - 默认 `flow=1.0, mse=1.0, lpips=2.0`，首帧 pixel/LPIPS 权重 `4.0`；
  - 学习率从 641 的 `1e-5` 降到 `5e-6`；
  - 训练退出后脚本会在本节点启动正常占卡。
- 本地检查：
  - `python3 -m py_compile train_flashvsr_stage2_v7_e_pixelloss_lora.py` 通过；
  - `bash -n FlashVSR-Stage2-Release-48GPU-v7-E-...sh` 通过；
  - YAML load 检查通过，确认 `learning_rate=5e-6`、`validation_num_inference_steps=50`。
- 远端启动：
  - 原计划使用 `6Node_vedvp`，但 `ddbj6ifhyy` 的旧占卡进程被 kill 后留下 defunct python，8 张卡显存无法释放，`nvidia-smi --gpu-reset` 返回 `Not Supported`；
  - 为保持 48 卡而不是退成 40 卡，本次使用 5 个 vedvp 正常节点 + `teafortwo` 的 `etpf5tf68s` 补位：
    `nuar88n3jj imatc2q44e s8h54wda44 v3rxjhaqc3 zh6rf39ybz etpf5tf68s`；
  - rank0: `nuar88n3jj`，`MASTER_ADDR=240.12.226.237`，`MASTER_PORT=29562`；
  - 实验目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val_20260526_031800`
  - W&B run:
    `https://wandb.ai/veralee/flashvsr/runs/npsgo0be`
  - 已确认：
    - 读取 `Stage2 v6.4.1 step-6000` 初始化成功；
    - W&B online 正常；
    - 3 个固定 Stage2 validation samples 已准备；
    - 已输出第一条 `stage2_v7e_loss` 和训练 `step=1`；
    - 6 节点 48 卡进入训练，绝大多数 GPU util 为 `100%`。

### 2026-05-26 7E 改回 DVP 原始 6 节点并补初始化 ddbj6ifhyy

- 用户要求 `ddbj6ifhyy` 已空，改回 DVP 原始 6 节点重启 7E，并保持训练步数 `10000`。
- 先核对 641 原配置：
  - `max_train_steps=10000`；
  - 原 641 `save_steps=100`，但 7E validation 更重，因此 7E 改为 `save_steps=1000`，并保留早期 `extra_save_steps="10,20,50,100,200,500"`。
- 第一次用 DVP 原始 6 节点启动时发现：
  - rank0 已过固定 validation samples 和 W&B，且输出第一条 `stage2_v7e_loss`；
  - 但 `ddbj6ifhyy` GPU 长时间低显存/低利用率，并出现 OpenCV stream timeout；
  - 进一步检查发现 `ddbj6ifhyy` 的 `conductor` 报 `/usr/local/bin/conductor: line 9: exec: aws: not found`，说明节点初始化不完整。
- 处理：
  - 停止不健康的 7E run，退出 trap 自动给节点挂回占卡；
  - 本地修正 `bolt_lxh/setup_after_docker1.sh`：将 `chsh` 改为 `timeout 5 chsh ... || true`，避免 Bolt 节点卡在交互式 shell 切换；
  - 在 `ddbj6ifhyy` 远端 `tmux init_ddbj_7e` 可见执行初始化；
  - 验证 `zsh -lc 'which aws; conductor s3 ls s3://lxh/ | head; /mnt/conda_envs/flashvsr/bin/python -V'` 通过；
  - 该初始化排查规则已写入 `~/.codex/skills/用远程集群工作流/SKILL.md`。
- 当前重启的 7E：
  - 节点：
    `nuar88n3jj imatc2q44e ddbj6ifhyy s8h54wda44 v3rxjhaqc3 zh6rf39ybz`
  - rank0: `nuar88n3jj`
  - `MASTER_ADDR=240.12.226.237`
  - `MASTER_PORT=29564`
  - 实验目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val_20260526_040200`
  - W&B run:
    `https://wandb.ai/veralee/flashvsr/runs/wd1c8kkp`
  - 已确认：
    - 固定 Stage2 validation samples 准备完成；
    - W&B online 正常；
    - rank0 已输出第一条 `stage2_v7e_loss`：
      `loss=0.905710 flow=0.002954 mse=0.010605 lpips=0.446076 latent=[20,22) frame=[77,85)`；
    - 第一条真实 train step 输出：
      `epoch=0 step=1 loss=0.813514`；
    - 随后输出第二条 `stage2_v7e_loss`：
      `loss=0.352846 flow=0.121524 mse=0.002958 lpips=0.114182 latent=[15,17) frame=[57,65)`；
    - 48 卡进入训练态，绝大多数 GPU 显存约 `128-139GB`，利用率 `100%`；
    - `ddbj6ifhyy` 初始化明显慢于其他节点，但已完成首步同步，不再是缺 aws/conductor 的状态。
    - 后续已继续到 `step=2`：
      `epoch=0 step=2 loss=0.719745`；
      `stage2_v7e_loss loss=0.290731 flow=0.158982 mse=0.003381 lpips=0.064183 latent=[0,2) frame=[0,5)`；
      第一阶段 LPIPS 初始化导致首步约 `508s`，第二步累计均值降到约 `376s/it`，需要继续观察后续稳定速度。
- 守护：
  - 7E 启动脚本本身带 `START_OCCUPY_ON_EXIT=1`，训练正常结束或普通异常退出会启动占卡；
  - 额外在 6 个 DVP 节点各启动远端 `tmux gpu_empty_guard_v7e`，检测本节点 8 张 GPU 持续空闲后兜底启动 `/mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`；
  - OF-A/B/C/D 已存在按 4 卡分组的 `gpu_empty_guard_OF_*_fixedlqgt`，不会依赖本机进程，实验死掉后会按对应 GPU 组启动占卡。

### 2026-05-26 DMD/fake 问题范围收缩与 OF-E

- 用户检查 fixed-LQGT OF validation 后确认：`OF-D = DMD/fake-only` 到最后仍有严重色偏和模糊；这说明问题范围已经明显集中到 DMD/fake 分支。
- 新增详细代码审阅文档：
  `doc/flashvsr_stage3_d44_dmd_code_audit_20260526.md`
  - 逐段记录 `v7-D4.4` 从 641 到现在所有 DMD/fake 新增逻辑；
  - 覆盖 `z_pred` 来源、`G_real/G_fake` 构造、DMD probe、fake FM loss、DMD student pseudo-target、dual optimizer dfake 调度和当前可疑点。
- 更新 DMD debug 计划：
  `doc/flashvsr_stage3_dmd_fake_branch_debug_plan_20260526.md`
  - 后续重点从“Stage3 整体有问题”收缩为验证 DMD direction、normalization、fake critic、projector temporal mode。
- 新增 OF-E 配置，不修改正式 D4.4 代码：
  `wanvideo/model_training/flashvsr/configs/history/stage3_of_e_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4.yaml`
  - Loss: `flow=1, mse=1, lpips=2, dmd=0, fake_fm=0`；
  - 目的：作为 “完整主链路但去掉 DMD/fake” 对照，和 OF-A/B/C/D 使用同一批 fixed LQ/GT。
- OF-E 远端启动：
  - 机器：`3ec6pb9art`；
  - GPU：`0,1,2,3`；
  - 目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4_fixedpt_20260525_231347`；
  - 明确设置：
    `FLASHVSR_STAGE3_FIXED_LQGT_ROOT=/mnt/task_wrapper/user_output/artifacts/data/overfit/stage3_overfit4_medium_fixed_lqgt_20260525`；
  - 已确认日志打印 `Using pre-generated fixed LQ/GT tensors` 和 `Loaded fixed LQ/GT batch for reuse`；
  - 第一条 loss：
    `loss=1.111800 flow=0.002922 mse=0.014376 lpips=0.547251 latent=[16,18) frame=[61,69)`；
  - 注意：此前有一次未带 `FLASHVSR_STAGE3_FIXED_LQGT_ROOT` 的 OF-E 启动，目录不带 `fixedpt`，已停止，不作为有效结果。

### 2026-05-26 teafortwo DMD 并行验证与 DVP 7E 重启

- teafortwo 当前并行实验：
  - `OF-E`：`3ec6pb9art` GPU `0,1,2,3`，目录
    `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_fixedlqgt_4gpu_overfit4_flow_recon_v7_d4_4_fixedpt_20260525_231347`；
    用于验证 `flow + MSE + LPIPS`、完全去掉 DMD/fake 后能否过拟合。
  - `DMD-3 fakecritic-only`：`yagex8unf4` GPU `0,1,2,3`，目录
    `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD3_fixedlqgt_4gpu_fakecritic_only_v7_d4_4_20260525_233413`；
    只训练 fake critic，不更新 student，用来观察 fake 分支是否本身稳定。
  - `DMD-4 fakeproj-frozen`：`yagex8unf4` GPU `4,5,6,7`，目录
    `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_frozen_v7_d4_4_20260525_233413`；
    DMD-only 且冻结 fake `lq_proj_in`，用于和旧 OF-D / every1 版本对比 projector 是否导致色偏。
  - `DMD-4 fakeproj-every1`：`etpf5tf68s` GPU `0,1,2,3`，目录
    `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every1_v7_d4_4_20260525_234500`；
    DMD-only 且 fake `lq_proj_in` 每个 runner step 更新；第一次启动时 empty-gpu guard 抢占了目标 GPU，已重启并关闭 `START_EMPTY_GPU_GUARD`，仅保留退出后占卡。
- teafortwo 当前确认：
  - OF-E 已跑到 `runner=94 / step=19`；
  - DMD-3 已出 fake loss，并在 generator turn 打印 `dmd_probe`；
  - DMD-4 frozen 已出 `dmd_student/dmd_grad`；
  - DMD-4 every1 已出 `runner=4 / step=1`，`fake_lq_proj_update=1`，GPU `4,5,6,7` 单独占卡。
- DVP 7E 重启：
  - 机器：`ghnn8rzfhn 5qfr6zsy8t ayzxx94tqr 4sdz8t77ha syyyrtw28j pxrpk9wpy8`；
  - rank0：`ghnn8rzfhn`，`MASTER_ADDR=240.12.26.14`；
  - 第一次启动卡在 W&B online `ConnectTimeout`，没有进入训练；
  - 已将 7E 专用 YAML 改为 `wandb_mode: offline`，并在 7E 启动脚本中设置 `WANDB_DIR=${RUN_DIR}`，避免 W&B 写到 artifacts 外；
  - offline 重启目录：
    `/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_48gpu_v7_e_lora_89f_videoonly_bs1_lr5e6_flow_pixel_lpips_blocksparse_worker2_val_20260526_145000`；
  - 当前已通过 W&B offline 和 DeepSpeed 初始化，6 节点都到 `0it`，仍在等待第一批数据/第一条 loss，需要继续监控。
  - 后续已确认进入真实训练 forward：
    `stage2_v7e_loss loss=0.905710 flow=0.002954 mse=0.010605 lpips=0.446076 latent=[20,22) frame=[77,85)`；
    rank0 GPU 显存约 `128GB`，利用率 `100%`。
  - 已确认第一条真实 train step：
    `epoch=0 step=1 loss=0.830325`；
    随后第二条 `stage2_v7e_loss loss=0.352914 flow=0.121557 mse=0.002957 lpips=0.114200 latent=[15,17) frame=[57,65)`；
    首步约 `228.9s/it`，训练正常滚动。
## 2026-05-26

- Stage3 DMD debug：新增 `wanvideo/model_training/flashvsr/tools/check_fixed_lqgt_root.py`，在 3ec 上完成 DMD-0 固定 LQ/GT checksum，输出 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD0_fixedlqgt_checksum_20260526/fixed_lqgt_checksum.json`。4 个固定样本均为 `[89,3,768,1280]`，后续 DMD probe 可排除在线退化差异。
- Stage3 DMD debug：新增独立入口 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_dmd_debug_lora.py` 和 config `wanvideo/model_training/flashvsr/configs/history/stage3_dmd1_fixedlqgt_4gpu_tensor_dump_v7_d4_4.yaml`。该入口 monkey-patch overfit 版本 DMD loss 做 tensor dump，不修改正式 D44 训练代码。
- Stage3 DMD debug：在 3ec GPU 4-7 跑 DMD-1 initial dump，输出 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_fixedlqgt_tensor_dump_20260526/dmd_dump/step_000000`。结果为 `G_fake == G_real` 初始 sanity，DMD grad 为 0。
- Stage3 DMD debug：在 3ec GPU 4-7 跑 DMD-1 after-fake dump，输出 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_fixedlqgt_tensor_dump_20260526/dmd_dump/step_000001`。runner 5 产生非零 DMD：`dmd_student=0.774298, dmd_grad=0.956290, dmd_skip=0, dmd_loss_clamp=0`。
- Stage3 DMD debug：新增 `wanvideo/model_training/flashvsr/tools/analyze_dmd_tensor_dump.py`，基于 after-fake dump 完成 DMD-2 sign/norm probe，输出 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD2_afterfake_sign_norm_probe_20260526/sign_norm_report.json`。当前符号会拉近 real、远离 fake；翻转符号相反。首轮判断 DMD 不是简单 sign 写反，后续重点查 fake critic 质量和尺度稳定性。
- Stage3 DMD debug：新增 `wanvideo/model_training/flashvsr/tools/decode_dmd_tensor_dump.py`，把 DMD-1 after-fake dump 中的 `student_z_pred / g_real_x0 / g_fake_x0 / shared_noisy_latents` 通过 Wan VAE 解码成视频。远端输出 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD1_afterfake_decode_videos_20260526/step_000001`，已同步到桌面 `/Users/lixiaohui/Desktop/stage3_DMD1_afterfake_decode_videos_20260526/step_000001`。该可视化用于判断 real/fake 分支本身是否已偏色或崩坏。
- Stage3 DMD code audit：新增 DMD2 对照文档 `doc/flashvsr_stage3_dmd2_code_audit_against_d44_20260526.md`。已读取 `/Users/lixiaohui/Library/CloudStorage/Box-Box/mac_code/DMD2/main/train_sd.py`、`main/sd_unified_model.py`、`main/sd_guidance.py`、`main/edm/edm_guidance.py`，确认 DMD2 的 DMD 符号、`abs(p_real).mean` normalization、fake detach、fake 每 runner step 更新、generator 按 `dfake_gen_update_ratio` 更新、`gradient_accumulation_steps == 1` 等核心逻辑；同时记录 FlashVSR 独有风险：视频 LQ projector / 89f streaming / fake `lq_proj_in` 更新策略需要实验验证。
- Stage3 DMD ownership：新增独立 debug copy `train_flashvsr_stage3_v7_d4_2_ownership_lora.py` 与 `train_flashvsr_stage3_v7_d4_4_ownership_fixed_lora.py`，不修改正式 D42/D44。已在 `if8r8fmkiv` 用同一 fixed LQ/GT batch 跑 D4.2 单 runner vs D4.4 dual accelerator 对照。结论：fake-only turn 未污染 student，G_real 全程未更新；但 D4.4 fake 参数更新幅度和 runner5 DMD loss/grad 明显大于 D4.2。详细记录见 `doc/flashvsr_stage3_dmd_fake_branch_debug_plan_20260526.md` 的 `DMD-6` 节。后续要固定 DMD timestep/noise 并检查 D4.4 fake optimizer/DeepSpeed 更新尺度。
- Stage3 DMD fixed-point ownership：在 ownership debug copy 中新增 `FLASHVSR_STAGE3_FIXED_DMD_TIMESTEP_ID` 和 `FLASHVSR_STAGE3_FIXED_DMD_NOISE_SEED`，同时固定 fake FM 和 DMD probe 的加噪点。已在 `if8r8fmkiv` 跑 D4.4 fixed-point 与 D4.2 fixed-point。固定后仍未发现 fake-only 污染 student 或 G_real 被更新，但 D4.4 fake_delta 和 runner5 DMD 明显大于 D4.2，说明差异不是随机 timestep/noise 造成，更指向 D4.4 dual accelerator / DeepSpeed fake optimizer 更新尺度问题。
- Stage3 DMD4：补齐 fake `lq_proj_in` every5/current 第三组。新增 config `wanvideo/model_training/flashvsr/configs/history/stage3_dmd4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4.yaml`，远端目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_DMD4_fixedlqgt_4gpu_dmdonly_fakeproj_every5_current_v7_d4_4_20260526_ownership_cmp`。早期日志显示 runner5 `dmd_student=0.932228`，runner10 `dmd_student=3.000000` 且 `dmd_loss_clamp=1`，需要继续看保存点/validation 判断是否对应色偏或灰屏。
- Stage3 DMD 实验台账：已在 `doc/flashvsr_stage3_dmd_fake_branch_debug_plan_20260526.md` 追加 `2026-05-26 实验台账：OF-E / DMD0-4`，逐项记录机器、GPU、卡数、状态、进度和实验目录。当前 OF-E、DMD-3、DMD-4 frozen/every1/every5-current 仍在运行；DMD0/1/2 已完成。

## 2026-05-26 OF-1000 overfit relaunch

- Downloaded current DMD3/DMD4 validation results to `/Users/lixiaohui/Desktop/stage3/DMD_validation_20260526`.
- Added isolated OF-only controls without modifying production D44 training code:
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_overfit_lora.py`: optional `FLASHVSR_OVERFIT_GT_SHARPEN=1` patch for OF-F.
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_dmd_debug_lora.py`: gated DMD tensor dumps at selected steps.
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-OF-Fast-4GPU-v7-D4-4.sh`: `EXTRA_ARGS` and DMD dump dir support.
- Synced fixed overfit LQ/GT root to `s3://lxh/tmp/stage3_overfit4_medium_fixed_lqgt_20260525` and pulled it to pfg/qcp.
- Launched 1000-step OF overfit set:
  - OF-A r2: qcp GPU 0-3, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_A_1000_fixedlqgt_4gpu_full_v7_d4_4_r2_20260526_032423`.
  - OF-B r2: qcp GPU 4-7, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_B_1000_fixedlqgt_4gpu_recononly_v7_d4_4_r2_20260526_032427`.
  - OF-C r2: pfg GPU 1,2,4,5, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_C_1000_fixedlqgt_4gpu_flowonly_v7_d4_4_r2_20260526_032432`.
  - OF-D r3: 3ec GPU 4-7, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_D_1000_fixedlqgt_4gpu_dmdfakeonly_v7_d4_4_r3_20260526_032641`.
  - OF-E: etpf GPU 4-7, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_E_1000_fixedlqgt_4gpu_flow_recon_v7_d4_4_20260526_031700`.
  - OF-F: if8 GPU 4-7, `/mnt/task_wrapper/user_output/artifacts/exp/stage3_OF_F_1000_sharpgt_4gpu_full_v7_d4_4_20260526_031705`.
- Early verification: OF-A/B/C/D/E/F all reached loss; B/C/E/F have early validation files; A/D/F have DMD tensor dump enabled for later real/fake decode.
- Stopped the older duplicate `OF-E fixedpt` run on `3ec6pb9art` GPU 0-3 at user request, leaving those GPUs free for collaborator debug. Downloaded latest validation snapshots to `/Users/lixiaohui/Desktop/stage3/DMD_OF_latest_validation_20260526`: `OFE_fixedpt_old/step-200`, `DMD3_fakecritic_only/step-50`, `DMD4_fakeproj_frozen/step-50`, `DMD4_fakeproj_every1/step-50`, and `DMD4_fakeproj_every5_current/step-20`. Also generated `_first_frame_montage.png` for quick HR/LQ/SR visual comparison; first-frame inspection shows `DMD4_every1` and `DMD4_every5_current` have severe artifacts/color shift, while `DMD3_fakecritic_only` and `OFE_fixedpt_old` look much closer to HR/LQ on the sampled frame.

## 2026-05-26 TeaForTwo DMD 排查计划执行快照

- 已更新桌面计划 `/Users/lixiaohui/Desktop/PLAN.md`，明确：
  - `dfake=5` 固定，不再作为主变量；
  - `visual-and-metrics-judge` 是正式 agent，指标只做辅助，最终要由我直接看 montage 判断黑/灰/黄/绿/糊/结构坏；
  - 新增 `Gate Y` 专门定位 DMD-only 黄绿/灰屏来源：fake branch、fake LQ projector、real/fake x0 drift、DMD grad 方向；
  - `3ec6pb9art:0-3` 属于用户/同事，不再占卡、不再清理、不再调度。
- 本地结果入口已整理：
  - `/Users/lixiaohui/Desktop/stage3/OF/long_clean`：保留第二轮 OF-A/B/C/D/E 的 `{gt,lq,sr}` clean 结构；
  - `/Users/lixiaohui/Desktop/stage3/DMD_clean`：保留 `DMD3_fakecritic_only`、`DMD4_fakeproj_frozen`、`DMD4_fakeproj_every1`、`DMD4_fakeproj_every5_current` 的 `{gt,lq,sr,meta}` clean 结构；
  - 新增辅助工具 `tools/flashvsr_stage3_visual_judge.py`，生成 `_visual_judge/first_frame_compare.png`、`metrics.csv`、`visual_judge.md`。
- 我直接审阅了 `/Users/lixiaohui/Desktop/stage3/DMD_clean/_visual_judge/first_frame_compare.png`：
  - `DMD4_fakeproj_every1` 和 `DMD4_fakeproj_every5_current` 很快出现明显黄绿/灰屏/结构破坏；
  - `DMD4_fakeproj_frozen` 明显缓解，但仍有色偏和不稳定；
  - `DMD3_fakecritic_only` 更接近 GT/LQ，但只能说明 fake critic alone 没有立刻崩，不证明 DMD student 方向有效；
  - 当前结论：220 step 已足够暴露 DMD-only 问题，后续不靠盲目长训，转向 fake score path / projector / x0 语义 / grad 方向拆解。
- TeaForTwo 当前卡位：
  - `3ec6pb9art:4-7` 已占卡；`3ec6pb9art:0-3` 空着但不归 Codex 管；
  - `yagex8unf4:0-7`、`etpf5tf68s:0-7`、`if8r8fmkiv:0-7`、`qcpdgx65xx:0-7` 已占卡，等待按 Gate 接管；
  - `pfg986en8d:0` 是僵尸显存 `166G/0%`，`pfg986en8d:3` 是 NVML 异常，不分主实验。
- 代码审阅 subagent 结论：
  - `teacher-equivalence`：G_real 应理解为 Stage1 weights 进入 Stage3/Stage2 22-latent wrapper 的 one-step x0 score，不是原生 Stage1 50-step 结果直接相等；
  - `condition-alignment`：需要落盘比较 student/G_real/G_fake 的 LQ projector 输出、mask、timestep/noise/noisy_latents、model IO；
  - `dmd-formula`：D44 当前 DMD 符号和 DMD2 per-sample `abs(p_real).mean` normalization 方向上不是显然写反，下一步重点查 fake score 质量和更新后 x0 漂移。
- Gate2 condition alignment：新增本地独立工具 `wanvideo/model_training/flashvsr/tools/stage3_gate2_condition_alignment_dump.py`，只用于固定 batch + 固定 timestep/noise 下 dump student/G_real/G_fake 的 DMD point condition，不修改正式 `train_flashvsr_stage3_v7_d4_4_lora.py` 或任何正式训练入口。输出 JSON 包含 LQ projector 输出、context、DMD point、理论/实际 block mask、model input/output stats，以及 real/fake 对齐结论；后续同步到远端后可单卡运行验证。
- GateY hack-probe：新增本地复制入口 `wanvideo/model_training/flashvsr/train_flashvsr_stage3_v7_d4_4_hack_probe_lora.py`，不修改正式 D44。新增 3 个最小魔改 variant：
  - `fake_x0_equal_real`：DMD 内强制 `fake_x0=real_x0`，用于确认破坏是否来自 real/fake score 差异；
  - `dmd_grad_scale0p1_clipnear`：保留 score 但缩小并裁剪 DMD grad，用于判断是否是更新幅度/spike 问题；
  - `color_match_fake_x0_to_real`：对齐 fake/real x0 的 per-sample mean/std，用于判断黄绿偏色是否来自 x0 全局颜色/尺度漂移。
  配套文件：`stage3_gateY_hack_probe_*_4gpu_dmdonly_dfake5.yaml`、`FlashVSR-Stage3-GateY-HackProbe-4GPU-v7-D4-4.sh`、`doc/flashvsr_stage3_gateY_hack_probe_20260526.md`。已通过本地 `py_compile` 和 `bash -n`，尚未远程启动。
- GateY hack-probe 远端启动：
  - `qcpdgx65xx:0-3` 启动 `fake_x0_equal_real`，目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_gateY_hack_probe_fake_x0_equal_real_4gpu_dmdonly_dfake5_20260526_fake_x0_equal_real_qcp`；
  - `qcpdgx65xx:4-7` 启动 `dmd_grad_scale0p1_clipnear`，目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_gateY_hack_probe_dmd_grad_scale0p1_clipnear_4gpu_dmdonly_dfake5_20260526_dmd_grad_scale0p1_clipnear_qcp`；
  - `if8r8fmkiv:4-7` 启动 `color_match_fake_x0_to_real`，目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_gateY_hack_probe_color_match_fake_x0_to_real_4gpu_dmdonly_dfake5_20260526_color_match_fake_x0_to_real_if8`；
  - 三组均固定 `dfake=5`，均为 DMD-only：`flow/mse/lpips=0`、`stage3_dmd_weight=1`、`stage3_fake_fm_weight=1`，保存 `1,2,5,10,20,50,100,150,200,220`；
  - 三组均有 `START_OCCUPY_ON_EXIT=1` 和 `gpu_empty_guard`，退出后会自动启动占卡。
- GateY step=1 早期信号：
  - `fake_x0_equal_real` 在 fake 更新后 `real_fake_mse_before=0.0133564`，但强制 `fake_x0=real_x0` 后 `dmd_grad_absmean=0`，这是预期的“关闭 score 差异”对照；
  - `dmd_grad_scale0p1_clipnear` 在 step=1 得到 `dmd_grad_absmean=0.152428`、`dmd_grad_absmax=0.25`，说明裁剪/缩放已生效；
  - `color_match_fake_x0_to_real` 在 step=1 即使 mean/std 对齐后仍有 `dmd_grad_absmean=3.34646`、`dmd_grad_absmax=18.481`，初步说明问题不是简单全局颜色均值方差漂移，更像 DMD 局部方向/幅度或 fake score 细节不稳定；
  - 下一步需要下载 step `1/2/5/10/20` validation，用 visual judge + 图像理解直接看哪组缓解黄/灰。
- TeaForTwo 占卡状态更新：
  - 不碰 `3ec6pb9art:0-3`；本轮只在 `3ec6pb9art:5-7` 补了 `occupy_safe_gpu5/6/7`，`3ec:4` 保持既有占卡；
  - `yagex8unf4:1-7` 补了 `occupy_safe_gpu1..7`，`yagex:0` 原本已有进程所以跳过；
  - `qcpdgx65xx:0-7` 正在跑 GateY 两组；`if8r8fmkiv:4-7` 正在跑 GateY color-match，`if8:0-3` 原有占卡不动；
  - `pfg986en8d` 仍视为不稳定节点，`pfg:0` 僵尸、`pfg:3` NVML 异常，不分主实验。

## 2026-05-26 GateY 扩展与 fake branch 回传排查

- GateY 第一批结果已下载到 `/Users/lixiaohui/Desktop/stage3/GateY_validation_20260526`，clean 入口为 `/Users/lixiaohui/Desktop/stage3/GateY_clean_20260526`，视觉报告为 `/Users/lixiaohui/Desktop/stage3/GateY_clean_20260526/_visual_judge`。
- 视觉结论：`fake_x0_equal_real` 稳定但属于零梯度对照；`color_match_fake_x0_to_real` 只能部分缓解，step-5 出现网格/过锐/结构漂移；`dmd_grad_scale0p1_clipnear` step-5/10 仍会发灰、过锐、结构坏。当前判断 DMD-only 失败不是简单颜色 mean/std 漂移，而是 DMD score path 的局部尖峰、normalization 尺度、fake 条件支路或 fake 更新尺度。
- 新增 GateY 第二批复制入口变体，仍固定 `dfake=5`，不修改正式 D44：`fake_score_percentile_clip`、`weight_factor_rms_detach`、`freeze_fake_lq_proj`、`fake_lr0p1`、`fake_lr0p01`。
- 当前启动：
  - `yagex8unf4:0-3` 跑 `fake_score_percentile_clip`，tmux `gateY2_fake_score_percentile_clip`。
  - `yagex8unf4:4-7` 跑 `weight_factor_rms_detach`，tmux `gateY2_weight_factor_rms_detach`。
  - `etpf5tf68s:0-3` 跑 `freeze_fake_lq_proj`，tmux `gateY2_freeze_fake_lq_proj`。
  - `etpf5tf68s:4-7` 跑 `fake_score_clip_p95`，tmux `gateY2_fake_score_clip_p95`。
  - `if8r8fmkiv:0-3` 跑 `trust0p03_clip0p05`，tmux `gateY2_trust_region_0p03_0p05`。
  - `3ec6pb9art:4-7` 跑 `fake_lr0p1`，tmux `gateY2_fake_lr0p1`；`3ec6pb9art:0-3` 按用户要求不碰。
- 代码审阅重点更新：D4.2 fake update 是单 runner `fake_loss.backward()` + `_average_stage3c_fake_gradients(fake_model)` + AdamW；D4.4 是独立 `fake_accelerator.backward(fake_loss)` + DeepSpeed fake engine。`fake_loss` detach 本身正确；若 `gradient_accumulation_steps > 1`，D4.4 fake side 不等价 DMD2。当前 GateY 配置为 1，所以优先查 fake DeepSpeed 更新尺度与 fake `lq_proj_in`。
- 22:37 更新：已把 GateY 第一批 `fake_x0_equal_real` 和 `dmd_grad_scale0p1_clipnear` 的 `step-20` validation 从 `qcpdgx65xx` 拉回到 `/Users/lixiaohui/Desktop/stage3/GateY_validation_20260526`，并刷新 clean 目录 `/Users/lixiaohui/Desktop/stage3/GateY_clean_20260526` 与 `_visual_judge`。图像审阅：`fake_x0_equal_real` 到 step20 仍稳定，`dmd_grad_scale0p1_clipnear` 到 step20 仍灰暗/条纹/过锐，说明全局缩放裁剪不足，后续重点继续查 fake branch score 内容和 fake 更新路径。
- 22:37 更新：新增 D4.2 single-runner DMD-only 对照，不修改正式 D44。新增本地文件 `wanvideo/model_training/flashvsr/configs/history/stage3_d42_fixedlqgt_4gpu_dmdonly_singlerunner_dfake5.yaml` 和 `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage3-D42-FixedLQGT-4GPU-DMDOnly-SingleRunner-Dfake5.sh`；已在 `qcpdgx65xx:0-3` 启动 tmux `gateD42_singlerunner_dmdonly_qcp03`，远端目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D42_fixedlqgt_4gpu_dmdonly_singlerunner_dfake5_qcp03_20260526_223400`。目的：确认 D4.2 单 runner/separated optimizer 是否也出现 DMD-only 黄灰；若 D4.2 正常而 D4.4 异常，优先修 D4.4 dual accelerator / fake DeepSpeed update scale；若 D4.2 也异常，回查 fake loss/score 语义。
- 22:44 更新：已下载 GateY2 `step-1/2/5` 到 `/Users/lixiaohui/Desktop/stage3/GateY2_validation_20260526`，整理到 `/Users/lixiaohui/Desktop/stage3/GateY2_clean_20260526`，并生成 `_visual_judge`。人工审阅写入 `/Users/lixiaohui/Desktop/stage3/GateY2_clean_20260526/_visual_judge/human_review.md`。结论：`fake_lr0p1`、`fake_score_clip_p95/p995`、`freeze_fake_lq_proj`、`trust0p03_clip0p05`、`weight_factor_rms_detach` 都没有单独修复 DMD-only 色偏/结构风险；后续关键是 D4.2 single-runner 对照和 fake score/x0 语义排查。
- 22:53 更新：GateY/GateY2 单点 probe 已按结论收尾停止，并在对应机器拉起高显存占卡：`yagex8unf4:0-7`、`etpf5tf68s:0-7`、`if8r8fmkiv:0-7`、`qcpdgx65xx:4-7`、`3ec6pb9art:4-7`。`3ec6pb9art:0-3` 按用户要求继续不碰。当前唯一继续训练的主诊断是 `qcpdgx65xx:0-3` 的 D4.2 single-runner DMD-only 对照，tmux `gateD42_singlerunner_dmdonly_qcp03`，目录 `/mnt/task_wrapper/user_output/artifacts/exp/stage3_D42_fixedlqgt_4gpu_dmdonly_singlerunner_dfake5_qcp03_20260526_223400`；当前到 runner 3 / step 1，需等 runner 5 / step 2 才有非零 DMD 诊断信号。
