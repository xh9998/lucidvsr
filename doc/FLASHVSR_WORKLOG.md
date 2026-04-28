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
