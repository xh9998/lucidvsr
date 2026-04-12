# FlashVSR 工作记录

这份记录只保留关键节点。

## 恢复目录后的训练调试

- 恢复了 `lucidvsr` 工作目录里的训练主文件、数据层、测试脚本和文档。
- 修掉了 `yaml` 启动时 `prompt_tensor_path` 仍被命令行强制要求的问题。
- 修掉了训练日志默认不打 `loss` 的问题，现在 `run.log` 会持续打印 `step/loss`。
- 修掉了长跑被外部 `SIGHUP` 杀掉的问题，后续长跑改成远程 `tmux remote` 启动。
- 修掉了 `dataset` 输出 `PIL.Image` 导致多卡拼 batch 失败的问题，训练数据改成 tensor 化返回。
- 修掉了 `parquet` 模式下 `video_iter=None` 的问题。
- 修掉了 `storymotion manifest + takano shard` 混合输入时只走一条源的问题。
- 修掉了 distributed/rank 切分不完整的问题，避免多卡重复读相同样本。

## DeepSpeed / FlashAttention / Checkpoint 调试

- 一开始 `deepspeed` 配置虽然写了，但实际训练没有完全走到正确路径。
- 确认了 `DeepSpeed ZeRO-2` 真实生效，不只是配置文件里写了 `zero_stage=2`。
- 确认了 `flash-attn` 真实生效，运行时实际走的是 `flash_attn_2` 分支。
- 发现 `gradient checkpoint` 最开始并没有真正启用，根因是 `pipe.dit` 在 LoRA patch 后仍停在 `eval()`。
- 修掉这个问题后，`gradient_checkpoint_forward()` 已经真正命中 `deepspeed_checkpoint` 分支。
- 修掉了 deepspeed checkpoint 路径里的输入/上下文传递问题，不再把 `DiT` 计算本身改掉。
- 修掉了 fixed prompt 在 `batch_size>1` 时没有按 batch 展开的错误。
- 修掉了 8 卡 `accelerator.prepare` 卡住的问题，关键是给 deepspeed 明确补齐 `train_micro_batch_size_per_gpu`。

## 8 卡训练现状

- 8 卡训练已经真正跑出过 `loss`，不是只停在初始化阶段。
- 当前稳定起跑的版本先关闭了 validation，避免在 `step-0` 就卡死。
- 现阶段说明主训练链路已经打通，后面重点从“能不能跑”转成“显存优化和长跑稳定性”。

## 推理测试现状

- 新增了 `wanvideo/model_inference/flashvsr/`，用于测试训练导出的 `step-xxx.safetensors`。
- 已确认训练导出的 `step-200.safetensors` 不是完整模型。
- 当前 ckpt 只包含：
  - `lq_proj_in.*`
  - `dit` 上的 LoRA 权重
- 已确认推理时能正确加载：
  - base Wan 1.3B
  - `lq_proj_in`
  - LoRA
- 已确认推理时也真实走到了 `flash_attn_2`。
- 已补充 `step-400` 的外部视频推理测试，结果放在 `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_step400_20260408_013404/`。
- 外部视频推理已确认能完整落出：
  - `sr.mp4`
  - `sr_lq.mp4`
- 外部视频测试里出现了“能看到输入影子，但结果非常淡、接近静态白色视频”的问题，因此继续拆分测试路径。
- 新增了“训练内 validation 同源推理”脚本：
  - `infer_flashvsr_validation_style.py`
  - 它直接复用训练里的 `collect_fixed_validation_samples()` 和 `lq_tensor` 路径，不再依赖磁盘里额外压过一遍的 `lq.mp4`。
- 新增了“外部 mp4 但走 tensor 风格输入”的脚本：
  - `infer_flashvsr_external_mp4_tensor_style.py`
  - 用来把“训练集同源测试”和“外部视频测试”彻底分开。

## 训练代码

- 搭好了 `FlashVSR Stage 1` 训练入口。
- 训练参数改成支持 `yaml` 配置。
- `prompt_tensor_path` 改成支持从 `yaml` 读取，不再强制只能命令行传。
- 接上了 `lq_proj_in`、固定 prompt、`dit lora`。
- 训练时会打印 `loss`。
- 接上了在线 validation，支持保存 `hr.mp4 / lq.mp4 / sr.mp4`。
- 接上了 `wandb` 参数。

## 数据代码

- `storymotion` 支持从轻量 `jsonl manifest` 读取。
- `takano` 支持直接读 tar shard。
- 两路数据都能做最小测试。
- 数据集支持统一 `global_seed`。
- 测试输出从 `gif` 改成 `mp4`。

## 退化

- 从简化退化改成了 `RealESRGAN / RealBasicVSR` 风格退化。
- 退化配置单独放在 `wanvideo/data/flashvsr/degradation/configs/`。

## 当前结论

- 数据链路已经打通。
- 在线 validation 已经单独验证通过。
- `DeepSpeed ZeRO-2 + gradient checkpointing` 路线已经接上。
- 现在的核心瓶颈是大分辨率长序列训练时的显存优化。
- 已补充从 `run.log` 提取 loss 并画图的小工具：
  - `plot_flashvsr_loss_from_log.py`
- 对实验 `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_8gpu_debug_noval_20260407_144928` 已导出：
  - `analysis/loss_points.csv`
  - `analysis/loss_curve.png`
- 当前更关键的问题已经从“训练起不来”变成：
  - 训练虽已能出稳定 loss
  - 但 `step-400` 的结果仍然过于模糊
  - 需要继续判断这是训练本身没学到位，还是测试/validation 路径还有未对齐的地方

## 2026-04-08 补充

- 已新增两条更严格的推理检查路径：
  - `infer_flashvsr_validation_style.py`
    - 直接复用训练里 `collect_fixed_validation_samples()` 取出的 `lq_tensor`
    - 尽量和训练内 validation 完全一致
  - `infer_flashvsr_external_mp4_tensor_style.py`
    - 外部 mp4 先转成 tensor，再按训练更接近的路径喂给 `infer_from_lq`
- 训练同源 validation-style 测试已落盘：
  - `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_validation_style_step400_20260408_014221`
  - 已有 `hr.mp4 / lq.mp4 / sr.mp4 / meta.json / run.log`
- 五个外部 mp4 tensor-style 批量测试已落盘：
  - `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_external_tensor_batch_20260408_031052`
  - 每个视频目录都有 `input_lq.mp4 / sr.mp4 / meta.json / run.log`
- 对照 FlashVSR 原仓后确认：
  - 当前 `FlashVSRLQProjIn` 的结构、hidden dim、causal conv、pixel shuffle 和逐层线性投影，与原仓 `Causal_LQ4x_Proj` 基本一致
  - 当前没有看到明显的 `lq_proj_in` 结构级错误
- 已新增训练前张量探针：
  - 可通过 `--debug_tensor_dump_dir` 开启
  - 会在真正进模型前落：
    - `00_input_hr_tensor`
    - `01_input_lq_tensor`
    - `02_preprocessed_lq_tensor`
    - `03_lq_proj_latents`
    - `04_model_token_alignment`
  - 用于核查：
    - 通道顺序
    - 数值范围
    - preprocess 前后是否异常
    - `x` 与 `lq_latents` 的 token 对齐是否符合 FlashVSR 预期
- 已在原始 `FlashVSR` 推理链路中加入同类 debug dump，但默认关闭，不影响原功能：
  - `examples/WanVSR/infer_flashvsr_full_cloud.py`
  - `diffsynth/pipelines/flashvsr_full.py`
  - `flashvsr_inference_cloud_full.sh`
- 原仓真实推理调试已跑通：
  - `/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_original_debug_20260408_051536`
  - 已生成：
    - `FlashVSR_v1.1_Full_windpower_seed0.mp4`
    - `debug/00_input_lq_video.json`
    - `debug/01_initial_noise.json`
    - `debug/02_lq_proj_latents_first_window.json`
    - `debug/03_model_token_alignment.json`
- 原仓推理关键统计：
  - 输入 `LQ_video`：
    - shape `[1, 3, 81, 1024, 1024]`
    - range `[-1, 1]`
  - 初始 noise：
    - shape `[1, 16, 20, 128, 128]`
    - std `~0.999`
  - `LQ_proj_in` 第一层输出：
    - shape `[1, 24576, 1536]`
    - std `~0.0837`
  - `patchify` 后模型 token：
    - shape `[1, 24576, 1536]`
    - 对齐后的 `LQ_latents` 也是 `[1, 24576, 1536]`
- 和训练探针对比后，当前可以进一步排除：
  - 原仓推理与我们训练路径在 `LQ` 基本值域上的根本不一致
  - `LQ_proj_in` 输出维度级别不一致
  - `patchify` 后 token 对齐维度不一致

## 2026-04-08 Release 风格训练对齐

- 基于公开 release 推理结构新增了单独训练配置：
  - `stage1_release_smoke_2gpu.yaml`
  - `stage1_release_8gpu.yaml`
  - 核心改动是 `lq_proj_layer_num: 1`
- 公开 release 与之前训练版的关键差异已确认：
  - 公开 release 推理 `LQ_proj_in` 只输出 `1` 组条件特征
  - 我们之前训练版是 `30` 层逐层注入
  - 因此公开 release 结构和论文中“30 层依次注入”的说法并不完全一致
- release 风格 `2` 卡 smoke 已经真正跑通：
  - 实验目录：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_20260408_122919`
  - 已确认：
    - `validation_at_start` 成功执行
    - validation 后训练继续进行
    - 训练路径真实命中 `gradient_checkpoint`
    - 已出现第一条 loss：`step=1 loss=1.358621`
- 期间定位并修掉的关键 bug：
  - validation 之后 scheduler 没有完整切回训练态
  - 之前只恢复了 `training` 布尔值，没有恢复训练用 `sigmas/timesteps`
  - 导致训练阶段在 `add_noise()` 里出现：
    - `IndexError: index ... is out of bounds for dimension 0 with size 1`
  - 修复方式：
    - 在训练 `forward()` 前强制 `scheduler.set_timesteps(1000, training=True, shift=5.0)`
    - validation 结束后如果原来是训练态，也显式切回同样的训练 scheduler 状态
- 当前结论：
  - release 风格训练已经不是“起不来”的问题
  - 公开 release 结构确实能带 validation 并继续训练

## 2026-04-09 Step-0 初始化排查

- 针对“step-0 validation 直接是 noise 马赛克”的怀疑，重点检查了 LoRA 和 `lq_proj_in` 的初始化。
- 远程训练环境确认：
  - `PEFT LoraConfig(init_lora_weights=True)` 默认值就是 `True`
  - 这说明 LoRA 默认是零影响初始化，不是当前 step-0 异常的主因
- 进一步确认当前训练代码里的真正问题在 `lq_proj_in`：
  - `FlashVSRLQProjIn` 的最终 `linear_layers` 之前是标准随机初始化
  - 这会导致即使 step-0 尚未训练，控制支路也已经在主动扰动底模
- 已修复：
  - 在 `DiffusionTrainingModule.add_lora_to_model()` 中显式写死 `init_lora_weights=True`
  - 在 `FlashVSRLQProjIn` 中默认将最终输出投影层清零初始化
  - 若从 `lq_proj_checkpoint` 恢复权重，则不覆盖 checkpoint 内容
- 新的 2 卡 release smoke 已启动：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_smoke_2gpu_zeroinit_20260408_125119`
- 这轮结果说明：
  - `step-0 validation` 已正常进入并跑到 `sr` 的 VAE decode
  - 说明零初始化后，validation 不再是“一开始就纯噪声崩掉”的状态
  - 后续失败发生在训练第一步 `backward` 的 OOM，不是 validation 先挂
- 当前结论：
  - LoRA 初始化不是主问题
  - `lq_proj_in` 非零初始化之前确实会污染 step-0 表现
  - 后续应基于这版“LoRA 零影响 + lq_proj_in 输出零初始化”的 release 训练继续验证

## 2026-04-09 Early validation checkpoints

- 为 release 8 卡训练新增 `extra_save_steps`，可在 `10,25,50,100` 这些早期 step 额外保存 checkpoint 并触发 validation，之后继续按 `save_steps=200` 节奏运行。
- 修改文件：
  - `diffsynth/diffusion/parsers.py`
  - `diffsynth/diffusion/logger.py`
  - `diffsynth/diffusion/runner.py`
  - `wanvideo/model_training/flashvsr/configs/stage1_release_8gpu.yaml`
- 目的：在不改变后续主训练节奏的前提下，快速观察 step 10/25/50/100 的 SR 变化，判断训练是否健康。

## 2026-04-09 Paper reading correction and release run diagnosis

- 更正对论文的理解：用户指出得对，当前已不再把“30 层逐层注入”当作论文明确写出的结论。用户给出的论文描述明确的是 `LQ_proj_in` 的结构：按 4 帧 clip，经像素重排与两层时序压缩的 `CausalConv3d` 后投影到与 DiT patch token 对齐的空间，再与 patchified latent tokens 做逐元素相加。`30` 层注入来自当前代码/实现设计，而不是这段论文文字本身的直接表述。
- `train_stage1_release_8gpu_20260408_135241` 的早期 validation 检查点目录已生成：
  - `step-0`
  - `step-10`
  - `step-25`
  - `step-50`
  - `step-100`
- `step-100.safetensors` 权重统计已确认：
  - `lq_proj_in` 并非未参与训练
  - `LoRA` 也并非未参与训练
  - 其中 `lq_proj_in.linear_layers.0.weight/bias` 已从零初始化状态被更新，但量级仍较小：
    - weight `abs_mean ≈ 2.76e-05`
    - bias `abs_mean ≈ 5.86e-05`
  - `LoRA` 平均绝对值约 `0.00644`
- 当前判断：`release` 风格训练出现“loss 低且不再下降、validation 与输入关系极弱”的问题，不能用“projection 层没参与训练”解释；更像是训练/validation 路径或目标设置仍存在不对齐。

## 2026-04-09 Stage1 inference alignment
- 确认此前 `infer_flashvsr_stage1.py` / `infer_flashvsr_external_mp4_tensor_style.py` 默认按 `len(pipe.dit.blocks)=30` 构建 `lq_proj_in`，这与 `stage1_release_*` 训练里 `lq_proj_layer_num=1` 不一致。
- 已修改离线推理脚本：从 ckpt 中自动推断 `lq_proj_layer_num`，首层注入版 ckpt 会按 `1` 层结构构建 `lq_proj_in`。
- 正在远程运行 `step-200 + windpower + 50步` 的 Stage1 对齐测试：`/mnt/task_wrapper/user_output/artifacts/inference/flashvsr_stage1_step200_50steps_20260408_200939`。
- 当前已确认：base Wan / LoRA / 外部 mp4 tensor-style / flash-attn 都已进入推理；结果目录已写出 `input_lq.mp4` 和 `run.log`，`sr.mp4` 仍在等待最终落盘或继续排查卡点。

## 2026-04-10 Prompt / validation / v1 复原

- 纯 `Wan` 基线对比继续推进后，确认：
  - `WanVideoPipeline` 本体没有被改坏。
  - 长 prompt 和 `cfg=1`/小 `cfg` 组合容易出现棕色视频。
  - 短英文 prompt 在纯 Wan 下能正常出视频。
- 由此判断：
  - 之前训练里的 `fixed prompt tensor + cfg=1` validation 不能再直接当作“训练必然正确/错误”的唯一依据。
- 为了和用户记忆中的旧逻辑重新对齐，新增了一套 **v1 版本**：
  - `train_flashvsr_stage1_v1.py`
  - `stage1_release_8gpu_v1.yaml`
  - `stage1_release_smoke_2gpu_v1.yaml`
  - `FlashVSR-Stage1-Release-8GPU-v1.sh`
  - `FlashVSR-Stage1-Release-Smoke-2GPU-v1.sh`
  - `infer_flashvsr_stage1_v1.py`
- 这套 `v1` 的意图是：
  - 复原 `train_stage1_release_8gpu_20260408_135241` 这一时期的核心思路
  - `lq_proj_in` 只做首层注入
  - 训练/validation 中不走后续新增的首层流式 `stream_forward` 拼接，而是整段 `fullclip` 投影
  - validation 仍沿用 `posi_prompt.pth + cfg=1`
  - validation 推理步数统一提升为 `50`
- 同时补了一份代码地图，避免继续混用脚本：
  - `docs/FLASHVSR_CODEMAP.md`

## 2026-04-10 V2 validation 对照版

- 新增 `V2` 版本，目标不是继续改训练主链，而是专门对照 `V1` 的 validation 行为。
- 新文件：
  - `train_flashvsr_stage1_v2.py`
  - `stage1_release_8gpu_v2.yaml`
  - `stage1_release_smoke_2gpu_v2.yaml`
  - `FlashVSR-Stage1-Release-8GPU-v2.sh`
  - `FlashVSR-Stage1-Release-Smoke-2GPU-v2.sh`
  - `infer_flashvsr_stage1_v2.py`
- `V2` 的核心思路已重新定义为：
  - 训练主链保持 `V1`
  - validation / inference 不再沿用旧的 `FlashVSRStage1Pipeline.infer_from_lq()`
  - 改为以 `Wan` 的 fixed-prompt 基线为起点，再接入：
    - 当前训练得到的 LoRA
    - 当前训练得到的 `lq_proj_in`
    - 首层注入
    - 非流式 `fullclip`
    - `posi_prompt.pth`
    - `cfg=1`
    - `50` 步
- 这样 `V1` / `V2` 的差异主要收敛到：
  - `V1`：旧 stage1 validation 链
  - `V2`：Wan fixed-prompt 基线 + stage1 条件支路

## 2026-04-10 记录要求

- 新增约束：训练 / 测试的重大改动需要及时更新 worklog，避免版本分叉后无法追踪到底哪套脚本和配置对应哪条实验。

## 2026-04-10 Scripts and cached eval samples

- 新增 `scripts/` 目录，用来放训练辅助脚本，避免继续散落在 `tools/` 和临时命令里：
  - `scripts/plot_flashvsr_loss_from_log.py`
  - `scripts/export_flashvsr_eval_samples.py`
- `plot_flashvsr_loss_from_log.py`：
  - 从 `run.log` 提取 `step/loss`
  - 导出 `csv/png`
- `export_flashvsr_eval_samples.py`：
  - 直接按训练 yaml 构建 dataset
  - 固定抽取 `hq/lq` 样本
  - 导出 `hr.mp4 / lq.mp4 / hr.pt / lq.pt / meta.json`
- 目的：
  - 后续 smoke / validation 不必每次都在线从 dataset 首次取样
  - 先把固定 `hq/lq` 样本缓存到 `artifacts`
  - 再让训练/推理复用这些固定样本，减少卡在 sample collection 的时间

## 2026-04-10 V2 deep dive doc

- 新增 `docs/FLASHVSR_V2_DEEPDIVE.md`
- 作用：
  - 专门解释 `train_flashvsr_stage1_v2.py`
  - 覆盖 `v2` 的模块职责、数据流、validation 两个分支、启动脚本快照机制、以及与 `v1` 的真实差异
- 目的：
  - 避免继续靠口头回忆理解 `v2`
  - 让后续 debug 时能明确知道“训练主链”和“validation 诊断链”分别在哪里变了

## 2026-04-10 V2 wantextdog crash fix

- `train_stage1_release_smoke_2gpu_v2_wantextdog_20260409_112420` 报错定位完成。
- 根因：
  - `train_flashvsr_stage1_v2.py` 里 `WanTextPromptLQPipeline.from_pretrained()` 使用了 `WanVideoUnit_PromptEmbedder()`
  - 但文件顶部没有 import 这个类
  - 运行到文本 validation 分支时直接 `NameError`
- 修复：
  - 在 `train_flashvsr_stage1_v2.py` 顶部补上 `WanVideoUnit_PromptEmbedder` 的 import
- 结论：
  - 这次 crash 不是模型/validation 数学问题
  - 是一个明确的代码遗漏，修完后需要重启 `v2` 文本 smoke 再继续看 step-0 结果

## 2026-04-10 V2 8GPU long-run config reset

- 按当前结论，将 `v2` 的 8 卡长期训练重新收敛成固定版本：
  - `batch_size=2`
  - `save_steps=100`
  - `validation_at_start=true`
  - validation 沿用 `cfg=1 + posi_prompt.pth + 50步`
  - `wandb` 开启
- 同时移除了 8 卡 launcher 里默认导出的 `FLASHVSR_DEBUG_DIR`，避免长期实验继续生成大批 debug 噪音文件。
- 对应文件：
  - `configs/stage1_release_8gpu_v2.yaml`
  - `lora/FlashVSR-Stage1-Release-8GPU-v2.sh`

## 2026-04-12 FlashVSR original ratio and V2 alpha scan

- 为了直接比较原始 `FlashVSR` 与当前 `v2` 的控制信号强度，给原始 `FlashVSR` full 推理链新增了**默认关闭**的统计导出：
  - 文件：`FlashVSR/diffsynth/pipelines/flashvsr_full.py`
  - 开关：`FLASHVSR_DEBUG_STATS=1`
  - 新增导出：`04_model_token_stats.json`
  - 统计内容：
    - patchify 后 `x`
    - 对齐后的 `LQ_latents[0]`
    - `ratio_std_lq_to_x`
    - `ratio_absmean_lq_to_x`
- 同时给 `lucidvsr` 的 `v2` inference 新增了可控注入强度：
  - 文件：`wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py`
  - 新参数：`--projection_scale`
  - 配套 shell：`wanvideo/model_inference/flashvsr/run_test_flashvsr_stage1_v2.sh`
- 当前已拿到两组关键证据：
  - 我们的 `v2 step-500`：
    - `ratio_std_lq_to_x ≈ 0.0533`
    - `ratio_absmean_lq_to_x ≈ 0.0516`
  - 原始 `FlashVSR full`（默认输入目录中的样本）：
    - `ratio_std_lq_to_x ≈ 0.2608`
    - `ratio_absmean_lq_to_x ≈ 0.2640`
- 这说明：
  - 当前 `v2` 首层注入的 `LQ` 信号量级只有主干 token 的约 `5%`
  - 原始 `FlashVSR full` 对齐后的 `LQ` 信号量级约为主干 token 的 `26%`
  - 两者有大约 `5x` 的差距
  - 这与现象高度一致：
    - `v2` 能学到颜色
    - 但难以形成强结构控制
- 已经在远程发起新的对照任务：
  - 原始 `FlashVSR` 同一输入对照（`sample_000/lq.mp4`）
  - `v2 step-800` 的 `projection_scale` 扫描：
    - `alpha = 0, 1, 2, 4, 8, 16`
  - 目的：
    - 判断结构控制不足是“训练没学到”
    - 还是“注入强度太弱、进入第一层后迅速被主干淹没”

## 2026-04-12 V2 Debug Overfit Line

- 新增独立 debug 训练入口：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2_debug.py`
- 设计目标：
  - 不改现有 `v2` 主线源码行为
  - 复用 `v2` 的模型、LoRA、projection、validation 逻辑
  - 只把数据入口换成固定样本：
    - 读取 `sample_xxx/hr.pt`
    - 读取 `sample_xxx/lq.pt`
    - 无限循环过拟合
- 新增配置：
  - `configs/stage1_release_8gpu_v2_debug_overfit.yaml`
  - `configs/stage1_release_smoke_2gpu_v2_debug_overfit.yaml`
- 新增启动脚本：
  - `lora/FlashVSR-Stage1-Release-8GPU-v2-Debug.sh`
  - `lora/FlashVSR-Stage1-Release-Smoke-2GPU-v2-Debug.sh`
- 新增脚本 override：
  - `BATCH_SIZE_OVERRIDE`
  - `MAX_TRAIN_STEPS_OVERRIDE`
  - `LQ_PROJ_SCALE_OVERRIDE`
- 首次 smoke 失败原因：
  - debug 版 2GPU 启动脚本错误地把 `accelerate` yaml 改成了顶层 `deepspeed_config_file`
  - 远程 `accelerate` 版本不接受这个顶层键
- 已修复：
  - 改成与已工作的 17 帧脚本相同的“字符串替换 deepspeed json 绝对路径”方式
- 当前远程 overfit smoke：
  - `train_stage1_release_smoke_2gpu_v2_debug_overfit_bs16_20260411_112805`
  - `train_stage1_release_smoke_2gpu_v2_debug_overfit_bs8_20260411_112806`
  - `train_stage1_release_smoke_2gpu_v2_debug_overfit_bs4_20260411_112806`
  - 三条都已经真正启动，仍在初始化阶段，尚未到 `step-0` 产物
- 2026-04-12:
  - 新增固定启动文件，避免再靠命令行临时覆盖导致实验配置混乱：
    - `configs/stage1_release_8gpu_v2_debug_overfit_17f_bs24.yaml`
    - `lora/FlashVSR-Stage1-Release-8GPU-v2-Debug-17f-bs24.sh`
    - `lora/history/20260412_overfit_17f_bs24.md`
  - 结论更新：
    - `8卡/17帧/alpha=5` 下，`bs=72` 与 `bs=48` 都会在真正训练前向阶段 OOM，不是 validation 假象。
    - 用户手动启动的 `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_8gpu_v2_debug_overfit_20260411_123719` 实际吃到的是 `stage1_release_smoke_2gpu_v2_debug_overfit_17f.yaml`，但训练链是通的，已跑到 `step=18`。
  - 后续调整：
    - `history/stage1_release_8gpu_v2_debug_overfit_17f_bs24.yaml` 的学习率从 `4e-5` 下调到 `1e-5`。
    - 原因：`bs24` overfit 训练中 loss 有明显上冲，先用更保守学习率验证是否为优化不稳定，而不是代码路径问题。
  - 新增全量训练集 17 帧实验：
    - `configs/history/stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5.yaml`
    - `lora/history/FlashVSR-Stage1-Release-8GPU-v2-17f-Full-bs4-lr1e5-alpha5.sh`
  - 目的：
    - 保持 `17帧 / alpha=5 / 50步validation / posi_prompt / cfg=1 / 8卡`
    - 把数据入口切回正常训练集
    - `batch_size=4`
    - `learning_rate=1e-5`
  - 为排查 `8卡 + validation_at_start` 初始化卡住问题，新增一条仅关闭起始 validation 的版本：
    - `configs/history/stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5_nostartval.yaml`
    - `lora/history/FlashVSR-Stage1-Release-8GPU-v2-17f-Full-bs4-lr1e5-alpha5-NoStartVal.sh`
  - 设计意图：
    - 只关闭 `validation_at_start`
    - 保留后续 `10/20/50/100...` 的 checkpoint 与 validation
    - 用于验证训练主链是否先能稳定出第一条 loss，再观察后续 validation 是否仍报错
  - 后续定位到的真实报错：
    - 后续 validation 在 `train_flashvsr_stage1_v2.py` 中调用 `baseline_pipe.load_lora(..., hotload=True)`。
    - 但这些 validation baseline pipe 没启用 VRAM Management，因此会直接报：
      - `ValueError: VRAM Management is not enabled. LoRA hotloading is not supported.`
  - 已修复：
    - `train_flashvsr_stage1_v2.py` 的两个 validation 分支都改成：
      - 每次 validation 新建 fresh baseline pipe
      - 不再使用 `hotload=True`
    - 这样同时解决：
      - `hotload` 直接报错
      - 复用 cached pipe 时 LoRA 累积融合导致结果被污染
  - 修复后已重新启动：
    - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_8gpu_v2_17f_full_bs4_lr1e5_alpha5_nostartval_20260412_023639`
