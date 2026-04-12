# FlashVSR V2 代码深度说明

这份文档只解释 `v2` 这条线：

- 训练主文件：
  `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py`
- 2 卡启动：
  `wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-Smoke-2GPU-v2.sh`
- 8 卡启动：
  `wanvideo/model_training/flashvsr/lora/FlashVSR-Stage1-Release-8GPU-v2.sh`
- 配套离线推理：
  `wanvideo/model_inference/flashvsr/infer_flashvsr_stage1_v2.py`

目标不是只讲“它大概做什么”，而是把：

- 每个类/函数负责什么
- 数据怎么流动
- 训练和 validation 怎么串起来
- `v2` 和 `v1` 差在哪里

尽量讲完整。

## 1. 先说 V2 想解决什么

`v2` 不是全新训练结构。

它的训练主链基本沿用 `v1`：

- base 是 `Wan`
- 训练对象还是：
  - `lq_proj_in`
  - `dit` 上的 LoRA
- `lq_proj_in` 还是首层注入
- 还是非流式 `fullclip`

`v2` 主要改的是 **validation / inference 主链**。

你做 `v2` 的目的，是拿它和 `v1` 对照，检查：

- `v1` 里那条旧 validation 链是不是本身有问题
- 如果把 validation 改成更接近 Wan 基线的方式，`step-0` 会不会恢复正常

所以：

- `v1` 和 `v2` 的核心差异在 validation
- 不是训练主干突然换了一个新架构

## 2. 文件最上面的公共小工具

### `REPO_ROOT` + `sys.path`

目的是保证：

- 直接运行这个训练脚本时
- 仓库根目录已经进了 `sys.path`
- 后面 `diffsynth` / `wanvideo` 这些模块能正确 import

### `CACHE_T = 2`

这是 `CausalConv3d` / `FlashVSRLQProjIn.stream_forward()` 的缓存帧数。

在 `v2` 里它仍然保留，但你当前 `v2` 训练主链并没有把首层注入切成正式流式训练模式。它更多是 `FlashVSRLQProjIn` 自己具备这个能力。

### `_append_flashvsr_debug`

如果环境变量 `FLASHVSR_DEBUG_DIR` 存在，就往那个目录的某个日志文件追加一行文本。

用来记录：

- gradient checkpoint 分支
- flash attention 分支
- 其他一次性 debug 信息

### `_tensor_debug_dir`

调试 tensor dump 的目标目录优先级：

1. `pipe.debug_tensor_dump_dir`
2. 环境变量 `FLASHVSR_TENSOR_DEBUG_DIR`

### `_tensor_to_display_frames`

把训练/validation 里拿到的 tensor 转成可保存的 PIL 帧：

- 输入要求：
  - `[T,C,H,W]`
  - 或 `[B,T,C,H,W]`
- 如果是 `[-1,1]`，先转回 `[0,1]`
- 如果单通道，复制成 3 通道

这是 debug 导出视频用的，不参与模型计算。

### `_dump_tensor_preview_once`

只导出一次某个 tensor 的调试快照。

它会同时写：

- `xxx.json`
  - shape/dtype/device/min/max/mean/std
- `xxx.pt`
  - 原始 tensor
- `xxx.mp4`
  - 可视化视频

这就是你前面调试：

- `00_input_hr_tensor`
- `01_input_lq_tensor`
- `02_preprocessed_lq_tensor`
- `03_lq_proj_latents`
- `04_model_token_alignment`

这些文件的来源。

## 3. `RMS_norm` / `CausalConv3d` / `PixelShuffle3d`

这三块是 `lq_proj_in` 的基础模块。

### `RMS_norm`

这是 projector 里用的 RMSNorm。

它支持：

- channel-first
- image / non-image 两种广播方式

在 projector 里这里用的是：

- `images=False`

因为后面它把某些维度当作 token/channel 特征处理。

### `CausalConv3d`

它继承自 `nn.Conv3d`，但自己接管了 padding。

关键点：

- 时间维只看过去和当前
- 不看未来
- 支持传 `cache_x`
- 有 cache 时会把过去的尾部帧拼到当前片段前面

所以它这里的“causal”含义是：

- 时间卷积不偷看未来帧

不是说模型最终永远只能理解一个小片段。

### `PixelShuffle3d`

这个名字容易误导。

它不是常见超分里的“把通道重排回空间”的那个 2D pixel shuffle。

这里更准确地说是：

- 3D 维度重排
- 把输入视频按
  - `ff=1`
  - `hh=16`
  - `ww=16`
 重新整理进通道

目的不是恢复图像，而是把原始像素域整理成后面 projector 更容易处理的形态。

## 4. `FlashVSRLQProjIn`

这是 `v2` 里最关键的控制支路。

### 构造函数

参数里最重要的是：

- `in_dim`
- `out_dim`
- `layer_num`
- `zero_init_output`

内部结构：

1. `PixelShuffle3d(1,16,16)`
2. `conv1 = CausalConv3d(..., kernel=(4,3,3), stride=(2,1,1))`
3. `norm1 + SiLU`
4. `conv2 = CausalConv3d(..., kernel=(4,3,3), stride=(2,1,1))`
5. `norm2 + SiLU`
6. `linear_layers = ModuleList([Linear(...)] * layer_num)`

也就是说：

- 空间经过 `16x16` 重排
- 时间经过两次 `2x` 压缩
- 最后通过线性层投影到和 `DiT` token 一样的维度

### `zero_init_output_projection`

把所有 `linear_layers` 的：

- `weight`
- `bias`

全部清零。

这一步的目的不是让 projector 永远失效，而是让：

- step-0 时它对主干近似零影响

只要前面的 conv 输出不是全零、loss 不是全零，这层后续仍然能学。

### `forward`

这是 projector 的“整段 fullclip”路径。

步骤：

1. `clear_cache()`
2. 取输入总帧数 `t`
3. 先把第一帧复制 3 次补到最前面
4. 再按 `4` 帧一段切
5. 每段调用 `stream_forward()`
6. 把每段结果按 token 维拼接

所以哪怕叫 `forward`，内部其实仍然是：

- 通过 `stream_forward()` 拼出来的

### `clear_cache`

重置：

- `conv1` cache
- `conv2` cache
- `clip_idx`

### `stream_forward`

这是 projector 的流式核心。

第一次调用：

- 会给第一段前面补 3 帧首帧
- 跑 `pixel_shuffle`
- 跑 `conv1`
- 只 warmup，不直接产出最终 token

后续调用：

- 用 cache 拼回去
- 跑两层 causal conv
- reshape 成 `[B, tokens, hidden]`
- 每个 `linear_layer` 各自产出一份 token

这里为什么返回的是 list：

- 因为每层注入都可以有一份不同 token

在你现在的 `v2` 里：

- `layer_num=1`
- 所以这个 list 实际上长度是 1

## 5. `_build_release_style_lq_latents`

这是一个辅助函数。

它直接手工按 `4` 帧切 `lq_video`，每段调用 `stream_forward()`，再拼接输出。

它的目标是更接近公开 `FlashVSR` release 的首层流式特征构造方式。

但要注意：

- 当前 `v2` 训练主链并没有默认强制用它
- 它更多是为对齐实验保留的工具函数

## 6. `FlashVSRUnit_FixedPrompt`

这是旧 stage1 validation 风格的 fixed prompt 单元。

做的事情：

1. 如果 `pipe.fixed_prompt_tensor` 还没缓存
2. 就从 `prompt_tensor_path` 里 `torch.load`
3. 转到当前 device / dtype
4. 输出：
   - `context`

也就是说，它输出的是：

- **原始 prompt tensor**
- 还没过 `dit.text_embedding`

## 7. `WanFixedPromptEmbeddedUnit`

这是 `v2` 里新加的固定 prompt 单元。

和上一个最大的区别是：

- 它不是直接输出 `context`
- 而是先：
  - `dit.text_embedding(raw_context)`
- 再输出：
  - `embedded_context`

这一步是为了更贴近你后来验证出来的 fixed-prompt 接法。

## 8. `WanTextPromptLQPipeline`

这是 `v2` 的一个关键 validation/inference pipeline。

它不是纯 Wan。

它是：

- Wan 文本推理主链
- + `lq_proj_in`
- + stage1 的 `model_fn`

### `from_pretrained`

做的事情：

1. 先用 `WanVideoPipeline.from_pretrained(...)` 构造纯 Wan pipe
2. 把类替换成 `WanTextPromptLQPipeline`
3. 把 `units` 改成：
   - `WanVideoUnit_ShapeChecker`
   - `WanVideoUnit_NoiseInitializer`
   - `WanVideoUnit_PromptEmbedder`
   - `FlashVSRUnit_LQVideoEmbedder`
4. `post_units = []`
5. `in_iteration_models = ("dit",)`
6. `model_fn = flashvsr_stage1_model_fn`
7. 创建自己的 `lq_proj_in`

注意：

- 这里 `tokenizer_config` 是启用文本 prompt 所必须的
- `zero_init_output=False`
  - 因为这条 pipe 是 validation/inference pipe
  - 它预期后续会加载训练得到的 projector 权重

### `infer_from_lq_text`

这是 `Wan` 文本 baseline + stage1 条件的核心推理函数。

流程：

1. `scheduler.set_timesteps(num_inference_steps, shift=5.0)`
2. 构造 `inputs_shared`
   - `input_video=None`
   - `lq_video=lq_video`
   - `cfg_scale=cfg_scale`
3. 构造：
   - `inputs_posi={"prompt": prompt}`
   - `inputs_nega={"negative_prompt": negative_prompt}`
4. 依次跑 `units`
5. 如果还没 `latents`，就用 `noise`
6. 进入采样循环
7. `cfg!=1` 时跑正负两支
8. `scheduler.step(...)`
9. 最后 `vae.decode(...)`

这是你现在拿来和 `cfg=1` / `cfg=5` 做对照的主要文本版 validation/inference 主链。

## 9. `FlashVSRUnit_LQVideoEmbedder`

这个 unit 的职责是：

- 把 `lq_video` 预处理
- 送进 `lq_proj_in`
- 得到 `lq_latents`

### 输入输出

输入：

- `lq_video`
- `height`
- `width`

输出：

- `lq_latents`

### 具体逻辑

1. 如果 `lq_video` 本来就是 tensor
   - 先 dump 原始输入 tensor
2. `pipe.preprocess_video(...)`
3. dump 预处理后的 tensor
4. 把它转成当前 device / dtype
5. 调：
   - `pipe.lq_proj_in(lq_input)`
6. 如果有结果，再写一份统计：
   - 层数
   - shape
   - dtype
   - min/max/mean

注意：

- 当前默认还是直接 `pipe.lq_proj_in(lq_input)`
- 也就是 **fullclip 路**
- 不是默认强制 `_build_release_style_lq_latents()`

## 10. `flashvsr_stage1_model_fn`

这是最核心的前向函数之一。

它是：

- 文本/普通 context 版

输入最重要的是：

- `dit`
- `latents`
- `timestep`
- `context`
- `lq_latents`

### 内部步骤

1. 时间嵌入：
   - `sinusoidal_embedding_1d`
   - `dit.time_embedding`
   - `dit.time_projection`

2. 处理 `context`
   - 如果是 2D，补 batch 维
   - 如果 batch=1 但 video batch>1，则 expand
   - 最后：
     - `context = dit.text_embedding(context)`

3. `patchify(latents)`
   - 得到 token `x`
   - 同时拿到 `(f,h,w)`

4. 构造 `freqs`
   - 用于 rope / video position

5. 对齐 `lq_latents`
   - 如果 token 数不一致
   - 按 frame token 数量做 pad 或 trim

6. 进入每个 `dit.blocks`
   - 如果当前 block 有对应 `lq_latents`
   - 就先：
     - `x = x + lq_latents[block_id]`

7. block 前向
   - 训练时走 `gradient_checkpoint_forward`
   - eval 时直接 `block(...)`

8. `dit.head`
9. `unpatchify`

这就是：

- stage1 条件到底是怎么进入 DiT 的

最关键的一句其实就是：

- `x = x + lq_latents[block_id]`

## 11. `flashvsr_stage1_fixed_prompt_model_fn`

这个函数和上一个非常像，但区别是：

- 它吃的是 `embedded_context`
- 不再内部做：
  - `dit.text_embedding(context)`

也就是说：

- 这条是为 fixed-prompt 已经 embedding 过的 validation/inference 链准备的

其余：

- patchify
- freq 构造
- `lq_latents` 对齐
- block 注入
- checkpoint
- head/unpatchify

都和上面的逻辑基本一样。

## 12. `FlashVSRStage1Pipeline`

这是传统 stage1 训练/validation 的主 pipeline。

### `from_pretrained`

主要做这些：

1. 基于 `WanVideoPipeline.from_pretrained(...)` 建 pipe
2. 改类为 `FlashVSRStage1Pipeline`
3. 存：
   - `prompt_tensor_path`
   - `fixed_prompt_tensor`
4. `units` 变成：
   - `WanVideoUnit_ShapeChecker`
   - `WanVideoUnit_NoiseInitializer`
   - `FlashVSRUnit_FixedPrompt`
   - `WanVideoUnit_InputVideoEmbedder`
   - `FlashVSRUnit_LQVideoEmbedder`
5. `model_fn = flashvsr_stage1_model_fn`
6. 创建 `lq_proj_in`

注意这里有一条很关键：

- `zero_init_output=zero_init_lq_proj_in and lq_proj_checkpoint is None`

也就是：

- 没有 projector checkpoint 时
- 可以按 zero-init 逻辑初始化输出层

### `infer_from_lq`

这是 `v1` 风格 validation 的核心。

特点：

- `input_video=None`
- `lq_video=lq_video`
- `cfg_scale=1.0`
- `FlashVSRUnit_FixedPrompt`
- 10/50 步推理

这条就是你之前一直怀疑有问题的旧 validation 链。

## 13. `WanFixedPromptFlashVSRStage1Pipeline`

这是 `v2` 的 fixed-prompt validation 主 pipeline。

它的定位是：

- Wan 固定 prompt 基线
- + stage1 的 LoRA / `lq_proj_in`

### `from_pretrained`

和 `FlashVSRStage1Pipeline` 不同的是：

- 不再用 `FlashVSRUnit_FixedPrompt`
- 改成：
  - `WanFixedPromptEmbeddedUnit`
- `units` 只保留：
  - `ShapeChecker`
  - `NoiseInitializer`
  - `WanFixedPromptEmbeddedUnit`
  - `FlashVSRUnit_LQVideoEmbedder`
- `model_fn = flashvsr_stage1_fixed_prompt_model_fn`

也就是说这条 fixed-prompt path 的核心变化是：

- prompt tensor 先走 `dit.text_embedding`
- 然后把 embedding 后的结果作为 `embedded_context`

### `infer_from_lq`

这条和 `WanTextPromptLQPipeline.infer_from_lq_text` 很像：

- `input_video=None`
- `lq_video=lq_video`
- `cfg=1`
- 50 步
- 用 fixed embedded prompt
- 只跑单支

这就是 `v2 cfg=1` 那条 smoke 的主要 validation 路。

## 14. `flashvsr_stage1_export`

这个函数用来把训练时 `state_dict` 转成 validation/inference 更容易读的格式。

规则：

- `pipe.dit.xxx` -> `xxx`
- `pipe.lq_proj_in.xxx` -> `lq_proj_in.xxx`
- 其他保持原样

作用：

- 后面的 callback 可以方便拆出：
  - projector 权重
  - LoRA 权重

## 15. `FlashVSRStage1TrainingModule`

这是训练模块总入口。

### `__init__`

主要做这些事：

1. 解析模型配置
2. 创建 `FlashVSRStage1Pipeline`
3. 保存 debug dump 目录
4. 用 `split_pipeline_units(...)` 把 pipe 切成训练模块
5. 用 `switch_pipe_to_training_mode(...)`
   - freeze/unfreeze
   - 注入 LoRA
   - 把可训练部分切到 train mode
6. 保存：
   - `use_gradient_checkpointing`
   - `use_gradient_checkpointing_offload`

注意：

- `v2` 训练主链本身仍然是这个 module
- 没有换成 WanTextPromptLQPipeline 做训练

### `get_pipeline_inputs`

从 dataset sample 中组装训练输入。

它会：

- 从 `data["video"]` 拿 HR
- 从 `data["lq_video"]` 拿 LQ
- 自动推导：
  - `height`
  - `width`
  - `num_frames`
- 把训练用的 shared inputs 准备好：
  - `input_video`
  - `lq_video`
  - `cfg_scale=1.0`
  - `use_gradient_checkpointing`
  - `seed=0`
  - 等

### `forward`

训练一步真正做的事情：

1. 如果没传 `inputs`，就先 `get_pipeline_inputs`
2. `pipe.scheduler.set_timesteps(1000, training=True, shift=5.0)`
3. `transfer_data_to_device`
4. 依次跑 `pipe.units`
5. 调：
   - `FlowMatchSFTLoss(self.pipe, **inputs...)`

也就是说训练 loss 还是建立在：

- `FlashVSRStage1Pipeline`
- 以及 stage1 的主 model_fn

## 16. parser / config 相关

### `flashvsr_parser`

这里定义了训练用所有 CLI 参数。

重点包括：

- `--config`
- `--prompt_tensor_path`
- `--lq_proj_checkpoint`
- `--lq_proj_layer_num`
- `--zero_init_lq_proj_in`
- streaming dataset 相关
- validation 相关
- `validation_prompt_file`
- `validation_negative_prompt`
- `validation_cfg_scale`
- `validation_use_wan_text_baseline`
- `debug_tensor_dump_dir`

### `_flatten_flashvsr_config`

把 yaml 的分组结构扁平化。

合并顺序是：

- `data`
- `model`
- `train`
- `lora`
- `validation`
- `output`
- `wandb`
- `runtime`

### `parse_flashvsr_args`

参数优先级是：

- CLI > YAML > parser 默认值

流程：

1. 先只读 `--config`
2. 如果有 yaml，就 `set_defaults`
3. 再完整 parse
4. 强制检查：
   - `prompt_tensor_path` 必须存在

### `dump_resolved_args`

把真正解析后的运行参数写到实验目录：

- `resolved_args.json`
- `resolved_args.yaml`

这对回溯实验非常关键。

### `configure_deepspeed_runtime`

这个函数会在 runtime 里把：

- `train_micro_batch_size_per_gpu`
- `gradient_accumulation_steps`
- `train_batch_size`

显式回填到 deepspeed config。

这是前面修过的一个重要点，不然 deepspeed prepare 阶段容易出问题。

## 17. validation 样本准备

### `_tensor_video_to_pil_frames`

把 `[T,C,H,W]` tensor 转成 PIL 帧列表。

是 validation 导出 `hr/lq.mp4` 用的。

### `collect_fixed_validation_samples`

validation 最前面会先从训练集里抽固定 sample。

如果 dataset 是 `FlashVSRStreamingDataset`：

- 它会优先从 `video_file_urls` 顺着取
- 调 `_process_video_bytes(...)`
- 把 sample 缓存成 CPU tensor

如果不是：

- 就直接 `iter(dataset)` 顺着抽

这是你前面看到某些 smoke 卡在：

- `Preparing fixed validation samples...`

的来源。

## 18. `FlashVSRValidationCallback`

这是 `v2` 最关键的对照点。

### 初始化参数

它会保存：

- 输出目录
- 固定 validation samples
- 推理步数 / fps / seed
- `use_wandb`
- validation prompt / negative prompt / cfg
- `validation_use_wan_text_baseline`
- validation 用模型配置
- tokenizer 配置
- prompt tensor path
- `lq_proj_layer_num`

还会缓存两个懒加载 pipe：

- `_v2_validation_pipe`
- `_wan_text_baseline_pipe`

### `_get_v2_validation_pipe`

懒加载：

- `WanFixedPromptFlashVSRStage1Pipeline`

也就是：

- fixed prompt
- cfg=1
- LoRA + projector + lq

### `_get_wan_text_baseline_pipe`

懒加载：

- `WanTextPromptLQPipeline`

也就是：

- 文本 prompt
- cfg 可调
- LoRA + projector + lq

### `__call__`

validation 真正执行时：

1. 先建：
   - `output/validation/step-{step}/sample_xxx`
2. 保存 `hr.mp4`
3. 保存 `lq.mp4`
4. 把当前训练 `model.state_dict()` 导出
5. 拆出：
   - `lq_proj_state`
   - `lora_state`

然后分两支：

#### 分支 A：`validation_use_wan_text_baseline=True`

走：

- `WanTextPromptLQPipeline`

步骤：

1. 加载 projector 权重
2. 清空已有 LoRA
3. 加载当前训练 LoRA
4. 调：
   - `infer_from_lq_text(...)`

#### 分支 B：`validation_use_wan_text_baseline=False`

走：

- `WanFixedPromptFlashVSRStage1Pipeline`

步骤：

1. 加载 projector 权重
2. 清空已有 LoRA
3. 加载当前训练 LoRA
4. 调：
   - `infer_from_lq(...)`

两条都会：

- 产出 `sr_frames`
- 保存 `sr.mp4`
- 写 `meta.json`
- sample 0 时可选同步到 wandb

最后还会恢复训练 pipe 的 scheduler/training 状态。

这就是 `v2` 的本质：

- **训练模块不变**
- **只把 validation/inference 分成两种可对照的 Wan 主链**

## 19. `main`

训练入口做的事情按顺序是：

1. 注册自定义 `excepthook`
2. `parse_flashvsr_args()`
3. 如果有 tensor dump 目录，塞进环境变量
4. 创建 `accelerator`
5. `configure_deepspeed_runtime`
6. 打一些 stage log
7. 主进程写 `resolved_args`
8. 构建 dataset
9. 抽固定 validation sample
10. 构建 `FlashVSRStage1TrainingModule`
11. 如果开 wandb，初始化 wandb
12. `DistributedDataLoader` 包装 dataset
13. 定义 optimizer / lr scheduler
14. `accelerator.prepare(...)`
15. `initialize_deepspeed_gradient_checkpointing`
16. 构造 validation callback
17. `launch_training_task(...)`
18. 训练完关 wandb

你前面大部分分布式初始化、validation callback、prepare 卡住的问题，都发生在这条主链里。

## 20. V2 和 V1 的真正差别

### 相同点

- 训练主模块都是 `FlashVSRStage1TrainingModule`
- 训练主 pipe 都是 `FlashVSRStage1Pipeline`
- `lq_proj_layer_num=1`
- projector 输出 zero-init
- LoRA 还是近零初始化
- 训练 loss 主链没换
- 默认都是首层注入、非流式 fullclip

### 不同点

`V1`：

- validation 走旧 stage1 链
- fixed prompt tensor
- `cfg=1`
- `FlashVSRStage1Pipeline.infer_from_lq()`

`V2`：

- validation 主链不再是旧 stage1 validation
- 改成两个分支可切换：
  - `WanFixedPromptFlashVSRStage1Pipeline`
  - `WanTextPromptLQPipeline`
- 都会带上：
  - 当前训练的 LoRA
  - 当前训练的 projector
  - 当前 sample 的 `lq_video`

所以：

- `V2` 不是纯 Wan baseline
- 也不是单纯“新建一个 pipe”
- 它的目标是：
  - 让 validation 更接近可控的 Wan 主链
  - 但仍然把 stage1 条件支路带进去

## 21. 当前 V2 最值得怀疑的地方

光看代码，当前最容易出怪结果的点有这些：

1. `FlashVSRUnit_LQVideoEmbedder` 现在默认还是 `fullclip`
- 不是 release-style 的严格流式首层对齐

2. `WanFixedPromptEmbeddedUnit`
- prompt tensor 先过 `dit.text_embedding`
- 这条语义是否和你期望完全一致，还要靠实验确认

3. 训练主链和 validation 主链仍然不是同一个 pipeline 类
- 这是 `v2` 故意做的对照
- 但也意味着它天然是“诊断版”，不是最终统一版

4. `collect_fixed_validation_samples`
- 还是可能卡在数据集首次取样

## 22. 怎么读 V2 的实验结果

如果看 `v2 cfg=1`：

- 重点是在测：
  - `posi_prompt.pth`
  - `cfg=1`
  - 首层 `projection`
  - LoRA
  - `lq` 输入
  - 放到 Wan fixed-prompt 主链后会发生什么

如果看 `v2 文本 cfg=5`：

- 重点是在测：
  - 正常文本 prompt
  - `cfg=5`
  - 同样的 LoRA / projector / lq

所以这两条不是为了谁一定最终更好，而是为了诊断：

- 是 fixed prompt 有问题
- 还是 stage1 条件支路本身有问题
- 还是文本链压过了 `lq`

## 23. 一句话总结 V2

`V2` 不是新训练框架。

它是：

- **保留 `V1` 的训练主链**
- **把 validation / inference 主链改成更可控、更能和 Wan 基线做对照的版本**

因此当你觉得 `v2` “怪”时，最该先分开看：

1. 训练主链有没有变
2. validation 主链变了什么
3. 当前看到的异常，是训练问题，还是 validation 诊断逻辑问题

这就是 `v2` 这份代码存在的真正目的。
