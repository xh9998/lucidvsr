# FlashVSR v5 版本迭代记录

这份文档是 `v5` 版本线的技术锚点。

写法要求：

- 不写流水账，流水过程看 `FLASHVSR_WORKLOG.md`
- 这里强调代码结构、变量、实现分叉、废弃原因
- 要能和实验目录、配置文件、训练入口一一对上

## 1. v5 这条线要解决什么

`v5` 不是 `v4` 的小修，而是围绕 image/video joint training 重新拆版本。

`v4` 暴露出的核心问题有三类：

1. image 混进训练后，loss 常常一开始就抬到很高，说明 sample 组织方式和 loss 定义并不稳定。
2. image 和 video 到底应该在 sample 层如何对齐，没有形成单一正确实现。
3. packed / mask / pseudo-video / grouped image 等逻辑混在一起，难以做严格对照。

所以 `v5` 的目标变成：

- 明确把 image/video 联合训练的不同假设拆成不同版本
- 把“数据如何组织”“loss 如何理解”“image branch 如何构造”显式写进代码
- 让 `v5.3` 这类正式对照线与 `v5.3.2` 这种特殊试验线分开，不互相污染

当前涉及的主文件：

- 主训练入口：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
- 特殊试验入口：
  - `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_2_lora.py`
- 数据集：
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v532_yubari_frames.py`
- 基础 streaming 数据类：
  - `wanvideo/data/flashvsr/datasets/streaming_dataset.py`
- 联合 attention 模型：
  - `diffsynth/models/wan_video_dit_joint_v5.py`

## 2. v5.1 / v5.2 / v5.3 的分工

### 2.1 v5.1

定位：

- grouped image sample
- 图像权重按“更接近视频首帧公平”理解

实现锚点：

- 数据集：
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
- 训练端会使用：
  - `collate_image_video_joint_v5`
  - `FlowMatchSFTLossV5`

这条线的基本想法是：

- 一个 sample 里不是简单的 `1 video + 1 image pseudo-video`
- 而是用 grouped image sample 组织 image 部分
- 希望让 image 在 loss 上更接近“视频首帧贡献”

废弃原因：

- grouped image 的 batch 表达复杂
- 中间一度引入了额外 padding / token 对齐压力
- 工程复杂度和显存成本都高
- 最终没有收敛成主线

### 2.2 v5.2

定位：

- 仍然是 grouped image sample
- 但图像权重更接近“整段视频等权”理解

实现上沿用：

- `tar_streaming_dataset_v5.py`
- `collate_image_video_joint_v5`
- `FlowMatchSFTLossV5`

技术点：

- `FlowMatchSFTLossV5` 会读取 `loss_sample_weights`
- 但当前代码里 `_build_v5_sample_loss_weights(...)` 已经明确写成：
  - `return torch.ones(len(sample_kinds), dtype=torch.float32)`
- 注释中明确说明：
  - `V5 uses first-frame fairness, not sample-count fairness.`
  - grouped-image sample 不再额外做 `1/K` 缩放

这意味着：

- 到当前代码状态，`v5.1 / v5.2` 的理论分叉没有继续向前深化
- 反而都停在 grouped image 这套工程路径上

当前判断：

- `v5.1 / v5.2` 不是当前推荐主线
- 代码仍可查，但不建议继续作为正式大训练线

### 2.3 v5.3

定位：

- author-style paired sample
- 一个 sample 明确由两条 branch 组成：
  - `video branch`
  - `image pseudo-video branch`

这是目前 `v5` 的正式主线。

原因：

- 样本语义最清楚
- 对照关系最明确
- 后续 `v5.3` / `v5.3.1` / `v5.3.2` 都可以围绕同一个外层训练框架演化

## 3. v5.3 的代码结构

### 3.1 数据集入口

`v5.3` 的正式数据集类是：

- `FlashVSRTarStreamingDatasetV53`
  - 文件：`wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`

类注释已经写得很明确：

- `Author-style paired dual-branch tar dataset.`
- `Each yielded training sample contains:`
  - `one real video branch`
  - `one image branch expanded into a pseudo-video`

关键初始化参数：

- `yubari_video_tar_url`
- `takano_video_tar_url`
- `yubari_video_prob`
- `takano_video_prob`
- `image_tar_root_url`
- `height`
- `width`
- `num_frames`

关键内部变量：

- `self.yubari_video_prob`
- `self.takano_video_prob`
- `self.image_branch_num_frames = max(1, ((int(num_frames) - 1) // 4) + 1)`
- `self.image_pseudo_video_generator = PseudoVideoGenerator(...)`

这里最重要的技术点是：

- image branch 的 pseudo-video 帧数不是硬编码 5
- 而是由：
  - `((num_frames - 1) // 4) + 1`
  动态决定

所以：

- `17` 帧视频时，image branch 会变成 `5` 帧 pseudo-video
- `89` 帧视频时，image branch 会变成 `23` 帧 pseudo-video

### 3.2 video branch 与 image branch 如何产出

`FlashVSRTarStreamingDatasetV53` 里有三组关键迭代逻辑：

- `_discover_video_source(...)`
- `_iterate_video_source(...)`
- `_image_iterator(...)`

video branch：

- 可以来自：
  - Yubari tar / manifest
  - Takano tar / manifest
- 最终会落到：
  - `_iterate_tar_videos_for_urls(...)`
  - `_iterate_direct_videos_for_urls(...)`

image branch：

- 最初设计就是“读单张图”
- 然后在 `_process_image(...)` 中：
  - 先检查 `_meets_min_resolution(...)`
  - 再通过 `self.image_pseudo_video_generator.generate(...)`
  - 把单张图扩成 pseudo-video
  - 然后生成：
    - `"video"`
    - `"lq_video"`
    - `"sample_seed"`
    - `"sample_id"`

注意：

- 这里 image branch 不是直接输出一张图
- 是输出一条已经展开好的 pseudo-video 分支

### 3.3 collate 形式

`v5.3` 的 collate 不是旧 streaming 的单一 `video/lq_video`，而是 paired tensor collate：

- `paired_tensor_collate_fn(batch)`

输出字段：

- `video`
- `lq_video`
- `image_video`
- `image_lq_video`
- `sample_id`
- `image_sample_id`
- `source_type`
- `image_source_type`

这意味着训练前向拿到的是两个 branch 明确分开的 batch。

## 4. v5.3 训练前向是怎么走的

### 4.1 主训练入口

训练文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

parser 中与 `v5` 强相关的参数：

- `--dataset_mode`
  - 可选：`unified / streaming / parquet_v2 / tar_v3 / tar_v5 / tar_v53`
- `--lq_proj_checkpoint`
- `--lq_proj_layer_num`
- `--lq_proj_scale`
- `--zero_init_lq_proj_in`
- `--yubari_video_tar_url`
- `--takano_video_tar_url`
- `--picked17k_image_tar_url`
- `--yubari_video_prob`
- `--takano_video_prob`
- `--image_video_joint_packed`
- `--degradation_config_path`
- `--global_seed`
- `--validation_num_samples`

当前正式 `v5.3` 路线实际用的是：

- `dataset_mode=tar_v53`
- `lq_proj_scale=1.0`

### 4.2 branch-aware 输入构造

`train_flashvsr_stage1_v5_3_lora.py` 中的关键函数：

- `_build_branch_inputs(...)`
- `get_pipeline_inputs(...)`

在 `get_pipeline_inputs(...)` 里，训练代码会把两条 branch 强制写成：

- `sequence_lengths = [video_frames + image_frames]`
- `segment_lengths = [[video_frames, image_frames]]`

这是当前 `v5.3` 最核心的结构表达。

也就是说：

- `v5.3` 不是让模型“自己猜哪里是图、哪里是视频”
- 而是在输入层就显式告诉后面的 packed 路径：
  - 这条 sample 的第一个 segment 是 video
  - 第二个 segment 是 image pseudo-video

### 4.3 VAE 与 LQ projector 的 segment-aware 编码

当前训练文件里有两个关键 helper：

- `_encode_video_segments_with_vae(...)`
- `_encode_lq_segments_with_projection(...)`

它们的职责是：

- 不把两个 segment 当成一个未经标注的整体去处理
- 而是按 `segment_lengths` 分段编码

其中 `_encode_lq_segments_with_projection(...)` 做了一个重要特殊处理：

- 对每个 segment 都会调用 `_repeat_frames_for_lq_segment(segment, min_frames=5)`

原因已经写在代码注释里：

- `FlashVSRLQProjIn` 的第一段 streaming clip 主要是 warm up cache
- 单帧 image segment 必须至少重复到 `5` 帧
- 才能让第二个 clip 真正产生一个 latent-time token

这个点很关键：

- `v5.3` 里 image branch 的 pseudo-video 不是装饰
- 它是为适配 `FlashVSRLQProjIn` 的 causal/streaming 结构而存在的

### 4.4 packed token 与 joint attention

`train_flashvsr_stage1_v5_3_lora.py` 中与 packed 路径直接相关的函数：

- `_resolve_per_sample_token_lengths(...)`
- `_resolve_raw_segment_lengths(...)`
- `_resolve_latent_segment_lengths(...)`

这些函数会基于：

- `sequence_lengths`
- `segment_lengths`

去构造 packed attention 需要的 token 长度信息。

对应模型侧文件：

- `diffsynth/models/wan_video_dit_joint_v5.py`

也就是说：

- `v5.3` 的 joint 不是简单 concat 后直接扔给原 DiT
- 而是已经接入 segment-aware 的 packed token 路径

## 5. v5.3.1 和 v5.3 的关系

`v5.3.1` 不是新结构，它是 `v5.3` 的退化对照版。

这两条应该保持一致的部分：

- 训练文件：
  - 都走 `train_flashvsr_stage1_v5_3_lora.py`
- 数据集结构：
  - 都是 `FlashVSRTarStreamingDatasetV53`
- joint 表达：
  - 都是 `video + image pseudo-video`
- attention 与 segment 表达：
  - 都用同一套 packed 结构
- `lq_proj_scale`
  - 都是 `1.0`

它们真正的差异只有退化配置：

- `v5.3`
  - `params_aliyun_video_compression_v1.yaml`
- `v5.3.1`
  - `params_aliyun_video_compression_v1_half.yaml`

### 5.1 half 退化到底改了什么

`params_aliyun_video_compression_v1_half.yaml` 的关键开关是：

- `disable_second_stage: true`

在 `wanvideo/data/flashvsr/degradation/aliyun_video_degradation.py` 中：

- 第一阶段始终执行：
  - `blur -> resize -> noise -> jpeg -> video compression`
- 第二阶段只有在：
  - `if not self.opt["disable_second_stage"]`
  时才执行

因此 `v5.3.1` 的含义是：

- 保留阿里云退化前半段
- 关掉第二段退化
- 不是改了一套全新的随机参数

## 6. v5.3.2 的定位与实现

`v5.3.2` 是独立特殊实验线，不是 `v5.3` 主线。

训练入口：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_2_lora.py`

这份文件本质上不是重写一套完整 trainer，而是做了一件很明确的事情：

- `base_train.FlashVSRTarStreamingDatasetV53 = FlashVSRTarStreamingDatasetV532YubariFrames`

也就是：

- 用 `FlashVSRTarStreamingDatasetV532YubariFrames`
  替换主训练入口里的 `FlashVSRTarStreamingDatasetV53`
- 其余训练主体仍沿用 `v5.3`

### 6.1 数据集类

文件：

- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v532_yubari_frames.py`

类：

- `FlashVSRTarStreamingDatasetV532YubariFrames`

类注释明确写了：

- `video branch: one normal Yubari video clip`
- `image branch: one single frame sampled from another Yubari video clip`
- `the sampled frame is then expanded into one fake-image-video branch`

也就是说，正确的 `v5.3.2` 是：

- 一个正常视频
- 加一张来自另一条 Yubari 视频的单帧图
- 这张图再扩成 pseudo-video

### 6.2 关键实现点

`FlashVSRTarStreamingDatasetV532YubariFrames` 的关键方法：

- `_extract_single_random_frame(...)`
- `_build_image_branch_from_video_bytes(...)`
- `_image_source_iterator(...)`

技术过程：

1. 从 Yubari 视频源拿到一段视频字节
2. `_extract_single_random_frame(...)`
   - 用 `cv2.VideoCapture`
   - 随机选一帧
   - 检查 `_meets_min_resolution`
3. `_build_image_branch_from_pil_frame(...)`
   - 把这张图扩成 pseudo-video
4. 最后仍返回 paired sample：
   - `video`
   - `lq_video`
   - `image_video`
   - `image_lq_video`

### 6.3 为什么这是特殊线

因为它改变的不是 joint 结构，而是 image branch 的来源：

- `v5.3 / v5.3.1`
  - image branch 来自外部图像源
- `v5.3.2`
  - image branch 来自视频中抽的一帧

所以：

- `v5.3.2` 不能反向污染 `v5.3 / v5.3.1`
- 它只是回答一个特殊问题：
  - 不用外部图片，只用视频帧当图像分支，训练行为会怎样

## 7. 图像源接口现在必须支持什么

这是最近 `v5` 调试中非常关键的一点。

当前要求是：图像源必须同时支持两种形式。

### 7.1 tar 根目录

即：

- `picked17k_image_tar_url` 指向一个 tar 根目录

这种情况下，代码会走：

- `self.image_tar_urls`
- `_iterate_tar_images(...)`

### 7.2 txt manifest 指向散图

即：

- `picked17k_image_tar_url` 实际上给的是一个 txt 文件
- 文件里一行一个远端 `jpg` 路径

这种情况下，代码会走：

- `self.image_manifest_urls`
- `_iterate_direct_images(...)`

最近补的关键修复就在这里：

- 以前 `streaming_dataset.py` 中 `_image_iterator(...)` 虽然知道 `image_manifest_urls` 存在
- 但没有真的把它消费进 image 分支
- 现在已经改成：
  - `_iterate_direct_images(...)` 会读取 `self.image_manifest_urls`
  - `_image_iterator(...)` 在有 `image_file_urls` 或 `image_manifest_urls` 时都启用 direct image 路径

这就是为什么最近 `v5.3 / v5.3.1` 能重新正常启动。

## 8. validation 在 v5 的正确原则

最近一个很重要的结论是：

- 训练样本可以是 joint image+video
- 但 fixed validation sample 只需要视频样本

技术锚点：

- `collect_fixed_validation_samples(dataset, num_samples)`
  - 文件：`train_flashvsr_stage1_v5_3_lora.py`

当前逻辑：

- 若 `dataset` 是 `FlashVSRStreamingDataset`
- 且存在视频源：
  - `dataset.parquet_records`
  - `dataset.video_tar_urls`
  - `dataset.video_file_urls`
  - `dataset.video_manifest_urls`
- 就优先使用：
  - `dataset._video_iterator(rng=...)`

只有在拿不到视频迭代器时，才回退到 `iter(dataset)`。

这条修复的意义是：

- validation 不再依赖 image branch 是否健康
- image 分支 bug 不应拖死 fixed validation sample 采集

## 9. `lq_proj_in` 初始化：为什么最近切到 flashinit

parser 相关参数在 `train_flashvsr_stage1_v5_3_lora.py` 中是：

- `--lq_proj_checkpoint`
- `--zero_init_lq_proj_in`
- `--lq_proj_scale`

当前结论：

- `lq_proj_scale` 继续固定为 `1.0`
- `v5` 的主要问题不在 scale 写错
- 也不在 `lq_proj_in` 结构与官方严重不一致
- 真正的问题是：
  - 如果 `zero_init_lq_proj_in=true`
  - `lq_proj_in` 的输出投影从零开始学
  - 那前期 LQ 支路太弱

最近 ratio probe 已经说明：

- `v5.3 step10` 很弱
- `v5.3 step300` 有在上涨
- 但仍明显低于旧 `v2` 和官方 FlashVSR

因此当前主线已经切成：

- `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
- `zero_init_lq_proj_in=false`

这就是文档里说的：

- `flashinit`

## 10. 当前明确废弃的方案

### 10.1 allow_small_image_upscale

原因：

- 它会把尺寸不够的图像强行上采样救回来
- 破坏当前“只用满足尺寸要求的原始样本”的数据假设
- 这条线已经从主逻辑中移除

### 10.2 在共享主线里塞临时 validation 特判去绕 bug

原因：

- 当时真正的问题是 image manifest 支持不完整
- 不应该用 validation 特判去掩盖
- 主线应保持干净

### 10.3 错误版 v5.3.2：多视频多帧打包

错误定义是：

- 一个 sample 里让 image branch 来自多条视频、各抽一帧再打包

废弃原因：

- 不符合实验定义
- 用户要的是：
  - 一个视频
  - 一张图
- 不是多个视频帧拼接的 grouped image

## 11. 当前有效实验与其技术含义

截至 2026-04-23，当前整理后的有效实验是：

### 11.1 `v5.3.2`

- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_20260423_150800`
- 技术含义：
  - joint 结构沿用 `v5.3`
  - image branch 来自 Yubari 视频单帧
  - 退化用完整 Aliyun
  - `lq_proj_in` 用 flashinit

### 11.2 `v5.3.1`

- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_20260423_154400`
- 技术含义：
  - 正式主线结构
  - 外部图像源 + Yubari + Takano
  - 半 Aliyun 退化
  - `lq_proj_in` 用 flashinit

### 11.3 `v5.3`

- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_20260423_154500`
- 技术含义：
  - 正式主线结构
  - 外部图像源 + Yubari + Takano
  - 完整 Aliyun 退化
  - `lq_proj_in` 用 flashinit

## 12. 当前推荐理解顺序

以后查 `v5`，建议按下面顺序理解，不要再混：

1. `v5.3` 是正式主线
2. `v5.3.1` 是 `v5.3` 的退化强度对照
3. `v5.3.2` 是特殊实验，只改 image 来源
4. `v5.1 / v5.2` 是早期 grouped-image 探索线，不是当前推荐方向

如果要继续扩 `v5`，新增版本也应该先回答三个问题：

1. image branch 来源是什么
2. image branch 在 sample 层如何表达
3. loss 是对齐“视频整段”还是“视频首帧”

只有这三点写清楚，版本号才有意义。

## 13. 2026-04-24：`freeze projector` 阶段与新 4k 图像源

### 13.1 图像源切换

`v5.3 / v5.3.1` 当前图像源已经从旧的 txt manifest：

- `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`

切到新的 4k tar 根目录：

- `s3://lucid-vr/datasets/takano_image/image/takano-image-20231106-train/4k/`

这次切换的考虑是：

- 旧的 txt manifest + 散图读取已经被证明会明显拖慢训练；
- 新路径按 tar 流式读取，更符合 `tar_v53` 这条正式主线的设计；
- 实际统计该目录下共有 `51168` 个 tar，若按 `100 图/tar` 估算，总图像量约为 `5.1M`。

但随后很快确认：

- 虽然已经能切过去跑，
- 当前阶段该新图像源的整体表现“不太对劲”，
- 因此 `v5.3 / v5.3.1` 正式线又回退到了原始 manifest 图像源：
  - `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`

所以当前有效结论是：

- `4k tar image` 路径已经验证过“能接上”
- 但当前正式训练主线暂不使用它
- `v5.3 / v5.3.1` 继续以旧 manifest 图像源为准

更新：

- 这一回退并没有成为最终决定。
- 在后续比较后，当前又统一切回：
  - `s3://lucid-vr/datasets/takano_image/image/takano-image-20231106-train/4k/`

因此截至当前，正式结论应更新为：

- `v5.3 / v5.3.1` 的正式图像源以 tar 为准
- manifest 版本只作为中间对照，不再继续作为主线

### 13.2 `freeze_lq_proj_in`

为验证 `flashinit` 前期很像 LQ、但后续视觉表现可能被训练带偏这一现象，`v5.3` 主训练入口新增：

- `--freeze_lq_proj_in`

对应实现位于：

- `train_flashvsr_stage1_v5_3_lora.py`

具体行为：

- 在 `FlashVSRStage1TrainingModule.__init__` 中，如果 `freeze_lq_proj_in=True`：
  - 对 `self.pipe.lq_proj_in.parameters()` 全部设置 `requires_grad=False`
  - 并执行 `self.pipe.lq_proj_in.eval()`
- 这样当前阶段只允许：
  - `DiT LoRA` 学习
- 而 `lq_proj_in` 保持：
  - `FlashVSR-v1.1` 的初始化值

这条线的目的不是替代正式训练，而是把问题拆开：

- 如果 projector 冻住后前期视觉更稳，说明过去 projector 也被一起训偏了；
- 如果 projector 冻住后仍然变坏，则更可能是 `LoRA` 在重新解释条件。

### 13.3 当前冻结版实验定义

新增并启动了三条 `freezeproj` 线：

- `v5.3.2 freezeproj`
  - `train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260423_224600`
- `v5.3.1 freezeproj`
  - 首次 4k 版本：
    - `train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_014800`
  - 中间 manifest 版本：
    - `train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_023200`
  - 当前有效 tar 版本：
    - `train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_freezeproj_20260424_025100`
- `v5.3 freezeproj`
  - 首次 4k 版本：
    - `train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_014900`
  - 中间 manifest 版本：
    - `train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_023300`
  - 当前有效 tar 版本：
    - `train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_freezeproj_20260424_025200`

这三条线共同特征：

- `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
- `zero_init_lq_proj_in=false`
- `freeze_lq_proj_in=true`
- `batch_size=12`

其中：

- `v5.3 / v5.3.1`
  - image branch 来自外部图像源
  - 当前已经改成新的 4k tar 根目录
- `v5.3.2`
  - image branch 仍来自另一条 Yubari 视频里随机解码的单帧
  - 不依赖外部图像路径

### 13.4 phase2 继续训练的推荐方式

当前不推荐：

- 直接用 `resume_training_state_dir` 从冻结版切到非冻结版

原因是：

- 冻结版与非冻结版的 optimizer 参数组不一致；
- 直接恢复完整 training state，容易因为 trainable parameter set 变化带来不必要风险。

当前推荐方式是：

- 把冻结版作为 phase1；
- 选一个满意的 `stage1 checkpoint`；
- 用新的 phase2 实验 warm-start；
- phase2 中重新打开：
  - `projector + LoRA`

因此新增了三份模板：

- `stage1_release_16gpu_v5_3_phase2_unfreeze_template.yaml`
- `stage1_release_16gpu_v5_3_1_phase2_unfreeze_template.yaml`
- `stage1_release_16gpu_v5_3_2_phase2_unfreeze_template.yaml`

以及对应模板脚本：

- `FlashVSR-Stage1-Release-16GPU-v5-3-Phase2-Unfreeze-Template.sh`

模板要求外部传入：

- `STAGE1_CKPT=<phase1 满意的 checkpoint>`

这意味着：

- phase1 的 projector 即使冻结，checkpoint 中仍然会保存 `lq_proj_in`；
- 但 phase2 的真正重点是继承：
  - 已经学到一定程度的 `LoRA`
- 同时继续以 `FlashVSR` 的 projector 先验作为稳定起点。

### 13.5 `freezeproj` checkpoint 导出缺陷与修正

在 `v5.3.x` 的第一版 phase1/phase2 设计里，存在一个实际导出缺陷：

- `ModelLogger.save_model(...)`
  - 先调用 `accelerator.get_state_dict(model)`
  - 再调用 `accelerator.unwrap_model(model).export_trainable_state_dict(...)`
- 而 `DiffusionTrainingModule.export_trainable_state_dict(...)` 默认只保留：
  - `requires_grad=True` 的参数

这会导致：

- 当 `freeze_lq_proj_in=true` 时
- `pipe.lq_proj_in.*` 因为不再可训练
- 会在导出阶段被整个滤掉

于是 `freezeproj` 阶段得到的 `step-*.safetensors` 实际只包含：

- `LoRA`

不包含：

- `projector`

这已经被直接验证过：

- `v5.3.2 freezeproj step-2000`
  - key 总数：`480`
  - `num_lora = 480`
  - `num_lq = 0`

这进一步导致早期的 `v5.3.2 phase2` 虽然写了：

- `resume_stage1_checkpoint=...step-2000.safetensors`

但实际只恢复了：

- `LoRA`

没有恢复：

- `projector`

因此 phase2 的真实起点变成：

- `LoRA` 来自 phase1
- `projector` 重新初始化

这解释了为什么：

- `v5.3.2 phase2` 的前 10 step 视觉结果明显不如冻结版 `step-2000`

修正方式已经落地：

1. 在 `FlashVSRStage1TrainingModule` 中重写：
   - `export_trainable_state_dict(...)`
2. 在默认“导出所有 trainable 参数”之外，额外强制保留：
   - `pipe.lq_proj_in.*`

这样之后新的 `freezeproj` checkpoint 会同时包含：

- `LoRA`
- `projector`

### 13.6 `v5.3.2 phase2` 初始化口径修正

为了避免在旧 `freezeproj` checkpoint 尚未重新导出之前出现“只恢复 LoRA、不恢复 projector”的问题，`v5.3.2 phase2` 现在改成：

- `resume_stage1_checkpoint`
  - 负责恢复 phase1 学到的 `LoRA`
- `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
  - 负责显式恢复 FlashVSR 官方 projector 初始化

这要求 parser 允许以下组合：

- `resume_stage1_checkpoint + lq_proj_checkpoint`

因此当前 `v5.3.x` parser 约束也一起调整为：

- 仍然禁止：
  - `resume_stage1_checkpoint + lora_checkpoint`
- 允许：
  - `resume_stage1_checkpoint + lq_proj_checkpoint`

这样 phase2 的语义才和实验意图一致：

- `LoRA` 继承 phase1
- `projector` 继承 FlashVSR v1.1

### 13.7 `v5.3.x` validation 采样路径修正

`v5.3 / v5.3.1 / v5.3.2` 的训练样本都是 paired sample：

- video branch
- image pseudo-video branch

训练时这样是正确的，但 validation 固定样本不应该为了抽 3 个验证视频而触发 image branch。旧逻辑在 `collect_fixed_validation_samples(...)` 里如果识别不到普通 video source，就回退到 `iter(dataset)`，这会走 paired sample，导致 validation 采样被 image 读取和尺寸过滤拖慢甚至卡住。

当前修法：

- `FlashVSRTarStreamingDatasetV53.validation_video_iterator(...)`
  - 只在 `yubari/takano` video source 内采样。
  - 保留 `yubari_video_prob / takano_video_prob` 的视频源比例。
- `FlashVSRTarStreamingDatasetV532YubariFrames.validation_video_iterator(...)`
  - 直接复用 Yubari video iterator。
  - 不触发“从另一条 Yubari 视频抽单帧做 image branch”的逻辑。
- `collect_fixed_validation_samples(...)`
  - 优先检测并调用 `dataset.validation_video_iterator(...)`。
  - 只有普通 streaming dataset 才继续使用原来的 `_video_iterator(...)` 或最终回退。

这次修复后，validation 仍然保持开启：

- `validation_num_samples: 3`
- 三条 warm-start 实验都成功打印 `Prepared 3 fixed validation samples.`

### 13.8 `v5.3.2` warm-start 入口修正

`v5.3.2` 是特殊实验线：

- video branch 来自 Yubari clip
- image branch 来自另一条 Yubari 视频中随机抽取的单帧
- 单帧再扩成 5 帧 pseudo-video

因此 `v5.3.2` 必须通过：

- `train_flashvsr_stage1_v5_3_2_lora.py`

该 wrapper 会把通用 `FlashVSRTarStreamingDatasetV53` 替换为：

- `FlashVSRTarStreamingDatasetV532YubariFrames`

本轮发现 `Step1300-bs10-worker2` warm-start 脚本误用：

- `train_flashvsr_stage1_v5_3_lora.py`

这会绕开 `v5.3.2` 的特殊数据逻辑。已经修正为正确 wrapper，并在 snapshot 中额外保存：

- `tar_streaming_dataset_v532_yubari_frames.py`

### 13.9 当前 `bs10 + worker2` warm-start 结果

为了保留 worker=2 的吞吐优化，同时避免 `bs12 + worker2` 的显存余量不足，本轮把三条 phase2 warm-start 统一改为：

- `batch_size: 10`
- `dataset_num_workers: 2`
- `validation_num_samples: 3`
- 从历史 safetensors 热启动，不声称 DeepSpeed optimizer/state 原地 resume

当前已启动并出 loss：

- `v5.3.2`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_phase2_warmrestart_step1300_bs10_worker2_seed20260428_20260428_052600`
  - 从 `step-1300.safetensors` 热启动
- `v5.3.1`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_phase2_warmrestart_step700_bs10_worker2_seed20260428_20260428_052700`
  - 从 `step-700.safetensors` 热启动
- `v5.3`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_warmrestart_step700_bs10_worker2_seed20260428_20260428_052800`
  - 从 `step-700.safetensors` 热启动

## 14. 2026-04-29：`v5.3` paired sample 的 LQ projector segment 对齐修复

### 14.1 问题背景

`v5.3` 的设计是 author-style paired sample：

- 一个样本内有一条真实 `video branch`。
- 同一个样本内再拼一条 `image pseudo-video branch`。
- 两条 branch 在 VAE / LQ projector 阶段应该按各自 raw frame count 独立处理。
- 到 latent/token 层后再通过 packed segment 进入 DiT。

这条路线的目标是避免 `v5.1 / v5.2` 那种 grouped image sample 带来的 batch shape 和显存补丁问题。

### 14.2 检查脚本

为了确认 branch 是否真的隔开，新增：

- `wanvideo/model_training/flashvsr/scripts/probe_v53_branch_separation.py`

它固定取一个 `v5.3` paired sample，并保存：

- 输入视频和假图像视频：
  - `inputs/video_gt.mp4`
  - `inputs/video_lq.mp4`
  - `inputs/image_pseudo_gt.mp4`
  - `inputs/image_pseudo_lq.mp4`
- VAE 重建：
  - `vae_recon/video_branch_recon.mp4`
  - `vae_recon/image_branch_recon.mp4`
- token norm heatmap：
  - `patch_before_lq`
  - `lq_proj_layer0_raw_all`
  - `lq_proj_layer0`
  - `patch_after_lq`
  - `dit_block_00 / 01 / 05 / 10 / 20 / last`
  - `dit_head`
- 结构摘要：
  - `branch_separation_summary.json`

当前用于定位问题的输出目录：

- 修复前：`/mnt/task_wrapper/user_output/artifacts/debug/v53_branch_probe_step1000_minimal_20260429_r3`
- 修复后：`/mnt/task_wrapper/user_output/artifacts/debug/v53_branch_probe_step1000_minimal_20260429_r4_segmentalign`

### 14.3 根因：旧 LQ projector padding 是全局 padding，不是 segment padding

`v5.3` 中 video branch 和 image branch 的 DiT latent 长度不同：

- `17` 帧视频经过 WAN VAE 后是 `5` 个 latent-time。
- `5` 帧 image pseudo-video 经过 WAN VAE 后是 `2` 个 latent-time。
- 因此一个 paired sample 的 DiT segment 是：
  - video segment：`5`
  - image segment：`2`

但是 `FlashVSRLQProjIn` 的 temporal 输出比 DiT latent 少一帧：

- `17` 帧视频的 LQ projector 输出 `4` 个 latent-time。
- `5` 帧 image pseudo-video 的 LQ projector 输出 `1` 个 latent-time。

旧代码只看总 token 数，然后在整个 LQ token 序列最前面做 global front padding。以 `17 + 5` 为例：

- DiT 期望布局：
  - video：`5`
  - image：`2`
- LQ projector 原始布局：
  - video：`4`
  - image：`1`
- 旧 global padding 后的布局等价于：
  - `[pad, pad, V0, V1, V2, V3, I0]`

这样按 DiT segment 切分时：

- video segment 拿到 `[pad, pad, V0, V1, V2]`
- image segment 拿到 `[V3, I0]`

也就是说，最后一个 video LQ latent 被错误送进了 image branch。这个问题会让图像支路和视频支路在 LQ 注入阶段发生串扰，尤其可能影响大运动场景。

### 14.4 修复：`_align_lq_latents_to_dit_tokens(...)` 改成 per-segment alignment

新增函数：

- `_lq_latent_length_from_raw_frames(raw_frames)`
- `_align_lq_latents_to_dit_tokens(lq_latents, expected_tokens, tokens_per_frame, raw_segment_lengths=None)`

当 `raw_segment_lengths` 存在时，不再做全局 padding，而是逐 segment 对齐：

- video raw frames = `17`
  - DiT latent frames = `5`
  - LQ latent frames = `4`
  - 该 segment 内补 `1` 个 front latent frame
- image raw frames = `5`
  - DiT latent frames = `2`
  - LQ latent frames = `1`
  - 该 segment 内补 `1` 个 front latent frame

修复后的布局是：

- video：`[pad, V0, V1, V2, V3]`
- image：`[pad, I0]`

这样 LQ token 不再跨 video/image branch 边界移动。

### 14.5 修改位置

主要修改文件：

- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

涉及路径：

- `flashvsr_stage1_model_fn(...)`
- `flashvsr_stage1_fixed_prompt_model_fn(...)`
- `_align_lq_latents_to_dit_tokens(...)`

原先逻辑：

- `lq_latent.shape[1] < x.shape[1]` 时全局 front pad。

现在逻辑：

- 优先根据 `_resolve_raw_segment_lengths(sequence_lengths, segment_lengths)` 解析每个 sample 的 raw segment。
- 如果是 `v5.3` paired sample，则按 segment 单独 pad。
- 如果不是 packed segment，才退回旧的全局 padding。

### 14.6 对 nonstreaming LR projector 的影响

`v5.3.4` 的非流式 LR projector 不是另一个不同问题。它和 streaming projector 一样，都会出现 “LQ temporal 输出比 DiT latent 少一帧”：

- `17 + 5`
  - DiT latent：`[5, 2]`
  - nonstream LQ latent：`[4, 1]`
- `89 + 5`
  - DiT latent：`[23, 2]`
  - nonstream LQ latent：`[22, 1]`

因此非流式版本同样必须使用 per-segment alignment。

检查记录：

- `/mnt/task_wrapper/user_output/artifacts/debug/v534_nonstream_alignment_20260429/nonstream_lq_alignment_summary.json`

### 14.7 退化配置传递修复

同时发现 `FlashVSRTarStreamingDatasetV53.__init__` 里之前没有把：

- `degradation_config_path`

继续传给 base dataset。结果是 YAML 写了 Aliyun 退化配置，但 `tar_v53` 构造时可能没有真正使用该配置。

已修复：

- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

后续新启动的 `tar_v53` 实验会按配置读取退化文件。

### 14.8 修复后验证实验

两卡母机上用 `v5.3 step-1000` checkpoint 启动修复版 warm-start：

- 配置：
  - `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_phase2_warmrestart_step1000_bs10_worker2_seed20260429_segmentalign.yaml`
- 脚本：
  - `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Phase2-WarmRestart-Step1000-bs10-Worker2-Seed20260429-SegmentAlign.sh`
- 目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_phase2_warmrestart_step1000_bs10_worker2_seed20260429_segmentalign_20260429_0335`

已看到 loss：

- `step=1 loss=0.082779`
- `step=2 loss=0.067493`
- `step=3 loss=0.087492`
- `step=6 loss=0.072633`

### 14.9 当前风险

六节点上已经跑起来的 `v5.3.4 nonstream LR projector` 实验是在这次修复前启动的：

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_48gpu_v5_3_4_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_worker1_20260428_1834`

它当前能正常出 loss，但如果后续要把它作为正式结论，需要考虑重启到包含 per-segment LQ alignment 的版本。
