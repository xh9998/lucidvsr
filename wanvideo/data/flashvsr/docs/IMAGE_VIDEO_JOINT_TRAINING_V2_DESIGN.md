# FlashVSR Image/Video Joint Training v2 Design

## Goal

在不改动当前 `v2/v3` 主训练线的前提下，单独设计一条更贴近论文 Stage 1 的 image/video joint SR training 路径。

目标对齐论文 3.2 的三点：

1. 图像按单帧视频输入，`f=1`
2. 图像和视频共用统一的 3D attention formulation
3. 同 batch 内使用 block-diagonal segment mask，避免不同样本段之间互相注意

## Current State

当前已经对齐的部分：

1. 新数据集 `parquet_tar_dataset_v2.py` 已支持：
   - 新 Takano 视频 parquet
   - image parquet -> `TARGET_S3_PATH`
   - Yubari video tar + metadata tar
2. 图像样本当前已按 `f=1` 输出
3. 数据层已经拆成三层：
   - index layer
   - media access layer
   - dataset layer

当前还未对齐的关键缺口：

1. collate 仍假设所有样本时间长度一致
2. 主训练模型没有接收 `segment_ids / segment_mask`
3. Wan 前向没有使用 block-diagonal segment mask

## Proposed Architecture

### 1. Dataset layer

统一输出标准 sample record：

- `video`: `(T, C, H, W)`
- `lq_video`: `(T, C, H, W)`
- `sample_seed`
- `sample_id`
- `source_dataset`
- `sequence_length`

其中：

- video sample: `T = 17` 或 `89`
- image sample: `T = 1`

### 2. Collate layer

新增 joint collate：

1. 收集 batch 内各样本的 `sequence_length`
2. 将所有样本 pad 到本 batch 的 `max_T`
3. 生成：
   - `segment_ids`
   - `sequence_lengths`
   - `attention_mask` 或 block metadata

推荐输出：

- `video`
- `lq_video`
- `sample_seed`
- `sequence_lengths`
- `segment_ids`

当前实验文件：

- `wanvideo/data/flashvsr/datasets/joint_batching_v1.py`

### 3. Model input layer

在 `train_flashvsr_stage1_v2.py` 单独分出 image/video joint 分支：

1. 视频和图像都先走相同的 VAE encode
2. 图像因为 `T=1`，天然视作单帧视频 latent
3. patchify 后保留每段对应的 token range
4. 基于 token range 构建 block-diagonal segment mask

### 4. Attention layer

需要让 Wan/DiT attention 接收 segment mask。

最保守实现：

1. 只在 joint-training 分支里启用
2. 不改动现有默认训练
3. 对 attention scores 加 mask：
   - 同段 token 可见
   - 跨段 token 置为 `-inf`

当前实验文件：

- `diffsynth/models/wan_video_dit_joint_v1.py`

当前这版 joint model 的接口是：

- `WanJointModelV1.forward(..., segment_lengths=[[...], ...])`

每个 batch item 都可以携带自己的分段帧长，模型会在 patchify 后换算出 token 级 block-diagonal self-attention mask。

### 5. Validation strategy

joint training 不建议直接复用当前 validation callback。

建议：

1. 继续保留现有视频 validation
2. 额外增加 image-only validation：
   - 单帧 image 走 `f=1`
   - 输出首帧恢复结果

## Implementation Order

推荐顺序：

1. 数据层
   - 完成 image parquet / video parquet / Yubari 的统一 record 输出
2. collate
   - 支持 variable-length segments
3. model input
   - 传入 `sequence_lengths / segment_ids`
4. attention mask
   - joint-training only
5. validation
   - 单独 image/video joint validation

## Practical Recommendation

当前阶段不建议直接把 joint training 混进主 `v2/v3` 训练线。

更稳的做法是：

1. 保持现有 `v2/v3` 主训练线不动
2. 新开一条 `image_video_joint_v2` 分支
3. 先做 2 卡 smoke：
   - 视频 + 图像混合 batch
   - 检查 segment mask 数学正确性
4. 再上 8 卡 / 16 卡

## Notes

- 本地 `wanvideo/data/flashvsr/docs/image_video.txt` 给出了最小 block-diagonal segment mask 代码草图，这次实验版按那个思路拆分。
- 当前 `streaming_dataset.py` 里的 pseudo-video image 路径不应再视作论文对齐方案。
- 当前推荐的新入口是：
  - `parquet_tar_dataset_v2.py`
