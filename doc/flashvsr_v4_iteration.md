# FlashVSR v4 Iteration Notes

这份文档只讲 `v4`。

目标：
- 说明 `v4` 是怎么一步一步长出来的
- 记录 `v4` 试过哪些方向
- 说明 `v4` 为什么没有直接继续做成最终主线
- 解释它相对 `v5` 的主要短板

## 1. v4 想解决什么

`v4` 是在 `v2 / v3` 之后，第一次认真把“多源数据 + image/video joint training”往正式训练线里推的一版。

在 `v4` 之前：
- `v2` 更偏纯视频 LoRA 训练
- `v3` 更偏 full finetune 和训练系统稳定性

`v4` 的目标开始变成：
- 不只训单一视频源
- 把 Takano / Yubari / image 这几类数据放进同一条训练线
- 验证 image/video joint training 会不会带来收益
- 让数据和 attention 的新逻辑先跑起来

所以 `v4` 本质上是“从纯视频训练过渡到联合训练”的桥梁版本。

## 2. v4 的核心组成

核心训练文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_fullb.py`

对应的数据和 joint 结构主要依赖：
- `wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py`
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v3.py`
- `diffsynth/models/wan_video_dit_joint_v1.py`

`v4` 做过的关键事情：
- 把新 Takano / Yubari / image 数据接进同一条训练线
- 增加显式数据比例控制
- 尝试 image/video packed attention
- 增加 aliyun 退化版本做对比
- 做多轮 probe 实验定位 loss 异常来源

## 3. v4 的迭代过程

### 3.1 第一阶段：先把多源数据接进来

这阶段的重点不是 author-style image/video joint，而是先回答：
- 新数据集能不能读
- 三源能不能按比例混训
- packed 和 non-packed 在工程上能不能跑

对应的实验大致就是这些 probe：
- `ProbeA`: Takano only, no packed
- `ProbeB`: Takano + Yubari, no packed
- `ProbeC`: Takano + image, no packed
- `ProbeD`: Takano + Yubari + image, no packed
- `ProbeE`: Takano only, packed
- `ProbeF`: Takano + image, packed
- `ProbeG`: 多源 packed，并调整比例

对应 config/sh 文件在：
- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v4_probe*.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v4-Probe*.sh`

这一阶段其实是在做“消融式工程排查”。

### 3.2 第二阶段：明确问题不是单一视频训练，而是 image 加进来之后 loss 抬高

`v4` 很快暴露出一个核心问题：
- 纯视频时，loss 初始通常在正常范围
- 一旦 image 混进去，loss 经常直接跳到 `5` 左右，然后才慢慢学

这个问题非常关键，因为它说明：
- 不是简单的 batch 太大
- 也不只是退化问题
- 而是 image/video joint 的组织方式本身可能不对

`v4` 围绕这个问题做了几轮排查：
- 只用 Takano
- Takano + Yubari
- Takano + image
- 三源一起
- packed / non-packed 对比
- image/video 比例调整

结论大致是：
- 只加视频源，问题相对没那么明显
- image 一进来，loss 抬升最明显
- 所以 image 组织方式比普通数据比例更值得怀疑

### 3.3 第三阶段：尝试 image/video packed attention

`v4` 里最早版本的 joint attention 主要是基于 `wan_video_dit_joint_v1.py`。

这版的思路是：
- 在 DiT 里给 image 和 video 加分段 mask
- 让它们互相看不见

它的意义是：
- 验证“用 block-diagonal mask 隔开 image/video”这个方向是不是成立

但这版有明显限制：
- 一开始是 dense mask 路线
- 有 mask 时会退到 `scaled_dot_product_attention(...)`
- 没法继续完整保留 flash-attn 的效率优势
- image 和 video 的 token 组织仍不够自然

所以 `v4` 虽然证明了“mask 思路可行”，但没有得到一条足够优雅的正式主线。

### 3.4 第四阶段：数据和退化都继续增强，但核心 joint 语义仍没彻底解决

在 `v4` 期间又继续做了这些事：
- 把多源数据入口做得更完整
- 把 aliyun 退化也接进来
- 增加断点续训
- 增加更明确的启动脚本和 history

这时候 `v4` 已经不是“跑不起来”，而是：
- 训练能跑
- 但 image/video joint 的设计语义不够干净
- loss 行为也不够让人放心

换句话说，`v4` 卡住的不是“系统工程完全不通”，而是“方案本身不够好”。

## 4. v4 为什么没有继续直接当最终版

`v4` 主要有四类问题。

### 4.1 image/video 的训练样本定义不够清楚

这是 `v4` 最核心的问题。

在 `v4` 里，image 虽然已经接进来了，但“一个图像样本到底应该怎样和一个视频样本对齐”没有完全定清楚：
- 是等价于视频首帧？
- 还是等价于整段视频？
- 应不应该按 packed segment 处理？
- 应不应该在 loss 上单独加权？

这些问题在 `v4` 阶段都还处于排查和试探状态。

### 4.2 attention 路线还不够优雅

`v4` 走过两条路：
- no packed
- packed / mask

但都存在问题：
- no packed：语义更粗
- packed + dense mask：效率不够好

尤其是当你明确要保留 flash-attn 时，`v4` 这套 attention 还不是最终答案。

### 4.3 数据结构还不够“按目标任务定制”

`v4` 更多是在“让现有 dataset 支撑联合训练”。

但后来越来越明确：
- 真正合理的 image/video joint 不是简单把 image 塞进来
- 而是要围绕 joint sample 的定义，反向设计数据组织

`v4` 做到了“能混”，但没做到“语义天然正确”。

### 4.4 loss 抬升问题没有被彻底消掉

`v4` 的一个直接现实问题是：
- 新数据、尤其是 image 混进来后，loss 初始经常很高
- 即使训练后面能往下走，这也说明进入模型的统计分布发生了明显变化

这个问题是 `v4` 最终没有成为主线的重要原因之一。

## 5. v4 相比 v5 的不足

这部分只讲相对关系。

### 5.1 v4 的 sample 语义不如 v5 清楚

`v5` 明确拆成：
- `v5.1`
- `v5.2`
- `v5.3`

每条线都明确在回答一个问题：
- image 和视频首帧公平，还是和整段视频公平？
- grouped image 好，还是 author-style paired sample 好？

而 `v4` 更像“边做边试的总实验场”。

### 5.2 v4 的 packed 路线不如 v5 干净

`v5.3` 已经把目标收敛到：
- video branch
- image branch
- branch-aware
- latent/token 层 packed

而 `v4` 里 packed 更多是一种技术尝试，不是已经定义好的 sample 语义。

### 5.3 v4 的数据组织没有围绕 image/video 等权问题展开

`v5` 的核心设计就是：
- image 到底按什么权重参与 loss
- image 该怎么和视频对齐

`v4` 虽然暴露了这个问题，但没有把它结构化地变成多条明确实验线。

## 6. v4 的价值

虽然 `v4` 最后没有成为最终主线，但它不是“失败代码”。

`v4` 的价值很大：
- 第一次把多源数据、joint 训练、packed/mask、aliyun 退化这些方向都串起来了
- 明确暴露出“image 一加进来 loss 抬升”的核心问题
- 逼着后面的 `v5` 不再只做工程拼接，而是回到 sample 定义本身

可以说：
- `v4` 的主要贡献不是给出最终答案
- 而是把错误方向和关键瓶颈明确暴露出来

## 7. 一句话总结

`v4` 是“联合训练工程化过渡版”：
- 它把多源数据、packed/mask、aliyun 退化这些能力先拉起来了
- 但 image/video sample 语义和 attention 组织还不够优雅
- loss 抬升问题也没有根治

所以 `v4` 最终促成了 `v5`：
- 从“把 image 塞进训练”转向“重新定义 image/video joint sample 本身”。
