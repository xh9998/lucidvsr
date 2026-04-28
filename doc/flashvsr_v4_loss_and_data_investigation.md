# FlashVSR V4 Loss 异常与数据速度排查

## 1. 背景

当前观察到的现象：

- `v2` 训练早期 loss 通常在 `0.1` 左右起步
- `v4` 训练早期 loss 会先冲到 `5` 左右，再慢慢开始学细节
- 这个现象在 `alpha=1` 和 `alpha=5` 都存在

对应实验：

- `v4 alpha=5`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v4_lora_17f_mix0404020_bs24_lr1e5_alpha5_nostartval_20260420_234100`
- `v4 alpha=1`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v4_lora_17f_mix0404020_bs24_lr1e5_nostartval_20260420_143500`
- `v2 alpha=5`
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v2_17f_takano_bs24_lr1e5_alpha5_resume_step1000_seed20260416_20260415_024800`

## 2. 结论先说

`v4` 和 `v2` 不是“只换了数据读取方式”。

目前已经确认的关键差异：

- `v2` 用的是 `streaming`
- `v4` 用的是 `parquet_v2`
- `v2` 只训练 Takano 视频
- `v4` 是三源混训
  - Takano `0.4`
  - image `0.4`
  - Yubari `0.2`
- `v4` 开了 `image_as_single_frame: true`
- `v4` 开了 `image_video_joint_packed: true`
- `v4` 的全局 seed 也不同

所以现在不能把问题归因成“parquet 读取导致 loss 变坏”。

更合理的怀疑顺序是：

1. `image 40%` 单帧样本把 early-stage loss 分布拉高了
2. `image_video_joint_packed` 改变了 attention 路径和优化问题
3. 新三源混合后，退化分布和原来纯 Takano 分布差异很大
4. `parquet_v2` 本身存在解码或样本构造差异

其中第 4 条目前不是第一怀疑对象。

## 3. v2 和 v4 的关键配置差异

### v2

- `dataset_mode: streaming`
- 只用 Takano 视频
- `image_dataset_prob: 0.0`
- `enable_degradation: true`
- `lq_proj_scale: 5.0`
- `num_frames: 17`
- `batch_size: 24`
- `learning_rate: 1e-5`

### v4

- `dataset_mode: parquet_v2`
- Takano + image + Yubari 三源混训
- `takano_dataset_prob: 0.40`
- `image_dataset_prob: 0.40`
- `yubari_dataset_prob: 0.20`
- `image_as_single_frame: true`
- `image_video_joint_packed: true`
- `enable_degradation: true`
- `num_frames: 17`
- `batch_size: 24`
- `learning_rate: 1e-5`

## 4. 建议的最小消融顺序

目标不是一次性解释全部，而是最快锁定是“哪一类改动”引起的。

### 实验 A：只保留 Takano，保留 v4 代码

配置目标：

- `dataset_mode: parquet_v2`
- `takano_dataset_prob: 1.0`
- `image_dataset_prob: 0.0`
- `yubari_dataset_prob: 0.0`
- `image_video_joint_packed: false`

意义：

- 如果这条 loss 一开始就恢复接近 `v2`
- 说明 `parquet_v2` 本身不是主因
- 问题更可能来自 `image` 或 `packed`

### 实验 B：Takano + Yubari，不加 image

配置目标：

- `takano_dataset_prob: 0.8`
- `image_dataset_prob: 0.0`
- `yubari_dataset_prob: 0.2`
- `image_video_joint_packed: false`

意义：

- 检查是不是 image 单帧源造成 early loss 抬高

### 实验 C：Takano + image，但关 packed

配置目标：

- `takano_dataset_prob: 0.6`
- `image_dataset_prob: 0.4`
- `yubari_dataset_prob: 0.0`
- `image_video_joint_packed: false`

意义：

- 检查 image 本身是否有问题
- 同时排除 packed attention 路径干扰

### 实验 D：三源都开，但关 packed

配置目标：

- `takano_dataset_prob: 0.4`
- `image_dataset_prob: 0.4`
- `yubari_dataset_prob: 0.2`
- `image_video_joint_packed: false`

意义：

- 如果 D 正常、当前 v4 不正常
- 问题就基本锁在 `image_video_joint_packed`

## 5. 推荐先做哪一个

优先做实验 A。

原因：

- 信息增益最大
- 改动最少
- 最快回答“是不是 parquet_v2 本身有毒”

如果 A 正常，再按 `A -> B -> C -> D` 顺序做。

## 6. 需要额外打印的统计

如果只看 loss，定位速度还是慢。建议给前 `20~50 step` 打一批统计。

建议打印：

- 当前 batch 的 source 构成
  - Takano / image / Yubari 各多少
- `video` 和 `lq_video` 的
  - min / max / mean / std
- VAE latent 的
  - min / max / mean / std
- `sequence_lengths`
- `segment_lengths`
- packed 前后 token 数

这些统计能直接回答几个问题：

- image 样本比例是否比预期更高
- 某一源的像素范围是否异常
- 单帧样本是否把 token 结构拉成了非常不均衡的 batch
- packed 路径是否把有效 token 分布改得很极端

## 7. 为什么 v4 数据读取明显更慢

这里要把“结构性慢”和“实现上不够轻”分开。

### 7.1 结构性慢的部分

`v4` 比 `v2` 天生更重，原因有这些：

- 三个数据源，不再是单一 Takano 视频
- image 是 parquet 索引后再找真实 jpg
- Yubari 是 parquet 索引 + tar 内 byte-range 取 mp4
- 每个样本不只是读路径，还要解码媒体
- 训练时还开了在线退化

也就是说，`v4` 的数据链路不是简单的：

- 读一个路径
- 打开一个 mp4

而是更接近：

- 先决定抽哪个 source
- 找对应 parquet shard
- 读 shard 里的记录
- 解析真实媒体路径或 tar 内 offset
- 再读取 mp4 / jpg
- 再 crop / resize / degrade

### 7.2 当前实现上偏慢的点

现在代码里真正会拖慢启动或吞吐的点主要有这些：

#### 1. source 初始化阶段会做目录发现

- Takano 会 `_discover_parquet_urls(...)`
- image 会 `_discover_parquet_urls(...)`
- Yubari 会构建 `part-xxxxxx.parquet` / `part-xxxxxx.tar` 对应关系

这一步不是读完整个 parquet 内容，但要先把 shard 列表发现出来。

如果目录很大、远端 list 比较慢，这一步就会拖启动。

#### 2. parquet 当前是“整文件读入 DataFrame / Arrow”再取行

虽然 parquet 不大，但现在不是完全流式到单行。

当前实现里：

- `_read_parquet_frame(...)` 会把 parquet 读进 pandas
- 或 `_open_parquet_for_arrow(...)` 读成 `ParquetFile`

这本身不一定非常慢，但如果频繁读很多 shard，还是会有累计开销。

#### 3. 媒体读取不是纯顺序扫描

Takano 新版不是老的单 tar 顺扫模式，而是：

- parquet 找真实 clip mp4 path
- 再打开对应 mp4

这意味着对象访问更分散，随机打开远端 mp4 的代价比顺扫单 tar 大。

#### 4. image/video 混训会让 batch 的媒体类型不统一

视频和图像混在一起时：

- 解码路径不同
- 样本长度结构不同
- batch 的计算形态更不稳定

这对吞吐和 early-stage loss 都可能有影响。

## 8. 如果不要随机性，会不会像原来一样快

会更快，但不一定完全回到 `v2` 那么快。

如果你放弃较强随机性，至少可以省掉这些开销：

- 不需要反复随机挑不同 parquet shard
- 不需要频繁切 source
- 可以对同一 shard 连续消费
- 远端对象局部性会更好

最直接的提速方式是：

- 每个 worker 先固定拿一个 parquet shard
- 把这个 shard 里的记录顺序或轻度打乱后消费一段时间
- 用完再切下一个 shard

## 9. 已完成排查结果

### A. Takano only + no packed

- 16 卡可正常训练
- 首轮 loss 正常
  - step1 `0.095169`
  - step2 `0.116454`

结论：

- `parquet_v2 + Takano` 本身不是根因

### B. Takano + Yubari + no image + no packed

- 16 卡可正常训练
- 首轮 loss 正常
  - step1 `0.058439`

结论：

- `Yubari` 不是把 loss 拉到 5 的主因

### C. Takano + image + no packed

- 训练直接失败
- 报错：
  - `stack expects each tensor to be equal size, but got [1,3,768,1280] and [17,3,768,1280]`

结论：

- image 单帧样本不可能走普通 tensor collate
- image 一旦混进 batch，结构上就需要 packed 或等价分组逻辑

### E. Takano only + packed

- 16 卡可正常训练
- 首轮 loss 仍正常
  - step1 `0.133371`
  - step2 `0.101324`

结论：

- `packed` 本身不是主因

### F. Takano + image + packed

- 16 卡能跑
- 但首轮 loss 明显异常偏高
  - step1 `1.977776`
  - step2 `2.708854`
  - step3 `2.141231`
  - step4 `3.709394`
  - step5 `1.188573`
  - step6 `2.757446`
  - step7 `2.580026`
  - step8 `2.165729`

结论：

- 问题已经收敛到：
  - `single-frame image`
  - 和
  - `packed mixed image/video batch`
  - 的组合

### G. 降低 image 比例到 0.1

- 配比：
  - Takano `0.7`
  - image `0.1`
  - Yubari `0.2`
- 16 卡可跑
- 首轮 loss 明显低于 F，但仍高于纯视频方案
  - step1 `0.746272`
  - step2 `0.677399`
  - step3 `0.563360`

结论：

- 降低 image 比例有效
- 但只是缓解，不是根治
- 所以下一步必须验证：
  - 是否把 image/video 改成分 batch 后，就能继续恢复到纯视频 loss 水平

## 10. 当前判断

当前最强结论不是“parquet 有毒”，而是：

- `image 单帧样本 + packed 混 batch`
  极可能是把 early loss 拉高的主要原因

当时进一步验证了一个假设：

- H：保持 `0.4 / 0.4 / 0.2`
- 但开启 image/video 分 batch
- 看是否在不放弃 image 数据的前提下，把首轮 loss 拉回正常区间

## 11. H / H2 分 batch 结果

### H. `0.4 / 0.4 / 0.2` + image/video 分 batch + bs16

- 代码路径已经接通
- 16 卡能进入 dataset 和 model 初始化
- 但首个 homogeneous batch 极慢，长时间未产出第一条 loss

结论：

- 分 batch 方案在大 batch 下至少存在明显吞吐代价
- 不能直接当成一个无代价修复

### H2. `0.4 / 0.4 / 0.2` + separate image/video batches + bs4

- 为了尽快拿结论，保持同样的数据比例和逻辑，只把 batch 降到 `4`
- 16 卡成功出 loss：
  - step1 `2.903294`
  - step2 `1.880248`

结论：

- image/video 分 batch 不是根治
- 它没有把 early loss 拉回 `0.1 ~ 0.7` 这个区间
- 所以问题不只是“image/video 同 batch 干扰”
- 更大的嫌疑仍然在：
  - image 单帧源本身
  - image 占比过高
  - image 与 video 的任务分布差异过大

## 12. 当前最新判断

目前证据链变成：

1. `parquet_v2` 不是主因
2. `Yubari` 不是主因
3. `packed` 本身不是主因
4. `image` 一旦进入训练，early loss 就明显抬高
5. 只靠降低 image 比例可以缓解
6. 只靠 image/video 分 batch 不足以解决

因此这条分 batch 逻辑已经不建议继续保留在主代码里。下一步优先级应是：

- I：继续下调 image 比例，例如 `0.02 ~ 0.05`
- J：改成 image 单独 step / 单独 dataloader，而不是只做 batch 内分流
- K：必要时先把 image 从主训练线移除，等视频版本稳定后再接回

这样会比“每次样本都全局随机 source + 随机 shard + 随机对象”轻很多。

但代价也明确：

- 全局随机性下降
- batch 间 source 多样性下降
- 短期内样本相关性更强

所以正确说法是：

- 不要随机性，速度会更好
- 但不会自动让三源混训退回到 `v2` 的轻量程度

## 9. 图像视频 mask 会不会影响速度

会，但要看实现方式。

当前主线 `v4` 不是走“通用 dense mask attention”。

当前实现是：

- `image_video_joint_packed: true`
- 走的是 `wan_video_dit_joint_v1.py`
- 内部优先走 `flash_attn_varlen_func`
- 不是退回 `scaled_dot_product_attention(attn_mask=...)` 那种重路径

所以当前这版 mask 的速度影响应该是：

- 有额外 packing / unpacking / segment bookkeeping 开销
- 但不会像 dense mask attention 那样暴慢

结论：

- 会有开销
- 但如果 flash-attn varlen 正常启用，这个不是“主要慢因”
- 真正更重的通常还是远端对象访问、媒体解码、三源切换和在线退化

## 10. 为什么“只流式读 parquet，再流式打开 mp4”理论上不该这么慢，但实际上会慢

你的直觉是对的，但前提是数据访问满足两个条件：

1. parquet shard 列表获取很快
2. 后续媒体对象访问有较好局部性

现在的实际问题主要在第二条。

因为当前样本不是：

- 固定读一个 shard
- 顺着读很多相邻对象

而更像：

- source 之间切换
- shard 之间切换
- mp4 / jpg / tar byte-range 混合访问

所以“理论上的流式”被真实对象存储延迟和随机访问模式吃掉了。

## 11. 当前最务实的方向

### 定位 loss 问题

先做：

- 实验 A：`Takano only + packed off`

这条最能快速回答问题是不是 `image` / `packed` 引起的。

### 提速方向

如果优先目标是先把训练稳定跑起来，而不是先做到最强随机性，建议：

- 降低 source 切换频率
- 降低 shard 切换频率
- 让 worker 在单个 parquet shard 内连续消费一段时间
- 保留 cache，但不要做重型预扫描和重型预下载

一句话说，现在最该优化的不是“parquet 本身”，而是：

- 样本调度策略
- 对象访问局部性
- source/shard 切换频率

## 12. 已完成的 16 卡消融结果

下面这些结果都来自母机3的真实 16 卡训练，不是推测。

### Probe A

配置：

- Takano only
- `image_video_joint_packed: false`
- `bs16`

结果：

- `step=1 loss=0.095169`
- `step=2 loss=0.116454`

结论：

- `parquet_v2` 本身不会把 loss 直接拉到 `5`
- `v4` 主训练代码本身也不是天然有毒

### Probe B

配置：

- Takano `0.8`
- Yubari `0.2`
- no image
- `image_video_joint_packed: false`
- `bs16`

结果：

- `step=1 loss=0.058439`

结论：

- Yubari 不是把 early loss 拉高的主因

### Probe C

配置：

- Takano `0.6`
- image `0.4`
- no Yubari
- `image_video_joint_packed: false`
- `bs16`

结果：

- 不是高 loss
- 而是直接训练失败

真实报错：

- `RuntimeError: stack expects each tensor to be equal size`
- image 样本是 `[1, 3, 768, 1280]`
- video 样本是 `[17, 3, 768, 1280]`

结论：

- `image_as_single_frame=true` 时，image 这条线天然依赖 packed
- 所以 image 进来以后，问题不再是单纯的数据源问题，而是 image 和 packed 的组合问题

### Probe E

配置：

- Takano only
- `image_video_joint_packed: true`
- `bs16`

结果：

- `step=1 loss=0.133371`
- `step=2 loss=0.101324`

结论：

- packed 自己不是主因
- 只用 Takano、开 packed，loss 仍然是正常的 `0.1` 级别

### Probe F

配置：

- Takano `0.6`
- image `0.4`
- no Yubari
- `image_video_joint_packed: true`
- `bs16`

结果：

- `step=1 loss=1.977776`
- `step=2 loss=2.708854`
- `step=3 loss=2.141231`
- `step=4 loss=3.709394`
- `step=5 loss=1.188573`
- `step=6 loss=2.757446`

结论：

- 这条已经稳定复现“loss 明显高于正常视频训练”的现象
- 结合 A/B/E，可以基本锁定：
  - 不是 parquet
  - 不是 Yubari
  - 不是 packed 单独的问题
  - 主要问题来自 `single-frame image + packed mixed batch`

## 13. 最终阶段性结论

到目前为止，最有把握的判断是：

- `v4` 里把 early loss 拉高的主因不是数据读取后端
- 也不是 Yubari
- `packed` 对纯视频不是问题
- 真正敏感的是 image 单帧源进入训练以后带来的分布变化

更准确地说，当前证据最支持的是：

- `image (f=1)` 本身就是主要扰动源
- `packed` 可能是放大器
- 但不是 packed 单独实现错误

已经验证过且现在不再建议继续保留的尝试：

- image/video 分 batch
  - 结论：更慢，而且不能把 loss 拉回正常区间

因此当前更合理的后续方向是：

- 继续下调 image 比例，例如 `0.02 ~ 0.05`
- 或把 image 改成单独 step / 单独 dataloader
- 如果主目标是先把视频主线训稳，就先把 image 从主训练线移除
