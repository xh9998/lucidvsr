# FlashVSR V4 vs ref_big Gaps

这份文档只记录当前 `FlashVSR v4` 数据侧相对于 `ref_big` 仍未完全对齐的地方。

## 已经对齐到位的部分

- 数据入口已经拆成三层：
  - source discovery
  - parquet/index reading
  - media reading
- `Takano / image / Yubari` 三源都已经走统一的 bridge 访问远端对象。
- parquet 已经支持 lazy shard discovery，不再在启动阶段全量展开所有行。
- parquet 行读取已经改成 batch 流式迭代，不再默认整 shard `pd.read_parquet -> list(rows)`。
- `Yubari` 已经走：
  - `video root`
  - `part-*.parquet`
  - `part-*.tar`
  - `data_offset/data_size` byte-range read
- 三源采样比例已经显式可配：
  - `takano_dataset_prob`
  - `image_dataset_prob`
  - `yubari_dataset_prob`
- 图像数据已经支持按 `f=1` 单帧样本进入统一数据管线。
- conductor cache 已经由统一对象层接管，不再由 dataset 到处自行 `tmp cp`。

## 还没完全对齐的部分

### 1. 还不是 ref_big 那种“对象层绝对主导”

当前状态：

- dataset 逻辑已经明显变轻，但仍然是 dataset 在决定：
  - 什么时候读哪个 shard
  - 什么时候开哪个 parquet
  - 什么时候取哪个媒体对象

ref_big 更完整的做法：

- dataset 更多只描述“要什么”
- object/cache/limiter 层主导“怎么拿、怎么复用、怎么节流”

当前差距：

- 还没有把所有对象访问完全收口到 ref_big 那套统一 object manager 语义。

### 2. node-level 复用和节流还不够强

当前状态：

- 已经有统一 conductor cache
- 已经有 cache size / ttl / verbosity 控制

但还缺：

- 更强的 node-level hot object reuse 策略
- 更明确的跨 rank 预取/共享对象协调
- 更成熟的 limiter / qps budget / backpressure 策略接线

影响：

- 当前版本已经能用，但在大规模训练时，数据启动和中途吞吐稳定性还不如 ref_big 成熟。

### 3. Takano 仍然偏“轻量 lazy iterator”，不是 ref_big 完整对象调度版

当前状态：

- Takano 新版已经是 direct clip path parquet
- shard 发现和行读取都已经 lazy 化

但还缺：

- 更强的 clip-level locality 优化
- 更明确的热 parquet / 热媒体对象复用策略
- 更完整的预取和淘汰协同

影响：

- 当前 Takano 已经从“全量展开”进化到“按 shard 流式读”
- 但还没达到 ref_big 那种非常稳的长期高吞吐状态

### 4. shard 内随机性目前是轻量版

当前状态：

- source 级随机有
- shard 顺序随机有
- shard 内是流式顺序读为主

ref_big 可进一步做到：

- 更丰富的 shard 内随机混洗
- 更细的多级 sampling/scheduling

影响：

- 当前版本更轻、更稳、启动更快
- 但“全局强随机性”不如重型实现

### 5. image/video joint 训练还没完全并进 ref_big 风格主线

当前状态：

- image 已经按 `f=1` 进 dataset
- image/video joint packed attention 已经单独接入可用版本

但还缺：

- 把 joint packed/mask 逻辑彻底沉到主训练线默认方案里
- 让数据侧、batch 组织、attention 侧都围绕同一套 joint 设计统一

影响：

- 现在功能是“能用”
- 但还不是“整个系统天然就是为 image+video joint 设计的”

### 6. 训练启动路径仍然偏保守

当前状态：

- 已经去掉了最重的全量 parquet 展开
- cache 也改成按需缓存

但还缺：

- 更强的边训练边预取
- 更明确的 warm shard / hot shard 策略
- 更成熟的 dataset startup latency 压缩

影响：

- 当前版本比旧版轻很多
- 但启动时仍然会在：
  - NCCL init
  - model load
  - dataset build
  - first shard touch
  这几个阶段花掉一些时间

## 当前结论

当前 `FlashVSR v4` 数据侧已经不是旧版那种“项目内零散特化实现”了。

更准确地说：

- 已经明显向 `ref_big` 的分层和对象缓存思路靠拢
- 已经具备正式训练可用性
- 但还没有把 `ref_big` 的对象层、节流层、复用层、joint 主线设计完整搬平

如果要一句话概括：

- 当前大致是“结构方向对了，轻量可用版已经成立”
- 但距离 `ref_big` 完整工程化版本，仍然差一轮对象层收口和吞吐稳定性优化
