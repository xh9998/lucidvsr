# 参考仓分析记录

这份记录只保留最终有用的结论。

## ref_big

`ref_big` 能稳定读 conductor/tar，不是因为原版 `webdataset` 自己就支持，而是因为它的路线更像：

- `fsspec / apple_fsspec`
- datapipe
- tar 流式展开

关键结论：

- 它不是 `parquet -> 单个 tar member` 这种读法
- 它更像 `先确定 shard，再顺序流式读整个 shard`

## WanControl

`WanControl` 也不是直接依赖原版 `webdataset`。

它做法更激进：

- 对 `webdataset` 相关读 tar 逻辑做了 patch

## 当前决策

当前 FlashVSR 数据层没有直接照抄 `WanControl` 的 patched webdataset。

最后采用的是：

- `storymotion`：manifest
- `takano`：直接 shard

这样更简单，也更容易维护。
