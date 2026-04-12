# FlashVSR 数据层说明

## 目标

当前这版数据层目标很明确：

- 让 Stage 1 训练能直接吃真实视频数据
- 同时兼容 `storymotion` 和 `takano`
- 在线生成 `lq_video`

## 两条主数据路线

### storymotion

当前推荐做法不是直接训练时频繁读原始 parquet，而是：

1. 先从 parquet 提炼轻量 manifest
2. 训练时只读 manifest

manifest 里保留训练真正会用到的字段，例如：

- `media_url`
- `caption`
- `width`
- `height`
- `fps`
- `duration`
- `num_frames_est`
- `source_dataset`

### takano

`takano` 直接按 tar shard 读，不再把 parquet 当训练主入口。

也就是：

- parquet 只适合做离线分析
- 真正训练时直接读 tar shard 文件夹

## 数据集输出

当前训练用 dataset 会输出：

- `video`
- `lq_video`

训练模式下会直接输出 tensor，而不是 `PIL.Image` 列表。

## 随机性

当前随机性主要在 dataset：

- 样本顺序
- clip 起点
- pseudo-video 轨迹
- control dropout
- 退化 seed

现在已经统一收进 `global_seed`。

## 最小测试

当前最常用的测试有两类：

- `storymotion manifest`
- `takano shards`

测试会输出：

- 第一帧图
- clip 视频

现在默认存 `mp4`，不再存 `gif`。
