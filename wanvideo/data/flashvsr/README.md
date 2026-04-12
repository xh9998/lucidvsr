# FlashVSR 数据目录

这里放的是 FlashVSR 当前使用的数据层实现。

核心文件：

- `datasets/streaming_dataset.py`
- `datasets/parquet_index.py`
- `degradation/realesrgan_kernels.py`

当前主路线：

- `storymotion`：先从 parquet 提炼 manifest，再训练时读 manifest
- `takano`：直接读 tar shard

当前数据层已经支持：

- `global_seed`
- 在线退化
- `storymotion + takano` 混合
- 训练时 tensor 输出
- 最小测试保存 `mp4`

常用脚本：

- `tests/test_streaming_dataset_minimal.py`
- `tests/run_test_storymotion_manifest_cloud.sh`
- `tests/run_test_takano_shards_cloud.sh`
- `tools/build_storymotion_manifest_from_parquet.py`
