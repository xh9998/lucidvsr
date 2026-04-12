# Config History

这个目录只放实验性或一次性配置的归档副本。

根目录里长期保留的标准配置：

- `stage1_release_8gpu.yaml`
- `stage1_release_8gpu_v1.yaml`
- `stage1_release_8gpu_v2.yaml`

如果某个配置只是为了：

- 临时改 batch
- 临时改 frame 数
- 临时加 debug/overfit
- 某次问题定位

则应归档到 `history/`，避免根目录越来越混乱。

建议配套规则：

1. 标准配置放根目录
2. 实验配置复制到 `history/`
3. 实验启动时在实验目录内保留 `resolved_args.*` 和 `snapshot/`
