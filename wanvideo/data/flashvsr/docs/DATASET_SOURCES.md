# FlashVSR Dataset Sources

这份文件只记录当前 `FlashVSR` 训练里用过或准备切换的数据源路径，方便后续统一改训练配置。

## 当前旧版 Takano 路径

这些是当前大多数训练 config 里还在使用的旧路径：

- Takano1 old
  - `conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier1-qwen3/1080p/`
- Takano2 old
  - `conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier2-qwen3/1080p/`

说明：

- 这两条旧路径已经在 `stage1_release_*` 多份 config 里使用。
- 目前 `takano` 的 seed 复现测试也是先基于这两条旧路径做的。

## 新版 Takano 路径

这两条是用户新提供的替代版本。据反馈：

- clip 已切成约 `5s`
- 质量更高
- 总量约 `27M`

### Takano1 new

- 视频数：`22,033,968`
- index link：
  - `s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/`
- 当前判断：
  - 这条目录下直接是大量 parquet 分片，例如：
    - `00000.parquet`
    - `00001.parquet`
    - `00002.parquet`
  - 说明它至少暴露出了 `parquet metadata/index` 层，不是一个已经确认的 tar shard 叶子目录。
  - 当前更像是应通过 `metadata_url` 读取，而不是直接当旧版 takano `internal_url tar shards` 用。

### Takano2 new

- 视频数：`15,320,040`
- index link：
  - `s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/`
- 当前判断：
  - 这条目录下直接是大量 parquet 分片，例如：
    - `00000.parquet`
    - `00001.parquet`
    - `00002.parquet`
  - 说明它至少暴露出了 `parquet metadata/index` 层，不是一个已经确认的 tar shard 叶子目录。
  - 当前更像是应通过 `metadata_url` 读取，而不是直接当旧版 takano `internal_url tar shards` 用。

## 切换建议

后续训练默认建议优先切到这两条新版 Takano。

推荐替换方向：

- 旧版训练入口（直接 tar shards）：
  - `conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier1-qwen3/1080p/`
  - `conductor://lucid-vr/datasets/dryrun_20m/video/takano-video-tier2-qwen3/1080p/`
- 新版候选入口（parquet metadata root）：
  - `s3://ve-t2222-datasets/datasets/takano-video-tier1/duration_cut-dedup-shuffled/`
  - `s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/duration_cut-dedup-shuffled/`

更保守的切换建议：

- 先不要直接把新版路径塞进 `internal_url`
- 先尝试作为：
  - `metadata_url`
- 并把：
  - `metadata_source=takano`
  一起打开做最小 smoke test

## 注意事项

- 这里只是记录，不代表现有训练 config 已经全部切换。
- 如果后续正式切换：
  - 需要同步改训练 config 里的 `internal_url`
  - 需要补一次最小 dataset smoke test
  - 需要重新验证 seed 复现性
  - 需要确认 `FlashVSRStreamingDataset` 对这两条新 index 的读取方式是否与旧 takano 完全兼容
- 2026-04-13 补充：
  - 已在本机通过 `conductor s3 ls ... | head -n 120` 实测：
    - Takano1 new 根目录下直接是 `00000.parquet ...`
    - Takano2 new 根目录下直接是 `00000.parquet ...`
  - 因此这两条路径目前可明确视为 parquet metadata root。
  - 已进一步确认 parquet 行里的 `path` 直接指向 clip mp4，例如：
    - `s3://ve-t2222-datasets/datasets/takano-video-tier1/video-clips-v2/...clip00000.mp4`
    - `s3://ve-t2222-datasets/datasets/takano-video-tier2-10m-final/video-clips-v2/...clip00000.mp4`
  - 这批新 Takano 不是旧版 `path_lucid + tar member` schema，而是 direct clip path parquet schema。

## 图像数据

- 图像根目录：
  - `s3://takano-assets/20231106/high_resolution/`
- 本机 `conductor s3 ls ... | head` 看到的一级结构：
  - `bigstock/`
  - `metadata/`
  - `metadata_parquet/`
  - `metadata_split_parquet/`
  - `pond5/`
  - `sstk/`

当前实现结论：

- 新的 `parquet_tar_dataset_v2.py` 已支持把图像读取成样本源。
- 这次已按论文 Stage 1 的要求，把**新的 v2 数据集**里的图像入口改成：
  - **单帧视频 (`f=1`)**
- 不再默认做“会动的伪视频”。
- 图像入口目前推荐优先走 metadata parquet，而不是直接递归扫描整棵图片目录。

推荐的 image metadata parquet 入口：

- `s3://takano-assets/20231106/high_resolution/metadata_split_parquet/train/apple_gen_20231117_full_metadata_6_5_0.parquet`

已确认字段：

- `ASSET_NAME`
- `DESCRIPTION`
- `TARGET_S3_PATH`
- `MAX_WIDTH`
- `MAX_HEIGHT`
- `TYPE`
- `SIZE`

当前新 v2 数据集已支持：

- 从 image parquet 读取记录
- 用 `TARGET_S3_PATH` 定位真实图片
- 直接产出 `f=1` 单帧视频样本

重要边界：

- 当前主训练代码还**没有**实现论文中的 image/video block-diagonal segment mask。
- 因此现在只能说：
  - 新 v2 数据集可以产出 `f=1` 的图像样本
  - 但还没有把论文里“图像和视频在同一 batch 中联合训练”的注意力掩码完整接通

## Yubari 数据

- 视频根目录：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/video/`
- 元数据根目录：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/metadata/`
- 数据量：
  - `1,342,334` videos

本机 `conductor s3 ls ... | head` 结果：

- `video/` 下是成对的：
  - `example.parquet`
  - `example.tar`
  - `part-000000.parquet`
  - `part-000000.tar`
  - ...
- `metadata/` 下也是成对的：
  - `example.parquet`
  - `example.tar`
  - `part-000000.parquet`
  - `part-000000.tar`
  - ...

进一步检查结果：

- `video/*.parquet` 列：
  - `file_name`
  - `header_offset`
  - `data_offset`
  - `data_size`
  - `flags`
- 第一行样例：
  - `1165884718_hd16.mp4`
- `metadata/*.parquet` 同样是上述五列
- 第一行样例：
  - `1165884718_hd16.json`
- `metadata/*.tar` 内部 json 可读字段包括：
  - `title`
  - `caption`
  - `video-quality-assessment`

当前实现结论：

- 新的 `parquet_index.py` / `parquet_tar_dataset_v2.py` 已加入 Yubari 支持：
  - 用 `video/*.parquet + video/*.tar` 取 mp4
  - 用 `metadata/*.parquet + metadata/*.tar` 取同 basename 的 json sidecar
 - `caption_text` 优先从 sidecar 里的 `caption/title` 提取
- 对于 Yubari，当前 smoke 建议先走：
  - `video/example.parquet`
  - `metadata/example.parquet`
  做最小验证

实现更新：

- `parquet_index._discover_parquet_urls(...)` 现在对**显式 `.parquet` 路径**直接短路返回，
  不再通过远端文件发现把 `example.parquet` 错误扩展成整个 root。
- 这使得 Yubari 的 `example.parquet` smoke 可以稳定落在“单个 parquet + 单个 tar 对”上。

如果后续仍然觉得 Yubari parquet root 太重，可以直接切到 tar/webdataset 路线：

- 视频 tar 根目录：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/video/`
- 元数据 tar 根目录：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/metadata/`
- 其中已确认存在：
  - `part-000000.tar`
  - `part-000001.tar`
  - ...

原因：

- 当前 `load_yubari_records(...)` 仍然会先把 sidecar metadata parquet 全部扫一遍再建索引
- 直接拿整个 `metadata/` 根目录做 smoke 会很慢
- 正式训练前应继续把 Yubari 索引改成“按需读取 / 限量读取”的形式

2026-04-16 补充：

- 已新增 `yubari_video_tar_url / yubari_sidecar_tar_url` 入口，允许 Yubari 直接走纯 tar 读取，不再依赖 parquet root 扩展。
- 当前 smoke 采用：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/video/example.tar`
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/metadata/example.tar`
- 这条纯 tar 路线已确认：
  - 可以出样本
  - 开启退化后，同 seed 的 `video_hash / lq_video_hash` 一致
  - diff seed 会改变 `sample_seed / video_hash / lq_video_hash`
- 注意：
  - `example.tar` 里基本只有一个底层视频 `1165884718_hd16.mp4`
  - 因此 diff seed 时 `sample_id` 仍可能相同，但 clip 窗口和退化结果会变化
- 当前判断：
  - Yubari 做 bounded smoke/debug：精确 parquet 或 `example.tar` 都可
  - Yubari 做正式训练：更推荐纯 tar / webdataset 风格，而不是 parquet root 发现

2026-04-16 再补充：

- Yubari 后续默认不再依赖 `metadata/`，只使用一个入口：
  - `s3://ve-t2222-datasets/projects/yubari/1.1/data/video/`
- 当前训练约定：
  - config / sh 里只写 `yubari_video_tar_url`
  - 不再写 `yubari_video_metadata_url`
  - 不再写 `yubari_sidecar_metadata_url`
  - 不再默认写 `yubari_sidecar_tar_url`
- `video/` 下的 `part-*.parquet` 作为同目录 `part-*.tar` 的 byte-range index 使用：
  - `file_name`
  - `data_offset`
  - `data_size`
- 当前实现会在发现 `part-*` 时自动跳过 `example.*`，避免 smoke 只落在 `example.tar`。
- 新增读取方式：
  - 扫描 `video/` 下的 `part-*.parquet / part-*.tar`
  - 用 parquet 里的 `data_offset / data_size` 直接对 tar 做 byte-range 读取
  - 不再顺序扫描整个 4GB tar
  - 不再读取 `metadata/` sidecar
- 已在母机2上跑过 Yubari root seed 验证，退化开启：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_yubari_video_root_noexample_seed_repro_17src_20260415_142739`
- 已在本地新 tmux 会话里确认当前 `video/` shard range：
  - `part-000000.tar` 到 `part-006714.tar`
  - `count=6715`
  - `has_gap=False`
- 因此后续 Yubari 默认建议直接按 range 构造：
  - `yubari_shard_start=0`
  - `yubari_shard_end=6714`

## Tar-Only V3 数据方案

2026-04-21 补充：

为了避免 `parquet_v2` 路线里的索引发现和 sidecar/index 构建成本，新增了一个只走 tar 的 `tar_v3` 数据方案。

设计目标：

- 不走 parquet metadata root
- 不走 image parquet
- 不走 Yubari 的 byte-range parquet index
- 只扫描 tar 并按 webdataset 风格顺序/轻 shuffle 读取

当前 `tar_v3` 只使用两个源：

- Yubari 视频
  - 采样概率：`0.9`
- picked17k 图像
  - 采样概率：`0.1`

### Yubari for tar_v3

入口：

- `conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/`

当前约定：

- `tar_v3` 下只把它当成 tar root
- 不再读取同目录 `part-*.parquet`
- 不再读取 `metadata/` sidecar root
- 训练时直接顺扫/分片消费 `part-*.tar`

判断：

- 这条比当前 Yubari 的 parquet/index 路线更轻
- 尤其适合先追求启动速度和稳定吞吐
- 代价是失去 byte-range 精确索引和 sidecar caption

### picked17k image for tar_v3

Whole set：

- `blobby://takano_analysis/images_high_resolution/17k_set`

已知子路径：

- outdoor scene
  - `blobby://takano_analysis/images_high_resolution/17k_set/outdoor_scene/{subcat}/{1st_asset_id}.tar`
- general
  - `blobby://takano_analysis/images_high_resolution/17k_set/general/{1st_asset_id}.tar`

当前 `tar_v3` 约定：

- 把这批 tar 当成纯图片 tar root
- 递归扫描所有 `.tar`
- 直接从 tar 中读取图片
- 不再走 metadata parquet

当前实现说明：

- 新增数据集类：
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v3.py`
- 训练入口新增：
  - `dataset_mode=tar_v3`
- 对应参数：
  - `yubari_video_tar_url`
  - `picked17k_image_tar_url`
  - `picked17k_dataset_prob`

重要取舍：

- `tar_v3` 里的 picked17k 图像暂时沿用旧 streaming 的 pseudo-video 路径
- 默认不是 `f=1` single-frame packed 训练
- 这样是为了优先保证：
  - tar-only
  - 启动快
  - 训练栈兼容
  - 不把 image single-frame 的高 loss 问题重新引回主线
  - 不再先扫整个 `video/` 目录
- 本地查询脚本：
  - `wanvideo/data/flashvsr/tools/get_yubari_video_range_local.sh`
- 验证结论：
  - `seed=1` 跑两次，三个样本的 `sample_id / sample_seed / video_hash / lq_video_hash` 全部一致
  - `seed=2` 和 `seed=1` 对比，三个样本全部不同
  - 样本来自 `part-000000.tar`，不是 `example.tar`
  - 该 smoke 为了速度使用 `256x448 / 17 frames / max_source_frames=17`，正式训练仍按训练配置分辨率执行

## 当前新 v2 数据集状态

文件：

- `wanvideo/data/flashvsr/datasets/parquet_index.py`
- `wanvideo/data/flashvsr/datasets/source_index_v2.py`
- `wanvideo/data/flashvsr/datasets/media_reader_v2.py`
- `wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py`

当前支持：

- 新 Takano direct clip path parquet
- 图像数据作为 `f=1` 单帧视频样本
- Yubari `video/part-*.parquet + video/part-*.tar` byte-range 读取
- Yubari 旧 metadata-root / sidecar 路线已降级，不再作为当前训练入口

2026-04-16 seed + degradation 验证：

- 统一输出目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_seed_repro_with_degradation_fix_20260415_134029`
- Takano：
  - `seed1` vs `seed1`：`sample_id / sample_seed / video_hash / lq_video_hash` 全一致
  - `seed1` vs `seed2`：上述字段全不一致
- image：
  - `seed1` vs `seed1`：`sample_id / sample_seed / video_hash / lq_video_hash` 全一致
  - `seed1` vs `seed2`：上述字段全不一致
- Yubari pure tar (`example.tar`)：
  - `seed1` vs `seed1`：`sample_id / sample_seed / video_hash / lq_video_hash` 全一致
  - `seed1` vs `seed2`：
    - `sample_id` 仍相同（同一底层视频）
    - `sample_seed / video_hash / lq_video_hash` 不一致

当前分层：

- index layer
  - `parquet_index.py`
  - `source_index_v2.py`
- media access layer
  - `media_reader_v2.py`
- sampling / dataset layer
  - `parquet_tar_dataset_v2.py`
- experimental joint image/video path
  - `joint_batching_v1.py`
  - `diffsynth/models/wan_video_dit_joint_v1.py`

这次 smoke 输出目录：

- Takano seed smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_20260415_120557`
- Image parquet smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_image_20260415_121228`
- Yubari example smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_yubari_example_20260415_121427`
- Yubari video root no-example seed smoke：
  - `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_yubari_video_root_noexample_seed_repro_17src_20260415_142739`

当前还未补齐：

- 主训练线还没有接入 image/video joint packed varlen self-attention
- 真正的图像/视频同 batch 联合训练还停留在实验分支

2026-04-16 最新 refresh smoke：

- `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_refresh_20260415_125710/takano_seed1`
- `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_refresh_20260415_125710/image_seed7`
- `/mnt/task_wrapper/user_output/artifacts/exp/test_outputs/flashvsr_dataset_v2_m2_refresh_20260415_125710/yubari_example_seed11`
