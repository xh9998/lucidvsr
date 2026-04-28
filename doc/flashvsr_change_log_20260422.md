# FlashVSR Change Log up to 2026-04-22

这份文档按当前代码和仍可追溯的实验/脚本整理，目标是把最近这一轮 FlashVSR 训练线里已经落地的改动完整记下来，方便后面汇报、复现实验和继续清理。

说明：
- 这里只记录已经进代码或已经形成固定 config/sh 的改动。
- 若某个方向只做过临时验证、但未并入当前主线，我会单独标明。
- 路径统一按本地仓库写；远端路径把仓库根替换成 `/mnt/task_runtime/lucidvsr` 即可。

## 1. 训练主线版本概览

### `v2`

用途：
- 当前最稳定的 LoRA + `lq_proj_in` 训练线
- 用于 17 帧 / 89 帧视频实验
- 没有 image/video joint packed 逻辑

核心文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v2.py`

特点：
- 主要服务纯视频训练
- validation / inference 逻辑都从这里延伸出去过多个 debug 版本
- 曾长期使用 `alpha=5` 做过拟合和正式训练对照

### `v3`

用途：
- 全量微调（full finetune）路线

核心文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v3_1.py`

特点：
- 重点排查过 GC / ZeRO2 / deepspeed activation checkpointing 的组合
- 当前可工作的落地版本是 `ZeRO2 + GC + no DS activation checkpointing`

### `v4`

用途：
- 数据侧重构后的 joint 训练线
- 支持多源数据和 image/video packed attention 的前一阶段版本

核心文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v4_fullb.py`

特点：
- 引入了新的数据入口和多源采样
- 后面又扩展了 aliyun 退化版本
- 这一版暴露出“换数据后初始 loss 抬升到 5 左右”的问题

### `v5.1 / v5.2 / v5.3`

用途：
- image + video joint training 的明确对照实验线

核心文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_1_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_2_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`
- `diffsynth/models/wan_video_dit_joint_v5.py`

当前分工：
- `v5.1`
  - grouped image sample
  - image 权重按“更接近视频首帧公平”去设计
- `v5.2`
  - grouped image sample
  - image 权重按“更接近整段视频等权”去设计
- `v5.3`
  - author-style paired sample
  - 一个 sample 里显式包含一条 video branch 和一条 image pseudo-video branch
  - attention 用 packed joint 逻辑

## 2. 数据集与采样逻辑改动

### 2.1 旧 streaming dataset 改为统一远端发现入口

核心文件：
- `wanvideo/data/flashvsr/datasets/streaming_dataset.py`
- `wanvideo/data/flashvsr/datasets/conductor_bridge_v2.py`

已落地改动：
- 统一了本地目录、manifest、`conductor://`、`s3://`、`blobby://` 的发现入口
- 增加了远端发现缓存：
  - `REMOTE_DISCOVERY_CACHE_DIR`
  - rank0 列目录，其他 rank 等缓存结果
- 不再要求 dataset 自己到处单独开远端对象
- 支持 manifest 文件作为数据入口，而不是强制扫整棵目录

目的：
- 降低大规模远端文件枚举的重复开销
- 避免 16 卡每个 rank 都独立扫同一批远端对象

### 2.2 新 Takano 视频接入

经历过两种形式：

1. `parquet + direct clip path`
- 文件：
  - `wanvideo/data/flashvsr/datasets/parquet_tar_dataset_v2.py`
- 用于读取新版 Takano parquet 索引，行里直接给真实 mp4 path

2. tar / manifest 形式
- 文件：
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
  - `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`
- `wanvideo/data/flashvsr/manifests/generated/takano_original_video_train_all.txt`（已废弃，后续统一改回直接读四个 Takano tar 子目录，避免静态 manifest 跟不上数据更新）
- 当前 `v5.x` 正式线主要走这条

当前 `v5.x` 的 Takano 入口：
- 使用四个子目录拼接：
  - `takano-video-20231214-train/1080p/`
  - `takano-video-20231214-train/4k/`
  - `takano-video-20250205-train/1080p/`
  - `takano-video-20250205-train/4k/`

### 2.3 Yubari 数据接入

核心文件：
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`

当前做法：
- 不再把 metadata root 作为主入口
- 直接从 `conductor://ve-t2222-datasets/projects/yubari/1.1/data/video/` 读取视频 tar
- 训练时按 tar shard 流式读取

原因：
- metadata/parquet 路线在这个数据源上不稳定、也更重
- tar 直读更接近旧 Takano 的稳定逻辑

### 2.4 picked17k 图像接入

核心文件：
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`

当前入口：
- `picked17k_image_tar_url`
- 实际常用的是 manifest：
  - `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`

说明：
- 图像路径量很大，直接全量目录扫描不合适
- 先生成 manifest，再按 manifest 流式取图

### 2.5 image / video joint 的三种数据组织方式

#### `v5.1 / v5.2`

文件：
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
- `wanvideo/data/flashvsr/datasets/joint_batching_v5.py`

逻辑：
- 视频样本单独产出
- 图像先按单帧处理
- 再把若干张图打包成一个 grouped image sample
- 组大小随 `num_frames` 动态变化：
  - `projector_group_size = max(1, (num_frames - 1) // 4)`
  - 17 帧时当前是 4
  - 89 帧时会自动变大

#### `v5.3`

文件：
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`

逻辑：
- 一个 sample 同时产出：
  - `video / lq_video`
  - `image_video / image_lq_video`
- image branch 在数据侧先生成 pseudo-video
- 当前 image branch raw frame 数：
  - `((num_frames - 1) // 4) + 1`
  - 17 帧对应 5

### 2.6 最近新增的采样过滤逻辑

文件：
- `wanvideo/data/flashvsr/datasets/streaming_dataset.py`
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v5.py`
- `wanvideo/data/flashvsr/datasets/tar_streaming_dataset_v53.py`

新增规则：
- 视频 clip 不再允许从首帧开始采样
  - `_select_clip()` 里 `start` 改成至少为 1
  - 如果一条视频只够从首帧起采样，则直接丢弃
- 图片 / 视频原始分辨率若任一边小于目标训练尺寸，直接丢弃
  - 不再尝试用上采样把小图/小视频救回来

目的：
- 避免模型过度依赖视频首帧
- 保证训练输入都来自足够大的原始素材
- 避免“先上采样再 crop”带来的伪细节

### 2.7 crop 策略

当前状态：
- 训练主线使用 `ImageCropAndResize`
- 近期已经明确把方向切到：
  - 足够大则 crop
  - 不够大直接丢

之前存在过的做法：
- 不够大时先上采样再 crop

现在的代码方向是：
- 对 `v5` 主线不再容忍 undersized 样本
- 这比“强行 resize 救回来”更干净

## 3. 退化系统改动

### 3.1 原始版退化

文件：
- `wanvideo/data/flashvsr/degradation/`
- 默认配置来自原始 FlashVSR / RealESRGAN 风格

特点：
- 训练时 HR/LQ 保持同尺寸
- 内部存在退化后再恢复尺寸的逻辑

### 3.2 aliyun 退化版本

参考来源：
- `aliyunvsr/vsr_train_singlemachine_degra.py`
- `aliyunvsr/Vdegra_btchw.py`
- `aliyunvsr/configs/Vdegra.yml`

本仓库接入方式：
- 通过
  - `wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1.yaml`
- 再由
  - `wanvideo/data/flashvsr/degradation/__init__.py`
  - `build_degradation_model(...)`
 统一构建

最近统一动作：
- `v5.1 / v5.2 / v5.3` 原先都还是默认原始退化
- 现已补出 aliyun 版 config/sh

新增配置文件：
- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_1_lora_17f_fullsources_bs24_lr1e5_aliyundegra.yaml`
- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_2_lora_17f_fullsources_bs24_lr1e5_aliyundegra.yaml`
- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_lora_17f_fullsources_bs16_lr1e5_aliyundegra.yaml`

新增启动脚本：
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-1-Lora-17f-FullSources-bs24-lr1e5-AliyunDegra.sh`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-2-Lora-17f-FullSources-bs24-lr1e5-AliyunDegra.sh`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Lora-17f-FullSources-bs16-lr1e5-AliyunDegra.sh`

## 4. Packed attention / image-video joint 模型侧改动

### 4.1 `wan_video_dit_joint_v5.py`

文件：
- `diffsynth/models/wan_video_dit_joint_v5.py`

职责：
- 构造 joint Wan 模型
- 处理 segment-based packed tokens
- 为 image/video joint attention 提供 packed/segment 长度信息

关键变化：
- 不再把 image/video 简单当成完全相同长度的 dense batch
- 支持按 segment 长度做 patchify / packed token 组织
- 为后续 image/video 不互看提供边界信息

### 4.2 `v5.3` branch-aware pipeline

文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

已经完成的内容：
- video branch 和 image branch 保留各自 raw frame count
- 不再把两边 raw frame 先粗暴合成一个唯一的 `num_frames`
- packed 主要发生在 latent/token 层

意义：
- 更接近作者式 paired sample
- 避免之前 `17 + 5 -> 22` 强行碰全局 `4n+1` 约束

当前仍需继续观察的点：
- 16 卡 `v5.3` 在某些设置下仍出现 barrier 后首 batch 推进慢 / 卡住
- 说明 branch-aware 结构虽然已落地，但分布式训练稳定性还要继续验证

## 5. Validation / inference 改动

### 5.1 validation 的 `sample_seed` bug 修复

文件：
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_1_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_2_lora.py`
- `wanvideo/model_training/flashvsr/train_flashvsr_stage1_v5_3_lora.py`

问题：
- grouped image sample 的 `sample_seed` 不是标量，而是多元素 tensor
- validation 保存 `meta.json` 时直接 `.item()` 会炸

已修复：
- 新增 `_serialize_sample_seed(...)`
- 标量 seed 仍写成 int
- 向量 seed 改写成 `List[int]`

影响：
- `v5.1 / v5.2 / v5.3` 都消除了“训练正常、存 ckpt 触发 validation 时炸掉”的同类问题

### 5.2 v2 系 validation / inference 对齐排查

这部分主要分布在：
- `train_flashvsr_stage1_v2.py`
- `train_flashvsr_stage1_v2_debug_compare.py`
- `wanvideo/model_inference/flashvsr/...`

做过的事：
- 对比过 validation 和外部 inference 的 scale / 输入路径 / tensor 化方式
- 明确发现过：
  - 训练内 validation 的 `alpha=5` 看起来正常
  - 外部脚本同样 `alpha=5` 会明显失真

状态：
- 这个问题还没有完全定性
- 已经排除过一部分明显的 scale / pipeline 分支不一致

## 6. 分布式训练 / resume / 远端启动改动

### 6.1 16 卡双机统一启动脚本体系

特征：
- 所有正式实验尽量落到独立 `history/*.sh`
- 启动命令不再散落在命令行
- 远端统一在 `tmux remote` 里持久运行

### 6.2 resume 体系

已经落地的经验：
- resume 不只是加载 safetensors
- 还要检查 training state / scheduler / optimizer / random states
- 双机时主从机 artifacts 不共享，resume 依赖文件要先补齐到从机
- 传输方式优先：
  - 主机上传 `s3://lxh/tmp/...`
  - 从机再下载

### 6.3 远程工作流规范

已沉淀到 skill：
- 先在本地固定 tmux 会话里 `bolt task ssh`
- 远端再进入 `tmux remote`
- 长训练统一发到 `remote` 的 window
- 训练脚本必须自己 tee 到 `run.log`

## 7. 当前三条 `v5` aliyun 实验状态（2026-04-22）

### `v5.1`
- 机器：母机1
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_1_lora_17f_fullsources_bs24_lr1e5_aliyundegra_20260422_133600`
- 状态：
  - 已过 barrier
  - `wandb` 在线
  - 已出首个 loss：
    - `step=1 loss=0.484587`

### `v5.2`
- 机器：母机2
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_2_lora_17f_fullsources_bs24_lr1e5_aliyundegra_20260422_133700`
- 状态：
  - 已过 barrier
  - `wandb` 在线
  - 已出首个 loss：
    - `step=1 loss=0.413891`

### `v5.3`
- 机器：母机3
- 实验目录：
  - `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs16_lr1e5_aliyundegra_20260422_133800`
- 状态：
  - 已重新对齐主从机时间戳并重启
  - 还在继续盯启动稳定性
  - 当前不能像 `v5.1 / v5.2` 一样宣称已经健康出 loss

## 8. 后续建议

### 优先级 1
- 继续把 `v5.3` 盯到首个 loss 或明确 traceback

### 优先级 2
- 如果 `skip first frame + undersized discard` 导致样本过滤率明显升高
  - 增加统计日志：
    - 因分辨率丢弃多少
    - 因 `start=0 only` 丢弃多少

### 优先级 3
- 后续再补一份更“按文件逐一列举”的 code inventory
  - 适合给外部汇报或代码清理前使用
  - 这份文档先强调训练主线和行为变化
