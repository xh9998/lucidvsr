# FlashVSR v5.3.x FlashInit 重启记录

日期：2026-04-23

这份文档只记录这一轮 `v5.3 / v5.3.1 / v5.3.2` 的统一重启，不写更早历史。

## 1. 这轮重启的目的

- 统一把 `lq_proj_in` 改成从官方 `FlashVSR-v1.1` 初始化
- 不再让 `lq_proj_in` 走纯零初始化
- 把 `batch_size` 从 `16` 降到 `12`
- 目标是：
  - 保持 GPU 利用率接近打满
  - 不再把显存顶到过激状态
  - 看早期 LQ 注入是否能学得更快

统一改动：

- `lq_proj_checkpoint=/mnt/models/FlashVSR-v1.1/LQ_proj_in.ckpt`
- `zero_init_lq_proj_in=false`
- `batch_size=12`

## 2. 停掉的旧实验目录

### 母机1：旧 `v5.3.2`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs16_lr1e5_aliyundegra_20260422_233000`

### 母机2：旧 `v5.3.1`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs16_lr1e5_aliyunhalf_20260422_203200`

### 母机3：旧 `v5.3`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs16_lr1e5_aliyundegra_20260422_203300`

## 3. 新增的 config / sh

### `v5.3.2`

- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-2-Lora-17f-Yubari-FrameImage-bs12-lr1e5-AliyunDegra-FlashInit.sh`

### `v5.3.1`

- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-1-Lora-17f-FullSources-bs12-lr1e5-AliyunHalf-FlashInit.sh`

### `v5.3`

- `wanvideo/model_training/flashvsr/configs/history/stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit.yaml`
- `wanvideo/model_training/flashvsr/lora/history/FlashVSR-Stage1-Release-16GPU-v5-3-Lora-17f-FullSources-bs12-lr1e5-AliyunDegra-FlashInit.sh`

## 4. 最终成功启动的新实验目录

### 母机1：`v5.3.2`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_2_lora_17f_yubari_frameimage_bs12_lr1e5_aliyundegra_flashinit_20260423_150800`

早期 loss：

- `step=1 loss=0.321344`
- `step=2 loss=0.109715`
- `step=3 loss=0.193490`

速度：

- step time 大约 `94-96s`

### 母机2：`v5.3.1`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_1_lora_17f_fullsources_bs12_lr1e5_aliyunhalf_flashinit_20260423_154400`

早期 loss：

- `step=1 loss=0.208661`

速度：

- 首步大约 `198.7s`

### 母机3：`v5.3`

- `/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_lora_17f_fullsources_bs12_lr1e5_aliyundegra_flashinit_20260423_154500`

早期 loss：

- `step=1 loss=0.092752`

速度：

- 首步大约 `199.4s`

## 5. 这轮重启里额外修掉的问题

### 图像 txt manifest 没真正接进 `v5.3 / v5.3.1`

这次排查确认：

- 当前 `v5.3 / v5.3.1` 的图像源用的是：
  - `/mnt/task_wrapper/user_output/artifacts/data/highres_manifest/highres_image_manifest_train.txt`
- 这个文件是一行一个远端 `jpg` 路径的大 txt
- 之前代码里 `image_manifest_urls` 没有真正接进 image iterator

后果是：

- 配置上看起来有 image source
- 实际运行时 image branch 取不到图
- fixed validation 在 joint dataset 上取样时直接报：
  - `RuntimeError: generator raised StopIteration`

修复后：

- 现在 image source 同时支持：
  - tar 根目录
  - txt manifest 指向散图

### fixed validation 不该再被 image branch 卡死

本轮也确认了训练侧原则：

- fixed validation sample 只需要视频样本
- 不应该因为 image branch 有问题就直接让 validation 启动失败

当前这轮成功启动说明：

- `v5.3 / v5.3.1` 已经跨过了之前的 fixed validation 卡点

## 6. 当前结论

- `flashinit + bs12` 这套目前三条线都已起稳
- 母机1 `v5.3.2` 明显更快
- 母机2 `v5.3.1` 和母机3 `v5.3` 都已成功出到首个 loss
- 这三条现在可以视作本轮后续继续观察的有效实验线
