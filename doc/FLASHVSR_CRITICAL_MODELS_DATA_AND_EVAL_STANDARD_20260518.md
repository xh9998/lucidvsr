# FLASHVSR CRITICAL MODELS DATA AND EVAL STANDARD 20260518

这个文件是 FlashVSR 后续汇报和对比测试的标准入口。以后要找关键 checkpoint、常用测试集、推理脚本、训练数据 manifest、云端备份位置，先看这里。

## 当前 PPT Benchmark

- 目标：生成新的 20 个 89f 轻退化合成测试视频，并和 11 个真实测试视频一起测试 7 个方法。
- 新合成测试集：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset20_89f_takano20250205_light_x4_lq_20260518`
- 新合成测试集备份：`s3://lxh/data/test/testset20_89f_takano20250205_light_x4_lq_20260518`
- 真实测试集：`/mnt/task_wrapper/user_output/artifacts/data/inference/challenging_test_lxh_89f_320x192_resizecrop_20260503`
- 真实测试集备份：`s3://lxh/data/test/challenging_test_lxh_89f_320x192_resizecrop_20260503/`
- Benchmark 输出：`/mnt/task_wrapper/user_output/artifacts/inference/ppt_benchmark_20synthetic_11real_20260518`
- Benchmark 输出备份：`s3://lxh/data/test/ppt_benchmark_20synthetic_11real_20260518`
- 运行机器：`6ikhpjzv3z`
- GPU 使用规则：4 张卡都可用，但同一时刻一张卡只跑一个模型的一个数据集，不叠任务，保证时间统计可解释。

## 关键内部模型

| 阶段 | 汇报名 | checkpoint | 用途 | 备份位置 |
|---|---|---|---|---|
| Stage1 | old v5.3.5 535 step10000 | `/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518/stage1_v535_step10000.safetensors` | 老一阶段 89f 基础 restoration，对应二阶段起点之一 | `s3://lxh/models/flashvsr/critical_ppt_20260518/stage1_v535_step10000.safetensors` |
| Stage1 | USMGT Takano20250205 step3000 | `/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518/stage1_usmgt_takano20250205_step3000.safetensors` | 加 GT sharpness 后的一阶段 finetune 结果 | `s3://lxh/models/flashvsr/critical_ppt_20260518/stage1_usmgt_takano20250205_step3000.safetensors` |
| Stage2 | v6.4.1 step6000 | `/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518/stage2_v641_step6000.safetensors` | 三阶段训练的二阶段 student/teacher 起始 pretrain；50-step streaming/KV-cache 推理 | `s3://lxh/models/flashvsr/critical_ppt_20260518/stage2_v641_step6000.safetensors` |
| Stage3 | v7-D3.2 step2000 | `/mnt/task_wrapper/user_output/artifacts/critical_models/flashvsr_ppt_20260518/stage3_v7d32_step2000.safetensors` | 当前三阶段 one-step streaming 模型 | `s3://lxh/models/flashvsr/critical_ppt_20260518/stage3_v7d32_step2000.safetensors` |

原始来源：

- Stage1 old 535 step10000：`s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage1_release_48gpu_v5_3_5_resume_step3000_seed20260501_lora_89f_fullsources_bs1_lr1e5_aliyundegra_randomproj_img5_nonstreamproj_aligned23_20260502_013300/output/step-10000.safetensors`
- Stage1 USMGT step3000 原始实验路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage1_release_16gpu_v5_3_5_lora_89f_takano20250205_4k_bs1_lr5e6_aliyundegra_usmgt_warmstart_step10000_20260516_usmgt_takano20250205_warmstart10000_gpuworker2_spawn/output/step-3000.safetensors`
- Stage1 USMGT step3000 S3 fallback：`s3://lxh/tmp/stage1_usmgt_takano20250205_warmstart10000_step3000_20260518/step-3000.safetensors`
- Stage2 v6.4.1 step6000 原始实验路径：`/mnt/task_wrapper/user_output/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- Stage2 v6.4.1 step6000 S3 fallback：`s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts/exp/train_stage2_release_40gpu_v6_4_1_lora_89f_videoonly_bs1_lr1e5_blocksparse_worker2_val_20260512_164100/output/step-6000.safetensors`
- Stage3 v7-D3.2 step2000：`s3://lxh/tmp/stage3_v7_d3_2_100plus_ckpts_20260516/step-2000.safetensors`

## 外部 Baseline 模型

| 方法 | 模型目录 | 标准推理 |
|---|---|---|
| FlashVSR official | `/mnt/models/FlashVSR-v1.1` | `wanvideo/model_inference/flashvsr/history/run_flashvsr_full_dir_20260421.sh` |
| SeedVR3B | `/mnt/models/SeedVR-3B` | `wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh` with `MODEL_KIND=seedvr1` |
| SeedVR2-3B | `/mnt/models/SeedVR2-3B` | `wanvideo/model_inference/flashvsr/history/run_seedvr_dir_20260421.sh` with `MODEL_KIND=seedvr2` |

## 标准推理入口

| 类型 | 标准脚本 | 关键设置 |
|---|---|---|
| Stage1 nonstreaming | `wanvideo/model_inference/flashvsr/history/run_stage1_v5_3_aligned_dir.sh` | `LQ_PROJ_TEMPORAL_MODE=nonstreaming_aligned`, `NUM_INFERENCE_STEPS=50`, `INPUT_BICUBIC_UPSCALE=4.0`, `COLOR_FIX_METHOD=adain` |
| Stage2 streaming | `wanvideo/model_inference/flashvsr/infer_flashvsr_stage2_v6_1_batch.py` | `NUM_INFERENCE_STEPS=50`, `stage2_attention_mode=block_sparse_chunk_causal`, `topk_ratio=2.0`, `local_num=-1`, `kv_ratio=3.0` |
| Stage3 one-step streaming | `wanvideo/model_inference/flashvsr/infer_flashvsr_stage3_v7_d_batch.py` | Stage2 同一 streaming/KV-cache 路径，`NUM_INFERENCE_STEPS=1` |
| PPT benchmark 总入口 | `wanvideo/model_inference/flashvsr/history/run_ppt_benchmark_20synthetic_11real_20260518.sh` | 单卡单任务，统一记录 timing，结束后同步 S3 并恢复占卡 |

## 标准测试集生成

- 新 20 合成测试集生成脚本：`wanvideo/data/flashvsr/tests/export_inference_testset20_takano20250205_light_x4_lq.py`
- 旧常用 10 合成测试集准备脚本：`wanvideo/data/flashvsr/tests/run_prepare_v536_eval_testsets_20260503.sh`
- 轻退化配置：`wanvideo/data/flashvsr/degradation/configs/params_aliyun_video_compression_v1_light_x4test.yaml`
- 轻退化规则：GT 取 1280x768、89f、fps=8；LQ 是 1/4 尺寸 320x192；使用 Aliyun degradation，关闭最终 bicubic restore。
- 旧常用 10 合成测试集：`/mnt/task_wrapper/user_output/artifacts/data/inference/testset10_89f_aliyun_light_x4_lq_20260503`
- 旧常用 10 合成测试集备份：`s3://lxh/data/test/testset10_89f_aliyun_light_x4_lq_20260503`

## 训练数据和 Manifest

| 数据 | S3 / conductor 位置 | 本地/远端 manifest |
|---|---|---|
| Takano 20250205 test 4K | `s3://lucid-vr/datasets/takano_original/video/takano-video-20250205-test/4k/` | `wanvideo/data/flashvsr/manifests/generated/takano_video_20250205_test_4k_tar_manifest.txt` |
| Takano 20250205 manifest backup | `s3://lxh/data/mainfest/takano_video_20250205_test_4k_tar_manifest.txt` | `/mnt/task_wrapper/user_output/artifacts/data/manifests/takano_video_20250205_test_4k_tar_manifest.txt` |
| 旧 b8 artifacts | `s3://bolt-prod-2320845741/tasks/b8gkuie2ns/artifacts` | 等价于旧 b8 的 `/mnt/task_wrapper/user_output/artifacts` |
| 真实 challenging test 本机源 | `/Users/lixiaohui/Library/CloudStorage/Box-Box/challenging_test_lxh` | 已处理成 89f 320x192 resize/crop 版本 |

## 运行约束

- 项目代码只改本地 `/Users/lixiaohui/Library/CloudStorage/Box-Box/code/lucidvsr`，等 `sync` 到远端 `/mnt/task_runtime/lucidvsr` 后运行。
- FlashVSR 统一使用 `/mnt/conda_envs/flashvsr/bin/python`。
- SeedVR 统一使用 `/mnt/conda_envs/seedvr/bin/python`。
- 推理完成后必须恢复正常占卡：`bash /mnt/task_runtime/bolt_lxh/occupy/gpu_stress_tc.sh`。
- 如果 conductor 在当前 shell 报凭证问题，先新开 zsh/tmux 或直接在远端 zsh 里试 `conductor s3 ls`，不要直接判定数据不可用。
