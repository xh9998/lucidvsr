# FlashVSR Stage1 推理与检查

这个目录只保留当前仍建议直接使用的推理入口。阶段性诊断脚本已经移到 `scripts/`。

## 先说结论

训练保存下来的 `step-xxx.safetensors` 不是完整模型，只包含训练时真正可学习的部分：

- `lq_proj_in.*`
- `dit` 上的 LoRA 权重

因此推理时必须同时提供：

- Wan 1.3B 基座权重目录
- `prompt_tensor_path`
- 训练导出的 `step-xxx.safetensors`

## 主目录保留文件

- `inspect_flashvsr_stage1_ckpt.py`
  - 查看 ckpt 里到底存了哪些键
- `infer_flashvsr_stage1_v1.py`
  - 对齐 `train_flashvsr_stage1_v1.py`
- `infer_flashvsr_stage1_v2.py`
  - 对齐 `train_flashvsr_stage1_v2.py`
- `run_test_flashvsr_stage1_v2.sh`
  - 单个 `v2` inference 模板
- `run_batch_flashvsr_stage1_v2.sh`
  - 批量扫描 ckpt 和输入视频目录做 `v2` inference

## scripts/

下面这些已经移到 `scripts/`，它们主要用于阶段性排查，不建议再当主推理入口：

- `infer_flashvsr_validation_style.py`
- `infer_flashvsr_external_mp4_tensor_style.py`
- `infer_flashvsr_release_aligned.py`
- `infer_flashvsr_stage1_wan_aligned.py`
- `inspect_flashvsr_stage1_v2_stats.py`

## 最常用流程

先检查 ckpt：

```bash
python wanvideo/model_inference/flashvsr/inspect_flashvsr_stage1_ckpt.py \
  --checkpoint_path /mnt/task_wrapper/user_output/artifacts/exp/xxx/output/step-200.safetensors
```

如果要对齐 `train_v2` 的 fixed-prompt validation，优先用：

```bash
bash wanvideo/model_inference/flashvsr/run_test_flashvsr_stage1_v2.sh
```

如果要批量扫多个 step 和多个输入视频，优先用：

```bash
bash wanvideo/model_inference/flashvsr/run_batch_flashvsr_stage1_v2.sh
```

## ckpt 里存了什么

训练保存时实际走的是：

- `accelerator.get_state_dict(model)`
- `export_trainable_state_dict(...)`
- `flashvsr_stage1_export(...)`

最终导出的键会被整理成：

- `lq_proj_in.xxx`
- LoRA 对应的 `dit` 层权重

这也是为什么它不能单独拿来当完整模型直接推理。
