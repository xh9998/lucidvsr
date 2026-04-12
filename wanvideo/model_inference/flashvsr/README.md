# FlashVSR Stage1 推理与检查

这个目录用于测试 `FlashVSR Stage1` 训练得到的 `step-xxx.safetensors`。

## 先说结论

训练保存下来的 `step-200.safetensors` **不是完整模型**，而是只包含训练时真正可学习的部分：

- `lq_proj_in.*`
- `dit` 上的 LoRA 权重

它 **不包含** 完整的 Wan 1.3B 基座参数，也不包含完整 VAE 权重。

所以推理时必须同时提供：

- Wan 1.3B 基座权重目录
- `prompt_tensor_path`
- 训练导出的 `step-xxx.safetensors`

## 文件

- `inspect_flashvsr_stage1_ckpt.py`
  - 查看 ckpt 里到底存了哪些键
- `infer_flashvsr_stage1.py`
  - 加载 base Wan + FlashVSR Stage1 ckpt，对一个 LQ 视频做推理
- `infer_flashvsr_stage1_v2.py`
  - 对齐 `train_flashvsr_stage1_v2.py` fixed-prompt validation 的 Stage1 推理
- `run_test_flashvsr_stage1_v2.sh`
  - `v2` 推理模板，默认按 `posi_prompt.pth + cfg=1 + 50步`
- `run_test_flashvsr_stage1.sh`
  - 一个可直接改路径的测试模板

## 最常用流程

先检查 ckpt：

```bash
python wanvideo/model_inference/flashvsr/inspect_flashvsr_stage1_ckpt.py \
  --checkpoint_path /mnt/task_wrapper/user_output/artifacts/exp/train_stage1_8gpu_debug_noval_20260407_144928/output/step-200.safetensors
```

再跑推理：

```bash
python wanvideo/model_inference/flashvsr/infer_flashvsr_stage1.py \
  --checkpoint_path /mnt/task_wrapper/user_output/artifacts/exp/train_stage1_8gpu_debug_noval_20260407_144928/output/step-200.safetensors \
  --base_model_dir /mnt/models/Wan2.1-T2V-1.3B \
  --prompt_tensor_path /mnt/task_runtime/FlashVSR/examples/WanVSR/prompt_tensor/posi_prompt.pth \
  --input_video /path/to/lq.mp4 \
  --output_video /path/to/flashvsr_sr.mp4 \
  --height 768 \
  --width 1280 \
  --num_frames 89
```

如果你要对齐 `train_v2` 的 fixed-prompt validation，优先用：

```bash
bash wanvideo/model_inference/flashvsr/run_test_flashvsr_stage1_v2.sh
```

这个脚本本身不会重复 fuse LoRA。它每次都是：

- 新建一个全新的 `WanFixedPromptFlashVSRStage1Pipeline`
- 加载一次当前 ckpt 中的 `lq_proj_in`
- 加载一次当前 ckpt 中的 LoRA
- 推理结束就退出进程

所以不会出现训练里那种“同一个 cached pipe 多次 validation，LoRA 反复融合”的污染问题。

如果你想从训练集拿测试视频，建议先用：

```bash
python wanvideo/model_training/flashvsr/scripts/export_flashvsr_eval_samples.py \
  --config wanvideo/model_training/flashvsr/configs/stage1_release_8gpu_v2.yaml \
  --output_dir /mnt/task_wrapper/user_output/artifacts/eval_samples/v2 \
  --num_samples 3
```

然后把导出的 `sample_xxx/lq.mp4` 作为 `run_test_flashvsr_stage1_v2.sh` 的 `INPUT_VIDEO`。

## ckpt 里存了什么

训练保存时实际走的是：

- `accelerator.get_state_dict(model)`
- `export_trainable_state_dict(...)`
- `flashvsr_stage1_export(...)`

所以最终导出的键会被整理成：

- `lq_proj_in.xxx`
- LoRA 对应的 `dit` 层权重

这也是为什么它不能单独拿来当完整模型直接推理。
