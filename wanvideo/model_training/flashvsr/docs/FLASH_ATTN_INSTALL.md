# flash-attn 安装说明

当前这版 Wan / FlashVSR 代码支持：

- 有 `flash-attn` 时优先使用
- 没有时回退到普通 attention

所以 `flash-attn` 不是训练启动的必要条件，但对大配置显存和速度有帮助。

## 推荐安装

先激活训练环境：

```bash
source /mnt/task_runtime/bolt_lxh/use_active_python.sh
```

然后直接装：

```bash
MAX_JOBS=16 /mnt/conda_envs/flashvsr/bin/pip install --no-build-isolation flash-attn
```

如果失败，再走源码：

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=16 /mnt/conda_envs/flashvsr/bin/pip install --no-build-isolation .
```

## 安装后检查

```bash
/mnt/conda_envs/flashvsr/bin/python - <<'PY'
try:
    import flash_attn
    print("flash_attn ok", getattr(flash_attn, "__version__", "unknown"))
except Exception as e:
    print("flash_attn import failed:", e)

try:
    import flash_attn_interface
    print("flash_attn_interface ok")
except Exception as e:
    print("flash_attn_interface import failed:", e)
PY
```
