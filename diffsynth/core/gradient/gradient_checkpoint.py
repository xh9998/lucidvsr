import torch
import os


try:
    import deepspeed
    _HAS_DEEPSPEED = True
except ModuleNotFoundError:
    _HAS_DEEPSPEED = False


_GC_BRANCH_REPORTED = set()
_GC_DEBUG_ENABLED = os.environ.get("FLASHVSR_TRAIN_DEBUG", "").lower() in ("1", "true", "yes", "y")


def _write_debug_marker(filename: str, message: str) -> None:
    debug_dir = os.environ.get("FLASHVSR_DEBUG_DIR")
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, filename), "a", encoding="utf-8") as file:
        file.write(message + "\n")


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)
    return custom_forward


def _module_name(module):
    if isinstance(module, torch.nn.Module):
        return f"{type(module).__module__}.{type(module).__name__}"
    return getattr(module, "__qualname__", repr(module))


def create_custom_forward_use_reentrant(module):
    target_dtype = None
    if isinstance(module, torch.nn.Module):
        for parameter in module.parameters():
            if parameter.is_floating_point():
                target_dtype = parameter.dtype
                break
    reported_inputs = {"raw": False, "normalized": False}

    def custom_forward(*inputs):
        if not reported_inputs["raw"]:
            reported_inputs["raw"] = True
            if _GC_DEBUG_ENABLED:
                rank = os.environ.get("RANK", "?")
                local_rank = os.environ.get("LOCAL_RANK", "?")
                payload = []
                for idx, value in enumerate(inputs):
                    if isinstance(value, torch.Tensor):
                        payload.append(
                            f"arg{idx}:shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
                        )
                    else:
                        payload.append(f"arg{idx}:type={type(value).__name__}")
                message = (
                    f"[gradient_checkpoint_inputs] rank={rank} local_rank={local_rank} "
                    f"model={_module_name(module)} "
                    f"target_dtype={target_dtype} " + " | ".join(payload)
                )
                print(message, flush=True)
                _write_debug_marker("gradient_checkpoint_inputs.log", message)
        if target_dtype is None:
            return module(*inputs)
        normalized_inputs = []
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != target_dtype:
                value = value.to(dtype=target_dtype)
            normalized_inputs.append(value)
        if not reported_inputs["normalized"]:
            reported_inputs["normalized"] = True
            if _GC_DEBUG_ENABLED:
                payload = []
                for idx, value in enumerate(normalized_inputs):
                    if isinstance(value, torch.Tensor):
                        payload.append(
                            f"arg{idx}:shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
                        )
                    else:
                        payload.append(f"arg{idx}:type={type(value).__name__}")
                _write_debug_marker(
                    "gradient_checkpoint_inputs_normalized.log",
                    f"[gradient_checkpoint_inputs_normalized] model={_module_name(module)} "
                    + " | ".join(payload),
                )
        return module(*normalized_inputs)
    return custom_forward


def judge_args_requires_grad(*args):
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            return True
    return False


def gradient_checkpoint_forward(
    model,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
    *args,
    **kwargs,
):
    branch = None
    if use_gradient_checkpointing and _HAS_DEEPSPEED and deepspeed.checkpointing.is_configured():
        branch = "deepspeed_checkpoint"
        all_args = args + tuple(kwargs.values())
        if not judge_args_requires_grad(*all_args):
            # get the first grad_enabled tensor from un_checkpointed forward
            model_output = model(*args, **kwargs)
        else:
            model_output = deepspeed.checkpointing.checkpoint(
                create_custom_forward_use_reentrant(model),
                *all_args,
            )
        model_output = model_output
    elif use_gradient_checkpointing_offload:
        branch = "torch_checkpoint_offload"
        with torch.autograd.graph.save_on_cpu():
            model_output = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model),
                *args,
                **kwargs,
                use_reentrant=False,
            )
    elif use_gradient_checkpointing:
        branch = "torch_checkpoint"
        model_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    else:
        branch = "no_checkpoint"
        model_output = model(*args, **kwargs)
    report_key = (branch, type(model).__name__)
    if report_key not in _GC_BRANCH_REPORTED:
        _GC_BRANCH_REPORTED.add(report_key)
        if _GC_DEBUG_ENABLED:
            rank = os.environ.get("RANK", "?")
            local_rank = os.environ.get("LOCAL_RANK", "?")
            message = (
                f"[gradient_checkpoint] rank={rank} local_rank={local_rank} "
                f"branch={branch} model={_module_name(model)}"
            )
            print(message, flush=True)
            _write_debug_marker("gradient_checkpoint_branches.log", message)
    return model_output
