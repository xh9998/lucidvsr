import json
import os
import random
import shutil
import time
import inspect

import torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import send_to_device
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger

try:
    import numpy as np
except Exception:
    np = None


_DATALOADER_SIGNATURE = inspect.signature(torch.utils.data.DataLoader)
_DATALOADER_SUPPORTS_IN_ORDER = "in_order" in _DATALOADER_SIGNATURE.parameters


def _first_item_collate(batch):
    return batch[0]


def _init_data_worker_no_cuda(worker_id: int):
    # DataLoader workers must stay CPU-only. Calling torch.cuda.is_available()
    # or torch.cuda.set_device() here creates one CUDA context per worker and
    # steals GPU memory from the training process.
    base_seed = torch.initial_seed() % (2 ** 32)
    random.seed(base_seed + worker_id)
    if np is not None:
        np.random.seed((base_seed + worker_id) % (2 ** 32))
    torch.manual_seed(base_seed + worker_id)


class _PreBatchedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: torch.utils.data.IterableDataset, batch_size: int, collate_fn):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.custom_collate_fn = None

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []


def _set_process_seed(seed: int, process_index: int):
    worker_seed = int(seed) + int(process_index)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed % (2 ** 32))
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def _training_state_dir(output_path: str, step: int) -> str:
    return os.path.join(output_path, "training_state", f"step-{int(step)}")


def _training_state_meta_path(state_dir: str) -> str:
    return os.path.join(state_dir, "flashvsr_training_state.json")


def _summarize_optimizer_state(optimizer) -> str:
    try:
        state_dict = optimizer.state_dict()
        state = state_dict.get("state", {})
        param_groups = state_dict.get("param_groups", [])
        return f"optimizer_state_entries={len(state)} param_groups={len(param_groups)}"
    except Exception as error:
        return f"optimizer_state_summary_unavailable={error}"


def save_training_state(
    accelerator: Accelerator,
    output_path: str,
    step: int,
    epoch_id: int,
    args=None,
):
    accelerator.wait_for_everyone()
    state_dir = _training_state_dir(output_path, step)
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(state_dir), exist_ok=True)
        if os.path.isdir(state_dir):
            shutil.rmtree(state_dir)
    accelerator.wait_for_everyone()
    accelerator.save_state(state_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = {
            "step": int(step),
            "epoch_id": int(epoch_id),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_seed": getattr(args, "global_seed", None) if args is not None else None,
        }
        with open(_training_state_meta_path(state_dir), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()
    return state_dir


def load_training_state(
    accelerator: Accelerator,
    state_dir: str,
    args=None,
):
    rank = int(getattr(accelerator, "process_index", 0))
    local_rank = int(getattr(accelerator, "local_process_index", -1))
    resume_debug_enabled = os.environ.get("FLASHVSR_RESUME_DEBUG", "").lower() in ("1", "true", "yes", "y")

    def resume_debug(message: str):
        if not resume_debug_enabled:
            return
        print(
            f"[resume_debug] rank={rank} local_rank={local_rank} {message}",
            flush=True,
        )

    def describe_path(path: str) -> str:
        try:
            stat = os.stat(path)
            return f"exists=1 size={stat.st_size} mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}"
        except FileNotFoundError:
            return "exists=0"
        except Exception as error:
            return f"stat_error={type(error).__name__}:{error}"

    if resume_debug_enabled:
        resume_debug(f"load_state_prepare state_dir={state_dir}")
        expected_paths = [
            os.path.join(state_dir, "flashvsr_training_state.json"),
            os.path.join(state_dir, "scheduler.bin"),
            os.path.join(state_dir, f"random_states_{rank}.pkl"),
            os.path.join(state_dir, "pytorch_model", "mp_rank_00_model_states.pt"),
            os.path.join(state_dir, "pytorch_model", f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"),
        ]
        for path in expected_paths:
            resume_debug(f"precheck path={path} {describe_path(path)}")

    original_torch_load = torch.load

    def debug_torch_load(*load_args, **load_kwargs):
        path_arg = load_args[0] if load_args else load_kwargs.get("f", None)
        path_text = os.fspath(path_arg) if isinstance(path_arg, (str, os.PathLike)) else str(path_arg)
        should_log = (
            resume_debug_enabled
            and isinstance(path_arg, (str, os.PathLike))
            and os.path.abspath(path_text).startswith(os.path.abspath(state_dir))
        )
        if should_log:
            resume_debug(f"torch_load_begin path={path_text} {describe_path(path_text)}")
        try:
            result = original_torch_load(*load_args, **load_kwargs)
        except Exception as error:
            if should_log:
                resume_debug(f"torch_load_error path={path_text} error={type(error).__name__}:{error}")
            raise
        if should_log:
            resume_debug(f"torch_load_end path={path_text}")
        return result

    if resume_debug_enabled:
        torch.load = debug_torch_load
    try:
        resume_debug("accelerator_load_state_begin")
        accelerator.load_state(state_dir)
        resume_debug("accelerator_load_state_end")
    finally:
        if resume_debug_enabled:
            torch.load = original_torch_load
    metadata = {}
    meta_path = _training_state_meta_path(state_dir)
    if accelerator.is_main_process and os.path.exists(meta_path):
        resume_debug(f"metadata_read_begin path={meta_path} {describe_path(meta_path)}")
        with open(meta_path, "r", encoding="utf-8") as file:
            metadata = json.load(file) or {}
        resume_debug(f"metadata_read_end metadata={metadata}")
    gathered = [metadata]
    try:
        from accelerate.utils import broadcast_object_list

        resume_debug("metadata_broadcast_begin")
        broadcast_object_list(gathered, from_process=0)
        metadata = gathered[0] or {}
        resume_debug(f"metadata_broadcast_end metadata={metadata}")
    except Exception:
        resume_debug("metadata_broadcast_failed")
        pass

    if args is not None and getattr(args, "resume_reset_rng_with_global_seed", False) and getattr(args, "global_seed", None) is not None:
        _set_process_seed(args.global_seed, accelerator.process_index)

    step = int(metadata.get("step", 0))
    epoch_id = int(metadata.get("epoch_id", 0))
    return {
        "step": step,
        "epoch_id": epoch_id,
        "metadata": metadata,
    }


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    debug_enabled = os.environ.get("FLASHVSR_TRAIN_DEBUG", "").lower() in ("1", "true", "yes", "y")

    def debug_log(message: str):
        if not debug_enabled:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[debug {timestamp} rank={rank} local_rank={local_rank}] {message}", flush=True)

    max_train_steps = None
    log_loss_steps = 1
    extra_save_steps = set()
    if args is not None:
        learning_rate = args.learning_rate
        batch_size = getattr(args, "batch_size", 1)
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        max_train_steps = getattr(args, "max_train_steps", None)
        log_loss_steps = getattr(args, "log_loss_steps", 1)
        extra_save_steps_raw = getattr(args, "extra_save_steps", "") or ""
        if extra_save_steps_raw:
            extra_save_steps = {
                int(step.strip())
                for step in extra_save_steps_raw.split(",")
                if step.strip()
            }
    else:
        batch_size = 1
    wandb_run = None
    if args is not None and getattr(args, "use_wandb", False) and accelerator.is_main_process:
        try:
            import wandb
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "flashvsr"),
                name=getattr(args, "wandb_name", None),
                entity=getattr(args, "wandb_entity", None),
                mode=getattr(args, "wandb_mode", "online"),
                config=vars(args),
            )
        except Exception as error:
            print(f"[wandb] init failed: {error}", flush=True)
    
    debug_log(f"launch_training_task start distributed_type={accelerator.distributed_type} device={accelerator.device}")
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    is_iterable_dataset = isinstance(dataset, torch.utils.data.IterableDataset)
    collate_fn = getattr(dataset, "custom_collate_fn", None)
    if collate_fn is None:
        collate_fn = _first_item_collate
    dataloader_dataset = dataset
    dataloader_batch_size = batch_size
    dataloader_collate_fn = collate_fn
    if is_iterable_dataset and batch_size > 1:
        dataloader_dataset = _PreBatchedIterableDataset(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_batch_size = 1
        dataloader_collate_fn = _first_item_collate
    dataloader_kwargs = {
        "batch_size": dataloader_batch_size,
        "shuffle": not is_iterable_dataset,
        "collate_fn": dataloader_collate_fn,
        "num_workers": num_workers,
    }
    if args is not None:
        dataloader_kwargs["pin_memory"] = bool(getattr(args, "dataloader_pin_memory", False))
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = max(1, int(getattr(args, "dataloader_prefetch_factor", 2)))
            dataloader_kwargs["persistent_workers"] = bool(getattr(args, "dataloader_persistent_workers", False))
            multiprocessing_context = getattr(args, "dataloader_multiprocessing_context", None)
            if multiprocessing_context:
                dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
            dataloader_kwargs["worker_init_fn"] = _init_data_worker_no_cuda
            if _DATALOADER_SUPPORTS_IN_ORDER:
                dataloader_kwargs["in_order"] = bool(getattr(args, "dataloader_in_order", True))
    debug_log(
        "building dataloader "
        f"iterable={is_iterable_dataset} "
        f"num_workers={num_workers} "
        f"prefetch_factor={dataloader_kwargs.get('prefetch_factor', None)} "
        f"persistent_workers={dataloader_kwargs.get('persistent_workers', False)} "
        f"multiprocessing_context={dataloader_kwargs.get('multiprocessing_context', None)} "
        f"pin_memory={dataloader_kwargs.get('pin_memory', False)} "
        f"in_order={dataloader_kwargs.get('in_order', None)} "
        f"collate_fn={getattr(collate_fn, '__name__', type(collate_fn).__name__)}"
    )
    dataloader = torch.utils.data.DataLoader(dataloader_dataset, **dataloader_kwargs)
    debug_log(f"dataloader constructed type={type(dataloader).__module__}.{type(dataloader).__name__}")
    model.to(device=accelerator.device)
    if is_iterable_dataset:
        debug_log("calling accelerator.prepare without dataloader for iterable dataset")
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        debug_log(
            "accelerator.prepare finished "
            f"dataloader_type={type(dataloader).__module__}.{type(dataloader).__name__}"
        )
    else:
        debug_log("calling accelerator.prepare")
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        debug_log(
            "accelerator.prepare finished "
            f"dataloader_type={type(dataloader).__module__}.{type(dataloader).__name__}"
        )
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        engine_class = type(model).__module__ + "." + type(model).__name__
        debug_log(f"prepared model engine={engine_class}")
        ds_config_runtime = getattr(model, "config", None)
        if ds_config_runtime is not None:
            zero_stage = None
            offload_optimizer_device = None
            offload_param_device = None
            activation_checkpointing = None
            try:
                zero_stage = ds_config_runtime.zero_optimization_stage()
            except Exception:
                pass
            try:
                offload_optimizer_device = ds_config_runtime.zero_config.offload_optimizer.device
            except Exception:
                pass
            try:
                offload_param_device = ds_config_runtime.zero_config.offload_param.device
            except Exception:
                pass
            try:
                activation_checkpointing = ds_config_runtime.activation_checkpointing_config
            except Exception:
                pass
            debug_log(
                "deepspeed runtime "
                f"zero_stage={zero_stage} "
                f"offload_optimizer_device={offload_optimizer_device} "
                f"offload_param_device={offload_param_device} "
                f"activation_checkpointing={activation_checkpointing}"
            )
    initialize_deepspeed_gradient_checkpointing(accelerator)
    debug_log("initialize_deepspeed_gradient_checkpointing finished")
    resumed_epoch_id = 0
    global_step = 0
    resume_training_state_dir = getattr(args, "resume_training_state_dir", None) if args is not None else None
    if resume_training_state_dir:
        debug_log(f"loading training state from {resume_training_state_dir}")
        resume_state = load_training_state(accelerator, resume_training_state_dir, args=args)
        global_step = int(resume_state["step"])
        resumed_epoch_id = int(resume_state["epoch_id"])
        model_logger.set_num_steps(global_step)
        resume_message = (
            "[resume] training state loaded "
            f"step={global_step} epoch_id={resumed_epoch_id} "
            f"reseed={getattr(args, 'resume_reset_rng_with_global_seed', False)} "
            f"lr={scheduler.get_last_lr()} "
            f"{_summarize_optimizer_state(optimizer)}"
        )
        print(resume_message, flush=True)
        debug_log(resume_message)
    if args is not None and getattr(args, "validation_at_start", False):
        if accelerator.is_main_process:
            if model_logger.validation_callback is not None:
                debug_log("validation_at_start begin")
                model_logger.validation_callback(
                    accelerator=accelerator,
                    model=accelerator.unwrap_model(model),
                    checkpoint_path=os.path.join(model_logger.output_path, "step-0-pretrain"),
                    step=0,
                )
                debug_log("validation_at_start finished")
            else:
                debug_log("validation_at_start skipped: validation_callback is None on main process")
        accelerator.wait_for_everyone()
        debug_log("validation_at_start barrier finished")
    for epoch_id in range(resumed_epoch_id, num_epochs):
        debug_log(f"epoch {epoch_id} start")
        progress_bar = tqdm(dataloader)
        data_iterator = iter(progress_bar)
        while True:
            debug_log(f"before fetch batch global_step={global_step + 1}")
            try:
                data = next(data_iterator)
            except StopIteration:
                debug_log(f"epoch {epoch_id} dataloader exhausted")
                break
            if is_iterable_dataset:
                data = send_to_device(data, accelerator.device)
            debug_log(f"after fetch batch global_step={global_step + 1}")
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                debug_log(f"before forward global_step={global_step + 1}")
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                debug_log(f"after forward global_step={global_step + 1}")
                accelerator.backward(loss)
                debug_log(f"after backward global_step={global_step + 1}")
                optimizer.step()
                saved_checkpoint_path = model_logger.on_step_end(accelerator, model, save_steps, extra_save_steps=extra_save_steps, loss=loss)
                if saved_checkpoint_path is not None:
                    save_training_state(
                        accelerator=accelerator,
                        output_path=model_logger.output_path,
                        step=model_logger.num_steps,
                        epoch_id=epoch_id,
                        args=args,
                    )
                debug_log(f"after logger global_step={global_step + 1}")
                scheduler.step()
                global_step += 1
                loss_value = float(accelerator.gather(loss.detach()).mean().item())
                if accelerator.is_main_process:
                    progress_bar.set_postfix(loss=f"{loss_value:.6f}", step=global_step)
                    if log_loss_steps is not None and log_loss_steps > 0 and global_step % log_loss_steps == 0:
                        print(f"[train] epoch={epoch_id} step={global_step} loss={loss_value:.6f}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": loss_value,
                                "train/step": global_step,
                                "train/epoch": epoch_id,
                                "train/lr": float(scheduler.get_last_lr()[0]),
                            },
                            step=global_step,
                        )
                if max_train_steps is not None and global_step >= max_train_steps:
                    break
        if save_steps is None:
            saved_checkpoint_path = model_logger.on_epoch_end(accelerator, model, epoch_id)
            if saved_checkpoint_path is not None:
                save_training_state(
                    accelerator=accelerator,
                    output_path=model_logger.output_path,
                    step=model_logger.num_steps,
                    epoch_id=epoch_id,
                    args=args,
                )
        if max_train_steps is not None and global_step >= max_train_steps:
            break
    saved_checkpoint_path = model_logger.on_training_end(accelerator, model, save_steps, extra_save_steps=extra_save_steps)
    if saved_checkpoint_path is not None:
        save_training_state(
            accelerator=accelerator,
            output_path=model_logger.output_path,
            step=model_logger.num_steps,
            epoch_id=epoch_id if num_epochs > 0 else 0,
            args=args,
        )
    if wandb_run is not None:
        wandb_run.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader_kwargs = {
        "shuffle": False,
        "collate_fn": _first_item_collate,
        "num_workers": num_workers,
    }
    if args is not None:
        dataloader_kwargs["pin_memory"] = bool(getattr(args, "dataloader_pin_memory", False))
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = max(1, int(getattr(args, "dataloader_prefetch_factor", 2)))
            dataloader_kwargs["persistent_workers"] = bool(getattr(args, "dataloader_persistent_workers", False))
            multiprocessing_context = getattr(args, "dataloader_multiprocessing_context", None)
            if multiprocessing_context:
                dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
            dataloader_kwargs["worker_init_fn"] = _init_data_worker_no_cuda
            if _DATALOADER_SUPPORTS_IN_ORDER:
                dataloader_kwargs["in_order"] = bool(getattr(args, "dataloader_in_order", True))
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        zero_opt = ds_config.get("zero_optimization", {}) if isinstance(ds_config.get("zero_optimization", {}), dict) else {}
        offload_optimizer = zero_opt.get("offload_optimizer", {}) if isinstance(zero_opt.get("offload_optimizer", {}), dict) else {}
        offload_param = zero_opt.get("offload_param", {}) if isinstance(zero_opt.get("offload_param", {}), dict) else {}
        act_config = ds_config.get("activation_checkpointing", None)
        zero_stage = zero_opt.get("stage", ds_config.get("zero_stage", None))
        print(
            "[deepspeed] config summary "
            f"zero_stage={zero_stage} "
            f"offload_optimizer_device={offload_optimizer.get('device', ds_config.get('offload_optimizer_device', None))} "
            f"offload_param_device={offload_param.get('device', ds_config.get('offload_param_device', None))} "
            f"has_activation_checkpointing={isinstance(act_config, dict)}",
            flush=True,
        )
        if isinstance(act_config, dict):
            import deepspeed
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
