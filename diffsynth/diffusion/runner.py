import os, time, torch
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import send_to_device
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


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
        collate_fn = lambda x: x[0]
    dataloader_dataset = dataset
    dataloader_batch_size = batch_size
    dataloader_collate_fn = collate_fn
    if is_iterable_dataset and batch_size > 1:
        dataloader_dataset = _PreBatchedIterableDataset(dataset, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_batch_size = 1
        dataloader_collate_fn = lambda x: x[0]
    debug_log(f"building dataloader iterable={is_iterable_dataset} num_workers={num_workers} collate_fn={getattr(collate_fn, '__name__', type(collate_fn).__name__)}")
    dataloader = torch.utils.data.DataLoader(
        dataloader_dataset,
        batch_size=dataloader_batch_size,
        shuffle=not is_iterable_dataset,
        collate_fn=dataloader_collate_fn,
        num_workers=num_workers,
    )
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
    global_step = 0
    for epoch_id in range(num_epochs):
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
                model_logger.on_step_end(accelerator, model, save_steps, extra_save_steps=extra_save_steps, loss=loss)
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
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        if max_train_steps is not None and global_step >= max_train_steps:
            break
    model_logger.on_training_end(accelerator, model, save_steps, extra_save_steps=extra_save_steps)
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
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
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
