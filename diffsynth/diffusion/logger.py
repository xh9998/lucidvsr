import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, validation_callback=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.validation_callback = validation_callback
        self.num_steps = 0


    def set_num_steps(self, num_steps: int):
        self.num_steps = int(num_steps)


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, extra_save_steps=None, **kwargs):
        self.num_steps += 1
        should_save = False
        if save_steps is not None and self.num_steps % save_steps == 0:
            should_save = True
        if extra_save_steps is not None and self.num_steps in extra_save_steps:
            should_save = True
        if should_save:
            return self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        return None


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            return path
        return None


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, extra_save_steps=None):
        should_save = False
        if save_steps is not None and self.num_steps % save_steps != 0:
            should_save = True
        if extra_save_steps is not None and self.num_steps in extra_save_steps:
            should_save = False
        if should_save:
            return self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        return None


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        path = os.path.join(self.output_path, file_name)
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            accelerator.save(state_dict, path, safe_serialization=True)
            if self.validation_callback is not None:
                self.validation_callback(
                    accelerator=accelerator,
                    model=accelerator.unwrap_model(model),
                    checkpoint_path=path,
                    step=self.num_steps,
                )
        accelerator.wait_for_everyone()
        return path
