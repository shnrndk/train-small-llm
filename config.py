import os
import dotenv
dotenv.load_dotenv()
from dataclasses import dataclass, field
from typing import Tuple, cast, Literal
from transformers import HfArgumentParser
from trl import SFTConfig, DPOConfig
from peft import TaskType



# ---------------------------------------------------------
# 1. Hyperparameters & configuration
# ---------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.2-3B",
        metadata={"help": "Base model to fine-tune"}
    )
    

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "Dataset to use for training"}
    )
    local_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to a local dataset to be used instead of the one from the hub"}
    )
    dataset_ops_seed: int = field(
        default=42,
        metadata={"help": "Seed for data operations"}
    )
    validation_size: float = field(
        default=0.1,
        metadata={"help": "Fraction of the dataset to use for validation"}
    )



@dataclass
class LoRAArguments:
    r: int = field(
        default=32,
        metadata={"help": "Rank for LoRA"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "Alpha for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout rate for LoRA"}
    )
    target_modules: str = field(
        default='all-linear',
        metadata={"help": "Target modules for LoRA, e.g., 'all-linear'"}
    )
    task_type: TaskType = field(
        default=TaskType.CAUSAL_LM,
        metadata={"help": "Task type for LoRA, e.g., CAUSAL_LM or SEQ_2_SEQ"}
    )



SFTConfig_metadata = {k: v.metadata for k, v in SFTConfig.__dataclass_fields__.items()}
@dataclass
class TrainArguments(SFTConfig):
    # logging and output parameters
    output_dir: str = field(
        default="./lora-model",
        metadata=SFTConfig_metadata['output_dir']
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory"}
    )
    save_steps: int = field(
        default=50,
        metadata=SFTConfig_metadata['save_steps']
    )
    logging_steps: int = field(
        default=5,
        metadata=SFTConfig_metadata['logging_steps']
    )
    report_to: str = field(
        default="wandb",
        metadata=SFTConfig_metadata['report_to']
    )
    logging_first_step: bool = field(
        default=True,
        metadata=SFTConfig_metadata['logging_first_step']
    )
    

    # training parameters
    num_train_epochs: int = field(
        default=1,
        metadata=SFTConfig_metadata['num_train_epochs']
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata=SFTConfig_metadata['per_device_train_batch_size']
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata=SFTConfig_metadata['gradient_accumulation_steps']
    )
    warmup_ratio: float = field(
        default=0.3,
        metadata=SFTConfig_metadata['warmup_ratio']
    )
    learning_rate: float = field(
        default=1e-4,
        metadata=SFTConfig_metadata['learning_rate']
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata=SFTConfig_metadata['lr_scheduler_type']
    )

    bf16: bool = field(
        default=False,
        metadata=SFTConfig_metadata['bf16']
    )
    fp16: bool = field(
        default=True,
        metadata=SFTConfig_metadata['fp16']
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata=SFTConfig_metadata['gradient_checkpointing']
    )
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {'use_reentrant': False},
        metadata=SFTConfig_metadata['gradient_checkpointing_kwargs']
    )
    max_grad_norm: float = field(
        default=0.3, #REMINDER: also set in accelerate config
        metadata=SFTConfig_metadata['max_grad_norm']
    )
    adam_beta2: float = field(
        default=0.95,
        metadata=SFTConfig_metadata['adam_beta2']
    )
    weight_decay: float = field(
        default=0.01,
        metadata=SFTConfig_metadata['weight_decay']
    )
    

    #data parameters
    max_length: int = field(
        default=1024,
        metadata=SFTConfig_metadata['max_length']
    )
    packing: bool = field(
        default=False,
        metadata=SFTConfig_metadata['packing']
    )
    completion_only_loss: bool = field(
        default=True,
        metadata=SFTConfig_metadata['completion_only_loss']
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata=SFTConfig_metadata['assistant_only_loss']
    )
    group_by_length: bool = field(
        default=True,
        metadata={"help": "Group sequences of similar length together to speed up training"}
    )
    dataloader_drop_last: bool = field(
        default=False,
        metadata=SFTConfig_metadata['dataloader_drop_last']
    )


    # eval parameters
    do_eval: bool = field(
        default=True,
        metadata=SFTConfig_metadata['do_eval']
    )
    eval_strategy: str = field(
        default="steps",
        metadata=SFTConfig_metadata['eval_strategy']
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata=SFTConfig_metadata['per_device_eval_batch_size']
    )
    eval_on_start: bool = field(
        default=True,
        metadata=SFTConfig_metadata['eval_on_start']
    )
    eval_steps: int = field(
        default=50,
        metadata=SFTConfig_metadata['eval_steps']
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata=SFTConfig_metadata['load_best_model_at_end']
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata=SFTConfig_metadata['metric_for_best_model']
    )
    greater_is_better: bool = field(
        default=False,
        metadata=SFTConfig_metadata['greater_is_better']
    )

    

def get_config_classes(training_type: str = "sft"):
    if training_type == "sft":
        parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainArguments))
        model_args, data_args, lora_args, train_args = cast(
            Tuple[ModelArguments, DataArguments, LoRAArguments, TrainArguments],
            parser.parse_args_into_dataclasses()
        )
        return model_args, data_args, lora_args, train_args
    # elif training_type == "dpo":
    #     parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, DPOTrainArguments))
    #     model_args, data_args, lora_args, dpo_train_args = cast(
    #         Tuple[ModelArguments, DataArguments, LoRAArguments, DPOTrainArguments],
    #         parser.parse_args_into_dataclasses()
    #     )
    #     return model_args, data_args, lora_args, dpo_train_args
    else:
        raise ValueError(f"Unsupported training type: {training_type}. Use 'sft' or 'dpo'.")
