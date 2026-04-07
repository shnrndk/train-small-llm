import os
import dotenv
import torch

import config 
from datasets import load_dataset

from peft import PeftModel
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig 
)

# ---------------------------------------------------------
# 0. Environment & Paths
# ---------------------------------------------------------
dotenv.load_dotenv()

# The 'final' folder saved at the end of your Stage 1 training
STAGE1_ADAPTER_PATH = "./sft-lora-Phi-3.5-mini-instruct-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/final" 
JSON_DATASET_PATH = "final_json_dataset_for_train.json"

# ---------------------------------------------------------
#  Training
# ---------------------------------------------------------
def main():
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    # 1. Config
    model_args, data_args, lora_args, train_args = config.get_config_classes(training_type="sft")

    # 2. Load, Format, and Split the JSON Dataset
    print(f"Loading local JSON dataset from: {JSON_DATASET_PATH}")
    dataset = load_dataset("json", data_files=JSON_DATASET_PATH, split="train")
    
    # Apply formatting BEFORE passing to the trainer to resolve completion_only_loss conflict
    def format_dataset_row(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        response = example.get('output', '')
        
        if input_text and str(input_text).strip() != "":
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(format_dataset_row)
    
    # Split 90/10 for train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=data_args.dataset_ops_seed)
    train_set = dataset["train"]
    val_set = dataset["test"]

    # Set up Stage 2 output directory
    output_dir = (
        f"./sft-lora-stage2-json"
        f"-r{lora_args.r}"
        f"-lr{train_args.learning_rate:.1e}"
    )
    train_args.output_dir = output_dir
    train_args.run_name = output_dir
    print("Stage 2 Output directory:", train_args.output_dir)

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Train dataset size:      {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")

    # 4. 4-bit Base Model & Stage 1 Adapter Loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    
    print(f"Loading Stage 1 Adapter from {STAGE1_ADAPTER_PATH}...")
    # Wrap the base model with the Stage 1 adapter and set is_trainable=True
    model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER_PATH, is_trainable=True)

    # 5. Train
    trainer = SFTTrainer(
        model=model, 
        args=train_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        processing_class=tokenizer
    )

    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    # Save the Stage 2 adapter
    trainer.save_model(os.path.join(train_args.output_dir, "final")) 
    print("Stage 2 Training complete. Updated LoRA adapter saved to:", train_args.output_dir)


if __name__ == "__main__":
    main()