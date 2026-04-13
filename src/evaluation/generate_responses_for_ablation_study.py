import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

BASE_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# Checkpoint Paths
STAGE1_ADAPTER = os.path.join(BASE_DIR, "checkpoints/sft-lora-Phi-3.5-mini-instruct-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/final")
STAGE2_ADAPTER = os.path.join(BASE_DIR, "checkpoints/sft-lora-stage2-json-r32-lr1.0e-04/final")

# New Ablation Checkpoint Paths
STAGE2_EPOCH1_ADAPTER = os.path.join(BASE_DIR, "checkpoints/sft-lora-stage2-json-r32-lr1.0e-04-epoch1/final")
STAGE2_EPOCH2_ADAPTER = os.path.join(BASE_DIR, "checkpoints/sft-lora-stage2-json-r32-lr1.0e-04-epoch2/final")

# Paths to your explicitly generated test sets
ALPACA_EVAL_PATH = os.path.join(BASE_DIR, "data/alpaca_eval_set.json")
JSON_EVAL_PATH = os.path.join(BASE_DIR, "data/final_json_dataset_for_test.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "eval_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper: Generate Responses ---
def generate_responses(model, tokenizer, dataset, task_name, checkpoint_name):
    out_file = os.path.join(OUTPUT_DIR, f"{checkpoint_name}_{task_name}_responses.json")
    
    # Check if we already generated this to save time
    if os.path.exists(out_file):
        print(f"[{checkpoint_name} - {task_name}] already exists. Skipping generation to save time.")
        return

    print(f"--- Generating {task_name} responses for {checkpoint_name} ---")
    results = []
    
    # Put model in eval mode
    model.eval()
    
    for i, row in enumerate(dataset):
        
        # --- Handle Different Data Structures ---
        if task_name == "alpaca":
            full_text = row.get("text", "")
            parts = full_text.split("### Response:\n")
            prompt = parts[0] + "### Response:\n"
            reference = parts[1].strip() if len(parts) > 1 else ""
            instruction = "Parsed from full text" 
            input_text = ""
            
        elif task_name == "json":
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            reference = row.get("output", "")
            
            if input_text and str(input_text).strip() != "":
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # --- Generation ---
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = full_output.split("### Response:\n")[-1].strip()
        
        results.append({
            "id": f"{task_name}_eval_{i}",
            "prompt_used": prompt,
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "generated_response": response_only
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} items...")

    # Save to disk
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {out_file}\n")

# --- Main Pipeline ---
def main():
    print("Loading test datasets...")
    
    alpaca_data = load_dataset("json", data_files=ALPACA_EVAL_PATH, split="train")
    json_data = load_dataset("json", data_files=JSON_EVAL_PATH, split="train")

    alpaca_limit = min(200, len(alpaca_data))
    alpaca_data = alpaca_data.select(range(alpaca_limit))
    
    print(f"Loaded {len(alpaca_data)} Alpaca items (subset) and {len(json_data)} JSON items.")

    print("\nLoading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, quantization_config=bnb_config, torch_dtype=torch.float16
    )

    # 1. Base Model
    generate_responses(base_model, tokenizer, alpaca_data, "alpaca", "ckpt0_base")
    generate_responses(base_model, tokenizer, json_data, "json", "ckpt0_base")

    # 2. Stage 1 
    print(f"\nLoading Stage 1 Adapter...")
    model_stage1 = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER)
    generate_responses(model_stage1, tokenizer, alpaca_data, "alpaca", "ckpt1_stage1")
    generate_responses(model_stage1, tokenizer, json_data, "json", "ckpt1_stage1")
    del model_stage1
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Stage 2 (Epoch 3 - Original)
    print(f"\nLoading Stage 2 Adapter (Original)...")
    model_stage2 = PeftModel.from_pretrained(base_model, STAGE2_ADAPTER)
    generate_responses(model_stage2, tokenizer, alpaca_data, "alpaca", "ckpt2_stage2")
    generate_responses(model_stage2, tokenizer, json_data, "json", "ckpt2_stage2")
    del model_stage2
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Stage 2 (Epoch 1 - Ablation)
    print(f"\nLoading Stage 2 Adapter (Epoch 1)...")
    model_stage2_ep1 = PeftModel.from_pretrained(base_model, STAGE2_EPOCH1_ADAPTER)
    generate_responses(model_stage2_ep1, tokenizer, alpaca_data, "alpaca", "ckpt2_stage2_epoch1")
    generate_responses(model_stage2_ep1, tokenizer, json_data, "json", "ckpt2_stage2_epoch1")
    del model_stage2_ep1
    torch.cuda.empty_cache()
    gc.collect()

    # 5. Stage 2 (Epoch 2 - Ablation)
    print(f"\nLoading Stage 2 Adapter (Epoch 2)...")
    model_stage2_ep2 = PeftModel.from_pretrained(base_model, STAGE2_EPOCH2_ADAPTER)
    generate_responses(model_stage2_ep2, tokenizer, alpaca_data, "alpaca", "ckpt2_stage2_epoch2")
    generate_responses(model_stage2_ep2, tokenizer, json_data, "json", "ckpt2_stage2_epoch2")
    del model_stage2_ep2
    torch.cuda.empty_cache()
    gc.collect()

    print("\nAll generation complete! Check the 'eval_results' folder. Ready for Judge Evaluation.")

if __name__ == "__main__":
    main()