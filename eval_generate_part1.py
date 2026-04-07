import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import gc

# --- Configuration ---
BASE_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
STAGE1_ADAPTER = "./sft-lora-Phi-3.5-mini-instruct-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/final"
STAGE2_ADAPTER = "./sft-lora-stage2-json-r32-lr1.0e-04/final"

# Paths to your explicitly generated test sets
ALPACA_EVAL_PATH = "alpaca_eval_set.json" 
JSON_EVAL_PATH = "final_json_dataset_for_test.json" 

OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper: Generate Responses ---
def generate_responses(model, tokenizer, dataset, task_name, checkpoint_name):
    print(f"--- Generating {task_name} responses for {checkpoint_name} ---")
    results = []
    
    # Put model in eval mode
    model.eval()
    
    for i, row in enumerate(dataset):
        
        # --- Handle Different Data Structures ---
        if task_name == "alpaca":
            # Alpaca data is pre-formatted in a single "text" string
            full_text = row.get("text", "")
            
            # Split it at the response marker. 
            # Parts[0] is the instruction/input. Parts[1] is the target output.
            parts = full_text.split("### Response:\n")
            
            prompt = parts[0] + "### Response:\n"
            reference = parts[1].strip() if len(parts) > 1 else ""
            instruction = "Parsed from full text" # Placeholder since it's baked into the prompt
            input_text = ""
            
        elif task_name == "json":
            # JSON data has distinct keys
            instruction = row.get("instruction", "")
            input_text = row.get("input", "")
            reference = row.get("output", "")
            
            # Format the prompt exactly like the training setup
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
                temperature=0.1, # Low temp for deterministic evaluation
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Decode and strip out the prompt to get just the model's new response
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
    out_file = os.path.join(OUTPUT_DIR, f"{checkpoint_name}_{task_name}_responses.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved {out_file}\n")

# --- Main Pipeline ---
def main():
    print("Loading test datasets...")
    
    # Load the datasets
    alpaca_data = load_dataset("json", data_files=ALPACA_EVAL_PATH, split="train")
    json_data = load_dataset("json", data_files=JSON_EVAL_PATH, split="train")

    # --- SUBSET LOGIC ---
    # Select only the first 200 items of the Alpaca dataset (or the max length if it's smaller than 200)
    alpaca_limit = min(200, len(alpaca_data))
    alpaca_data = alpaca_data.select(range(alpaca_limit))
    
    print(f"Loaded {len(alpaca_data)} Alpaca items (subset) and {len(json_data)} JSON items.")

    # 1. Load Base Model & Tokenizer (Checkpoint 0)
    print("\nLoading Base Model (Checkpoint 0)...")
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

    # Generate Checkpoint 0
    generate_responses(base_model, tokenizer, alpaca_data, "alpaca", "ckpt0_base")
    generate_responses(base_model, tokenizer, json_data, "json", "ckpt0_base")

    # 2. Load Stage 1 Adapter (Checkpoint 1)
    print(f"\nLoading Stage 1 Adapter: {STAGE1_ADAPTER}...")
    model_stage1 = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER)
    
    # Generate Checkpoint 1
    generate_responses(model_stage1, tokenizer, alpaca_data, "alpaca", "ckpt1_stage1")
    generate_responses(model_stage1, tokenizer, json_data, "json", "ckpt1_stage1")

    # Free memory before loading the next adapter
    del model_stage1
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Load Stage 2 Adapter (Checkpoint 2)
    print(f"\nLoading Stage 2 Adapter: {STAGE2_ADAPTER}...")
    model_stage2 = PeftModel.from_pretrained(base_model, STAGE2_ADAPTER)
    
    # Generate Checkpoint 2
    generate_responses(model_stage2, tokenizer, alpaca_data, "alpaca", "ckpt2_stage2")
    generate_responses(model_stage2, tokenizer, json_data, "json", "ckpt2_stage2")

    print("\nAll generation complete! Check the 'eval_results' folder. Ready for Judge Evaluation.")

if __name__ == "__main__":
    main()