import os
import torch
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import data_utils as du

dotenv.load_dotenv()

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_MODEL   = "meta-llama/Llama-3.2-3B"
# Pointing to the 'final' folder saved at the end of your training script
ADAPTER_PATH = "./sft-lora-Llama-3.2-3B-alpaca-r32-a64-d0.05-lr1.0e-04-wd0.01/final" 

INSTRUCTION  = "What is the capital of France?"
INPUT        = ""  

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_P          = 0.9
# ---------------------------------------------------------

def load_model(base_model: str, adapter_path: str):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in 4-bit to match training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        # device_map="auto",
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("Model ready.\n")
    return model, tokenizer

def generate(model, tokenizer, instruction: str, input_text: str = "") -> str:
    sample = {"instruction": instruction, "input": input_text}
    prompt = du.alpaca_row_to_prompt_eval(sample)["text"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    model, tokenizer = load_model(BASE_MODEL, ADAPTER_PATH)

    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    while True:
        instruction = input("Instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            break
        if not instruction:
            continue
        input_text = input("Input (leave blank if none): ").strip()

        response = generate(model, tokenizer, instruction, input_text)
        print(f"\nResponse:\n{response}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()