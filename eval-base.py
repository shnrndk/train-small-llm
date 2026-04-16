import torch
import dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

dotenv.load_dotenv()

# ---------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.2-3B"

PROMPT = "What is the capital of France?"

MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_P          = 0.9
# ---------------------------------------------------------


def load_model(base_model: str):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    print("Model ready.\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
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

    # Decode only the newly generated tokens (strip the input prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    model, tokenizer = load_model(BASE_MODEL)

    print("=" * 60)
    print(f"Prompt: {PROMPT}")
    print("=" * 60)

    response = generate(model, tokenizer, PROMPT)
    print(f"Response:\n{response}")
    print("=" * 60)

    # ---------------------------------------------------------
    # Interactive loop — keep prompting until the user quits
    # ---------------------------------------------------------
    print("\nEntering interactive mode. Type 'quit' to exit.\n")
    while True:
        prompt = input("Prompt: ").strip()
        
        # 1. Check for quit commands
        if prompt.lower() in ("quit", "exit", "q"):
            break
            
        # 2. Add this check to prevent crashing on empty inputs!
        if not prompt:
            print("Please enter a valid prompt.")
            continue

        response = generate(model, tokenizer, prompt)
        print(f"\nResponse:\n{response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()