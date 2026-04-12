import json
import re
import os
from openai import OpenAI

# --- Configuration ---
UTSA_API_KEY = "gpustack_50e00c9281422bc5_0c0696dfcb1696d7635e58a2e56d6282"
UTSA_BASE_URL = "http://10.246.100.230/v1"
TEACHER_MODEL = "llama-3.3-70b-instruct-awq" 

# The 5 base filenames you generated
TASK_FILES = [
    "extraction.json",
    "json-repair.json",
    "label-classification.json",
    "schema-constrained.json",
    "tool-call.json"
]

def clean_and_parse_json(response_text: str):
    """Strips markdown formatting and attempts to parse the LLM output as JSON."""
    cleaned = re.sub(r"```json\s*", "", response_text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def process_dataset(input_dir: str, output_file: str, client: OpenAI):
    """Processes a directory of prompt files and saves the generated dataset."""
    final_dataset = []
    total_processed = 0
    total_failed = 0

    print(f"\n" + "="*40)
    print(f"STARTING PIPELINE: {input_dir.upper()}")
    print("="*40)

    for base_filename in TASK_FILES:
        filepath = os.path.join(input_dir, base_filename)
        
        if not os.path.exists(filepath):
            print(f"\n[!] Warning: Could not find {filepath}. Skipping.")
            continue
            
        print(f"\n--- Processing {filepath} ---")
        
        with open(filepath, "r") as f:
            tasks = json.load(f)
            
        for idx, task in enumerate(tasks):
            instruction = task.get("instruction", "")
            user_input = task.get("input", "")
            
            attempts = 0
            success = False
            
            # Retry up to 3 times if the LLM hallucinates bad JSON
            while not success and attempts < 3:
                attempts += 1
                try:
                    response = client.chat.completions.create(
                        model=TEACHER_MODEL,
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a strict data formatting AI. You must fulfill the user's instruction and output raw, perfectly valid JSON. Do NOT include markdown blocks like ```json. Do NOT output any conversational text."
                            },
                            {
                                "role": "user", 
                                "content": f"Instruction: {instruction}\n\nInput Data: {user_input}"
                            }
                        ],
                        max_tokens=1024,
                        temperature=0.1 # Very low temperature for strict precision
                    )
                    
                    response_text = response.choices[0].message.content
                    parsed_json = clean_and_parse_json(response_text)
                    
                    if parsed_json is not None:
                        print(f"  [{idx+1}/{len(tasks)}] Success on attempt {attempts}!")
                        
                        # Store in the standard Alpaca format for fine-tuning
                        final_dataset.append({
                            "instruction": instruction,
                            "input": user_input,
                            "output": json.dumps(parsed_json, indent=2) # Save as formatted string
                        })
                        success = True
                    else:
                        print(f"  [{idx+1}/{len(tasks)}] JSON parsing failed on attempt {attempts}. Retrying...")
                        
                except Exception as e:
                    print(f"  API Error on attempt {attempts}: {e}")
            
            if success:
                total_processed += 1
            else:
                total_failed += 1
                print(f"  [!] Failed to generate valid JSON for item {idx+1} after 3 attempts.")

    # Save the master dataset for this directory
    with open(output_file, "w") as f:
        json.dump(final_dataset, f, indent=4)
        
    print("\n" + "-"*40)
    print(f"DATASET COMPLETE: {output_file}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed items: {total_failed}")
    print("-"*40)

def main():
    print(f"Connecting to UTSA Server ({UTSA_BASE_URL})...")
    client = OpenAI(api_key=UTSA_API_KEY, base_url=UTSA_BASE_URL)

    # 1. Generate the Training Dataset
    process_dataset(
        input_dir="../../prompts/prompts-for-train", 
        output_file="../../data/final_json_dataset_for_train.json", 
        client=client
    )

    # 2. Generate the Evaluation/Test Dataset
    process_dataset(
        input_dir="../../prompts/prompts-for-eval", 
        output_file="../../data/final_json_dataset_for_test.json", 
        client=client
    )

if __name__ == "__main__":
    main()