import os
import json
import random
import re
import evaluate
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Initialize the client using the key from the environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = "gpt-5-mini" 

RESULTS_DIR = "eval_results"
JUDGE_OUT_DIR = "judge_results"
os.makedirs(JUDGE_OUT_DIR, exist_ok=True)

# Load Evaluation Metrics
rouge = evaluate.load('rouge')
bertscore = evaluate.load('bertscore')

# --- Helper: Parse JSON for Validity ---
def clean_and_parse_json(text):
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except:
        return None

# --- Helper: JSON Granular Metrics (Schema & F1) ---
def calculate_json_metrics(parsed_pred, parsed_ref):
    exact = (parsed_pred == parsed_ref)
    
    # If it's not a dict, we can't do field-level comparison
    if not isinstance(parsed_pred, dict) or not isinstance(parsed_ref, dict):
        return exact, False, 0.0

    # Schema Compliance: Do the keys match exactly?
    schema_compliant = (set(parsed_pred.keys()) == set(parsed_ref.keys()))
    
    # Field-level F1 (Precision & Recall on keys and values)
    tp, fp, fn = 0, 0, 0
    for k, v in parsed_ref.items():
        if k in parsed_pred and parsed_pred[k] == v:
            tp += 1
        else:
            fn += 1
            
    for k in parsed_pred.keys():
        if k not in parsed_ref or parsed_pred[k] != parsed_ref.get(k):
            fp += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return exact, schema_compliant, f1

# --- Helper: The LLM Judge API Call ---
def query_judge(client, prompt_text, resp_a, resp_b, id_a, id_b, prompt_id):
    system_prompt = """You are an impartial expert evaluator of large language models. You will be given a user prompt and two responses (A and B).
    
Proper response should be concise, short, accurate, and directly address the user's prompt without unnecessary fluff. 

CRITICAL PENALTIES: Severely penalize a response (score 1 for conciseness and instruction_following) if it exhibits:
- Self-Prompting: Leaking template tags like `### Instruction:` or continuing to generate fake inputs.
- Degenerate Loops: Repeating the same phrases or list structures endlessly.
- Ignoring Context: Failing to use the specific data provided in the prompt.

Evaluate both responses on a scale of 1-5 for: 
- instruction_following: How well does it follow the core directive?
- correctness: Is the information factually accurate?
- clarity: Is the language clear and easy to understand?
- completeness: Does it answer all parts of the prompt?
- conciseness: 5 means perfectly concise without missing details; 1 means overly verbose, rambling, or contains irrelevant fluff.
- structured_output_validity: Rate high if formatting is perfect JSON if requested, otherwise N/A or 5.
- hallucination_risk: 5 means NO hallucinations, 1 means high hallucination.

You MUST output strictly in this JSON format:
{
  "prompt_id": "<prompt_id>",
  "checkpoint_a": "<checkpoint_a_name>",
  "checkpoint_b": "<checkpoint_b_name>",
  "response_a_scores": {"instruction_following": 0, "correctness": 0, "clarity": 0, "completeness": 0, "conciseness": 0, "structured_output_validity": 0, "hallucination_risk": 0},
  "response_b_scores": {"instruction_following": 0, "correctness": 0, "clarity": 0, "completeness": 0, "conciseness": 0, "structured_output_validity": 0, "hallucination_risk": 0},
  "winner": "<A, B, or Tie>",
  "justification": "<Brief reason highlighting why the winner is better, specifically mentioning length/conciseness if applicable>"
}"""

    user_message = f"Prompt ID: {prompt_id}\n\nUSER PROMPT:\n{prompt_text}\n\n---\nRESPONSE A:\n{resp_a}\n\n---\nRESPONSE B:\n{resp_b}"

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={ "type": "json_object" } # This ensures OpenAI returns perfect JSON
            # max_tokens=2048,
            # temperature=0.1
        )
        return clean_and_parse_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Judge API Error: {e}")
        return None

# --- Pipeline 1: Automatic Metrics (JSON + Text) ---
def run_automatic_metrics(task_name, checkpoints):
    print(f"\n" + "="*50)
    print(f"=== Calculating Auto Metrics for {task_name.upper()} ===")
    print("="*50)
    
    for ckpt in checkpoints:
        file_path = os.path.join(RESULTS_DIR, f"{ckpt}_{task_name}_responses.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            
        preds = [str(d["generated_response"]) for d in data]
        refs = [str(d["reference"]) for d in data]
        
        # 1. ROUGE & BERTScore
        rouge_res = rouge.compute(predictions=preds, references=refs)
        bert_res = bertscore.compute(predictions=preds, references=refs, lang="en")
        avg_bert = sum(bert_res['f1']) / len(bert_res['f1'])
        
        # 2. Output Length & Completion
        avg_len = sum(len(p.split()) for p in preds) / len(preds)
        completion_rate = sum(1 for p in preds if len(p.strip()) > 5) / len(preds)
        
        # 3. JSON Specific Metrics
        valid_json_count = 0
        exact_match_count = 0
        schema_match_count = 0
        total_f1 = 0.0
        
        if task_name == "json":
            for p, r in zip(preds, refs):
                parsed_p = clean_and_parse_json(p)
                parsed_r = clean_and_parse_json(r)
                
                if parsed_p is not None:
                    valid_json_count += 1
                    if parsed_r is not None:
                        exact, schema, f1 = calculate_json_metrics(parsed_p, parsed_r)
                        if exact: exact_match_count += 1
                        if schema: schema_match_count += 1
                        total_f1 += f1
                        
        print(f"\n--- Checkpoint: {ckpt} ---")
        print(f"  ROUGE-1: {rouge_res['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_res['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_res['rougeL']:.4f}")
        print(f"  BERTScore F1: {avg_bert:.4f}")
        print(f"  Avg Tokens: {avg_len:.1f} | Completion Rate: {completion_rate:.1%}")
        
        if task_name == "json":
            validity_rate = valid_json_count / len(preds)
            print(f"  JSON Validity Rate: {validity_rate:.1%}")
            # Protect against divide-by-zero if nothing was valid
            if valid_json_count > 0:
                print(f"  Schema Compliance (of valid JSON): {schema_match_count / valid_json_count:.1%}")
                print(f"  Exact Match (of valid JSON): {exact_match_count / valid_json_count:.1%}")
                print(f"  Field-level F1 (of valid JSON): {total_f1 / valid_json_count:.4f}")
            else:
                print("  Schema Compliance / Exact Match: 0.0% (No valid JSON generated)")

# --- Pipeline 2: Pairwise LLM Judge ---
def run_judge_eval(client, file_1, file_2, name_1, name_2, output_filename):
    print(f"\n=== Running Judge: {name_1} vs {name_2} ===")
    
    with open(file_1, "r") as f: data_1 = json.load(f)
    with open(file_2, "r") as f: data_2 = json.load(f)
        
    results = []
    win_1, win_2, ties = 0, 0, 0
    
    for i, (item_1, item_2) in enumerate(zip(data_1, data_2)):
        prompt_id = item_1["id"]
        prompt_text = item_1["prompt_used"]
        resp_1 = item_1["generated_response"]
        resp_2 = item_2["generated_response"]
        
        # Randomize order to prevent position bias
        flipped = random.choice([True, False])
        if flipped:
            resp_a, resp_b = resp_2, resp_1
            id_a, id_b = name_2, name_1
        else:
            resp_a, resp_b = resp_1, resp_2
            id_a, id_b = name_1, name_2
            
        judge_out = query_judge(client, prompt_text, resp_a, resp_b, id_a, id_b, prompt_id)
        
        if judge_out:
            judge_out["checkpoint_a"] = id_a
            judge_out["checkpoint_b"] = id_b
            
            # Map winner back to actual model
            actual_winner = "Tie"
            if judge_out.get("winner") == "A":
                actual_winner = name_2 if flipped else name_1
            elif judge_out.get("winner") == "B":
                actual_winner = name_1 if flipped else name_2
                
            judge_out["actual_winner"] = actual_winner
            results.append(judge_out)
            
            if actual_winner == name_1: win_1 += 1
            elif actual_winner == name_2: win_2 += 1
            else: ties += 1
            
        print(f"[{i+1}/{len(data_1)}] Processed {prompt_id} | Winner: {actual_winner}")
        
    print(f"\n--- Final Score ---")
    print(f"{name_1} Wins: {win_1}")
    print(f"{name_2} Wins: {win_2}")
    print(f"Ties: {ties}")
    
    # Prevent divide by zero if something fails
    total_matches = win_1 + win_2 + ties
    if total_matches > 0:
        print(f"Win Rate for {name_2} over {name_1}: {(win_2 / total_matches):.1%}")
    else:
        print("No successful evaluations to calculate win rate.")
    
    with open(os.path.join(JUDGE_OUT_DIR, output_filename), "w") as f:
        json.dump(results, f, indent=4)

# --- Main ---
def main():
    checkpoints = ["ckpt0_base", "ckpt1_stage1", "ckpt2_stage2"]
    
    # 1. Automatic Metrics
    run_automatic_metrics("alpaca", checkpoints)
    run_automatic_metrics("json", checkpoints)
    
    # 2. Pairwise Judge Evaluation 
    
    # EXPERIMENT 4.1: Base vs Stage 1
    run_judge_eval(
        client, 
        f"{RESULTS_DIR}/ckpt0_base_alpaca_responses.json", 
        f"{RESULTS_DIR}/ckpt1_stage1_alpaca_responses.json", 
        "ckpt0_base", "ckpt1_stage1", 
        "judge_alpaca_0_vs_1.json"
    )
    
    # EXPERIMENT 4.4: Stage 1 vs Stage 2 
    run_judge_eval(
        client, 
        f"{RESULTS_DIR}/ckpt1_stage1_alpaca_responses.json", 
        f"{RESULTS_DIR}/ckpt2_stage2_alpaca_responses.json", 
        "ckpt1_stage1", "ckpt2_stage2", 
        "judge_alpaca_1_vs_2.json"
    )
    
    # EXPERIMENT 4.3: Stage 1 vs Stage 2 (JSON specific)
    run_judge_eval(
        client, 
        f"{RESULTS_DIR}/ckpt1_stage1_json_responses.json", 
        f"{RESULTS_DIR}/ckpt2_stage2_json_responses.json", 
        "ckpt1_stage1", "ckpt2_stage2", 
        "judge_json_1_vs_2.json"
    )

if __name__ == "__main__":
    main()