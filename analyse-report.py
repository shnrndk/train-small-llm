import json
import os

JUDGE_FILE = "judge_results/judge_alpaca_1_vs_2.json"

def calculate_averages():
    with open(JUDGE_FILE, 'r') as f:
        data = json.load(f)

    # Initialize counters for the 5 dimensions
    dimensions = ["instruction_following", "correctness", "clarity", "completeness", "conciseness"]
    scores_1 = {dim: 0 for dim in dimensions}
    scores_2 = {dim: 0 for dim in dimensions}
    
    forgetting_examples = []
    
    for item in data:
        # Determine which checkpoint was A and which was B
        is_ckpt1_A = item["checkpoint_a"] == "ckpt1_stage1"
        
        # Add scores
        for dim in dimensions:
            scores_1[dim] += item["response_a_scores"][dim] if is_ckpt1_A else item["response_b_scores"][dim]
            scores_2[dim] += item["response_b_scores"][dim] if is_ckpt1_A else item["response_a_scores"][dim]
            
        # Catch examples where Stage 2 regressed (Forgetting)
        if item["actual_winner"] == "ckpt1_stage1" and len(forgetting_examples) < 2:
            forgetting_examples.append({
                "prompt": item.get("prompt_id", "Unknown"),
                "justification": item["justification"]
            })

    total = len(data)
    print("="*50)
    print("=== Section 4.2: Average Judge Scores (Stage 1 vs Stage 2) ===")
    print(f"{'Dimension':<25} | {'Stage 1 Avg':<12} | {'Stage 2 Avg':<12} | {'Change (Stage 2 - 1)':<12}")
    print("-" * 70)
    for dim in dimensions:
        avg1 = scores_1[dim] / total
        avg2 = scores_2[dim] / total
        diff = avg2 - avg1
        print(f"{dim:<25} | {avg1:<12.2f} | {avg2:<12.2f} | {diff:<12.2f}")

    print("\n" + "="*50)
    print("=== Section 4.4: Examples of Catastrophic Forgetting ===")
    for ex in forgetting_examples:
        print(f"\nPrompt ID: {ex['prompt']}")
        print(f"Judge Justification for why Stage 1 won: {ex['justification']}")

if __name__ == "__main__":
    if os.path.exists(JUDGE_FILE):
        calculate_averages()
    else:
        print(f"Could not find {JUDGE_FILE}. Make sure you are in the right directory.")