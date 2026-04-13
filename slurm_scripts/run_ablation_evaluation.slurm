#!/bin/bash
#SBATCH --job-name=eval_generate
#SBATCH --output=../logs/eval_generate_%j.log   # Path changed to ../logs
#SBATCH --error=../logs/eval_generate_%j.err    # Path changed to ../logs
#SBATCH --partition=gpu1v100                    # UTSA V100 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# 1. Load modules and activate your environment
module load anaconda3
source activate myenv

# 2. Verify dependencies pointing to the root requirements file
echo "Verifying dependencies..."
pip install -r ../requirements.txt -q

# 3. Run the Generation script using the relative path to root
echo "Starting Evaluation Generation..."
python ../src/evaluation/generate_responses_for_ablation_study.py

echo "Generation completed!"