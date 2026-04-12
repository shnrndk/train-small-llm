#!/bin/bash
#SBATCH --job-name=eval_stage3
#SBATCH --output=stage3_%j.log
#SBATCH --error=stage3_%j.err
#SBATCH --partition=gpu1v100     # UTSA V100 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # Requesting 1 GPU
#SBATCH --time=04:00:00          # 4 hours limit

# Load modules and activate your environment
module load anaconda3
source activate myenv

# Install required packages automatically before training
echo "Verifying dependencies..."
pip install -r requirements.txt -q

# Run the Stage 2 training script using your accelerate config
echo "Starting Stage 3 Generation..."
accelerate launch --config_file accelerate-config-v2-ds.yaml src/evaluation/generate_responses.py