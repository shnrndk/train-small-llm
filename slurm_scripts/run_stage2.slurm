#!/bin/bash
#SBATCH --job-name=json_stage2
#SBATCH --output=../logs/stage2_%j.log
#SBATCH --error=../logs/stage2_%j.err
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
pip install -r ../requirements.txt -q

# Run the Stage 2 training script using your accelerate config
echo "Starting Stage 2 JSON Fine-Tuning..."
accelerate launch --config_file ../accelerate-config-v2-ds.yaml ../src/training/train_stage2_json.py --num_train_epochs 3