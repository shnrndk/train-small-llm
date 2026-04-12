#!/bin/bash
#SBATCH --job-name=alpaca_stage1
#SBATCH --output=stage1_%j.log
#SBATCH --error=stage1_%j.err
#SBATCH --partition=gpu1v100     # UTSA V100 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # Requesting 1 GPU
#SBATCH --time=04:00:00          # 4 hours limit

# Load modules and activate your environment
module load anaconda3
source activate myenv

# Install required packages automatically before training
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q

# Run the training script using your accelerate config
accelerate launch --config_file accelerate-config-v2-ds.yaml src/training/train_stage1_alpaca.py