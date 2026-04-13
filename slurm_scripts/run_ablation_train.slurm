#!/bin/bash
#SBATCH --job-name=ablation_stage2
#SBATCH --output=../logs/ablation_%j.log
#SBATCH --error=../logs/ablation_%j.err
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# 1. Load your environment
module load anaconda3
source activate myenv

# 2. Install requirements
pip install -r ../requirements.txt -q

echo "=================================================="
echo "Starting Ablation Run 1: Epoch 1"
echo "=================================================="

# Changed --epochs to --num_train_epochs
accelerate launch --config_file ../accelerate-config-v2-ds.yaml ../src/training/train_stage2_ablation_json.py --num_train_epochs 1

echo "=================================================="
echo "Starting Ablation Run 2: Epoch 2"
echo "=================================================="

# Changed --epochs to --num_train_epochs
accelerate launch --config_file ../accelerate-config-v2-ds.yaml ../src/training/train_stage2_ablation_json.py --num_train_epochs 2

echo "=================================================="
echo "Ablation Runs Completed Successfully!"
echo "=================================================="