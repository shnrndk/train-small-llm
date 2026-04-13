#!/bin/bash
#SBATCH --job-name=teacher_gen
#SBATCH --output=logs/gen_%j.log
#SBATCH --error=logs/gen_%j.err
#SBATCH --partition=gpu2v100     # IMPORTANT: Requesting the 2x V100 queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2             # IMPORTANT: Requesting 2 GPUs
#SBATCH --time=04:00:00

module load anaconda3
source activate myenv

# Install requirements quietly just in case
pip install -r ..requirements.txt -q

# Run the generation script
python ../src/data_prep/generate_json_dataset.py