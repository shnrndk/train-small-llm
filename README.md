# LLM Project: Train Small LLM

Read the full blog post about this project here: [https://shnrndk.github.io/train-small-llm/blog-post/](https://shnrndk.github.io/train-small-llm/blog-post/)

## Overview
This repository contains the code and configuration for training and evaluating a small Large Language Model (LLM). The project is structured to separate data preparation, training, evaluation, and deployment configurations into modular components.

## Project Structure

```text
llmProject/
├── README.md               # This file
├── config.py               # Shared configuration settings and paths
├── requirements.txt        # Python dependencies required for the project
├── checkconnection.py      # Utility script to test environment/connections
├── src/                    # Source code directory
│   ├── data_prep/          # Scripts for formatting and preparing datasets
│   │   ├── extract_alpaca_eval.py
│   │   └── generate_json_dataset.py
│   ├── training/           # Scripts for training the model
│   │   ├── train_stage1_alpaca.py
│   │   ├── train_stage2_json.py
│   │   └── train_stage2_ablation_json.py
│   ├── evaluation/         # Scripts for automated generation and evaluation
│   │   ├── generate_responses.py
│   │   ├── generate_responses_for_ablation_study.py
│   │   ├── eval_judge_gpt.py
│   │   ├── eval_judge_gpt_ablation.py
│   │   └── analyse-report.py
│   └── utils/              # Helper utilities
│       └── data_utils.py
├── slurm_scripts/          # Slurm batch scripts for cluster job execution
│   ├── run_stage1.slurm
│   ├── run_stage2.slurm
│   ├── run_ablation_train.slurm
│   ├── run_generation.slurm
│   ├── run_ablation_evaluation.slurm
│   └── run_stage3_part1.slurm
├── data/                   # Directory containing datasets (JSON and CSV)
├── prompts/                # Directory containing prompt templates for generation/evaluation
├── checkpoints/            # Directory where trained model checkpoints are saved
├── logs/                   # Directory for training and execution logs
├── eval_results/           # Output directory for generation results
├── judge_results/          # Output directory for evaluations by GPT judge
└── wandb/                  # Weights & Biases synchronization and log directory
```

## Directory Details

### `src/`
The core logic of the repository is modularized into the `src` directory:
- **`data_prep/`**: Contains scripts used to clean, prepare and extract the evaluation set (e.g. `alpaca_eval`) or construct JSON datasets for training.
- **`training/`**: Contains the training scripts. It includes a baseline stage 1 training script alongside subsequent training stages (`stage2`) and ablation studies.
- **`evaluation/`**: Handles generating responses with the trained models and using GPT as a judge to evaluate these generations. You can find scripts for generating text, running evaluations, and analyzing reports.
- **`utils/`**: Shared tools such as data loading utilities (`data_utils.py`).

### `slurm_scripts/`
A collection of bash scripts optimized for execution on Slurm clusters. They load the environment and specify computational requirements for executing modules across different pipeline stages (Stage 1 train, Stage 2 train, generation, evaluation, etc.).

### `data/`
Your centralized location for training and testing datasets. Scripts in `src/data_prep/` typically output processed data here to be ingested by the training modules.

### `prompts/`
Contains text files and templates utilized to guide the models during inference or evaluation scenarios. For example, templates that tell the GPT Judge how to perform its evaluation.

### Operational Directories
- **`checkpoints/`**: Used by the training scripts to persist intermediate and final states of the learned model.
- **`logs/`**: Standard output and error logs.
- **`eval_results/` & `judge_results/`**: Model prediction dumps and corresponding automated evaluation scores respectively.
- **`wandb/`**: Local cache for experiment tracking.

## Root Level Scripts
- **`config.py`**: Centralizes parameters such as paths, hyperparameters, or training setup variables.
- **`eval-base.py`, `eval_interactive.py`, `eval_judge.py`**: Ad-hoc or deprecated evaluation artifacts residing at the root level.
