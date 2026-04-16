# Methodology

This project investigates the complex dynamics of post-training and sequential fine-tuning in Large Language Models (LLMs). Specifically, we aim to quantify the trade-offs between acquiring highly structured, rigid capabilities (JSON generation) and retaining general, fluid conversational alignment. To achieve this, we engineered a two-stage training pipeline utilizing imitation learning, accompanied by an ablation study on epoch scaling.

## 1.1 Model Selection

**Student Model:** A highly capable but compact open-weight base model (`meta-llama/Llama-3.2-3B`) was selected as the student. Smaller parameter models are ideal for post-training experiments as they exhibit more pronounced shifts in behavior during fine-tuning compared to their larger counterparts, making phenomena like catastrophic forgetting easier to measure within academic compute constraints. During training, the base model is dynamically quantized to 4-bit (`nf4`) precision using `bitsandbytes` (with double quantization and `float16` compute type) to drastically reduce VRAM footprints.

**Teacher Model:** We utilized a frontier-level model (`llama-3.3-70b-instruct-awq`) deployed locally to generate synthetic training data. These large-scale models possess robust internal representations of complex schema structures, making them ideal teachers for distilling strict reasoning frameworks into smaller student architectures.

## 1.2 Data Sourcing and Imitation Learning Pipeline

The training corpus was divided into two distinct datasets to support the sequential training stages:

**Stage 1 (General Alignment):** We utilized the open-source Stanford Alpaca dataset (`tatsu-lab/alpaca`), which contains 52,000 instruction-following records. A 90/10 train-validation split was utilized. It covers a broad taxonomy of tasks and serves as the academic standard for transforming a base completion model into a conversational assistant. 

**Stage 2 (Structured Output):** To teach the model strict structural compliance, we constructed a synthetic *JSON Instruct* dataset using an **Imitation Learning Pipeline**:
- The teacher model (`llama-3.3-70b-instruct-awq`) received prompts defining five unique sub-tasks: extraction, json-repair, label-classification, schema-constrained generation, and tool-call formatting.
- **Parametric strictness**: The teacher model was generationally constrained using an extremely low temperature (`0.1`) and enforced system prompts mandating raw string output without markdown blocks.
- **Automated Validation & Retry**: The generation pipeline programmatically verified the output syntax. If the teacher hallucinated invalid JSON or injected conversational fluff, the pipeline initiated up to 3 automated retries per sample to guarantee a pristine structural dataset of maximum quality for student distillation.

## 1.3 Training Design, Ablation Setup, and UTSA HPC Setup

Training was executed sequentially leveraging Parameter-Efficient Fine-Tuning (PEFT) to ensure stable convergence without over-saturating the base model weights. 

**Hyperparameters (Stages 1 & 2):**
- **LoRA Configuration**: Rank (`r=32`), Alpha (`64`), Dropout (`0.05`), Target Modules (`all-linear`), Task Type (`CAUSAL_LM`).
- **Optimization Strategy**: Learning rate of `1e-4` using a `cosine` scheduler with a `0.3` warmup ratio.
- **Batching & Regularization**: Per-device batch size of `4` with `8` gradient accumulation steps (yielding a larger effective batch size). Max gradient norm clipped at `0.3`, weight decay at `0.01`. Half-precision (`fp16`) and `gradient_checkpointing` were enabled for maximal memory efficiency on sequences up to `1024` tokens.

**Sequential Pathway:**
- **Checkpoint 0**: The untuned base model (`Llama-3.2-3B`).
- **Checkpoint 1 (Stage 1)**: Base model fine-tuned exclusively on the Alpaca dataset.
- **Checkpoint 2 (Stage 2)**: Checkpoint 1 fine-tuned exclusively on the synthetic JSON Instruct dataset.

**Ablation Study Design:**
To understand how quickly catastrophic forgetting occurs, we conducted an ablation study during Stage 2. Custom arguments (`num_train_epochs`) were passed to the SLURM batch scripts. We trained independent adapters for 1, 2, and 3 epochs to observe how quickly JSON syntax mastery scales against the deterioration of core conversational fluency.

**HPC Configuration:**
All computational workloads—including training, evaluation, ablation, and text generation—were executed on the UTSA High-Performance Computing (HPC) cluster. Jobs relied upon SLURM workload manager, allocating 1 compute node on the `gpu1v100` partition, ensuring deterministic execution via dedicated V100 GPU acceleration.

## 1.4 Evaluation Protocol

Evaluation requires moving beyond traditional string-matching. A multi-modal hybrid pipeline was adopted:

**Automated Text Metrics:** 
ROUGE (1, 2, L) and BERTScore (configured with `roberta-large`) were employed via Hugging Face's `evaluate` framework to quantify semantic alignment against conversational reference distributions.

**Automated Structural Metrics (Custom Engine):** 
A localized Python assessment engine parsed string outputs to measure:
- *JSON Validity Rate*: Binary checks resolving parsing errors.
- *Schema Compliance*: Ratio of outputs yielding identically matched Dictionary/Key objects against a reference constraint.
- *Exact Match*: Absolute fidelity against ground truth extraction patterns.
- *Field-level F1*: Calculated using precision and recall aggregations across deeply nested Key:Value pairings.

**LLM-as-a-Judge API:**
`gpt-4o-mini` was utilized via OpenAI's API as an impartial scoring engine for randomized pairwise A/B matchups between checkpoints. 
- Over 200 Alpaca prompts and 100 JSON prompts were assessed.
- An environment-constrained system prompt penalized the model for hallucination or formatting deviations, utilizing `temperature=0.1` and `response_format={"type": "json_object"}`.
- Outputs were scored strictly across five dimensions: *Instruction Following*, *Correctness*, *Clarity*, *Completeness*, and *Conciseness* on a unified 1-5 scalar system, with randomized response injection orders (A vs B) to mathematically mitigate positional evaluation biases.
