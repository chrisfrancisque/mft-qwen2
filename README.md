# MFT-Qwen2: Sign-Corrected Influence Masking for Code Generation

Replication of the Mask Fine-Tuning (MFT) paper's coding domain experiment using **Sign-Corrected Influence (SCI)** instead of learned binary masks on **Qwen2-0.5B**.

## Overview

This project tests whether SCI-based parameter masking can improve a fully fine-tuned model on coding tasks, similar to the MFT paper's results (29.3% → 31.7% on HumanEval for LLaMA2-7B).

**Key Differences from MFT:**
- Model: Qwen2-0.5B (24 layers, 494M params)
- Masking: Sign-Corrected Influence (SCI) instead of learned masks
- Layer band: 15-17 (mapped from LLaMA 20-23)
- Hardware: TPU v4-8

## Pipeline

1. **Data Prep**: Load 3 coding datasets (30k train, 1k gradient subset)
2. **FFT**: Full fine-tune Qwen2-0.5B on 30k examples
3. **Eval FFT**: Baseline HumanEval/HumanEval+ pass@1
4. **Compute SCI**: Calculate influence scores on 1k holdout
5. **Mask**: Zero top 5% SCI-positive parameters in layers 15-17
6. **Eval Masked**: HumanEval/HumanEval+ pass@1
7. **Compare**: Results analysis

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For HumanEval evaluation
pip install human-eval
```

## Usage

See `scripts/` for individual stage scripts.

## Project Structure

```
mft-qwen2/
├── configs/           # Experiment configs
├── src/              # Core modules
├── scripts/          # Run scripts
├── data_raw/         # Downloaded datasets
├── data_processed/   # Preprocessed data
├── checkpoints/      # Model checkpoints
└── logs/            # Results and metrics
```

## Citation

MFT Paper: [arXiv:2503.22764](https://arxiv.org/abs/2503.22764)
