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

1. **Data Prep**: Load 3 coding datasets (30k train, 999 gradient subset, ~2k validation)
   ```bash
   python3 scripts/prepare_data.py
   ```

2. **FFT**: Full fine-tune Qwen2-0.5B on 30k examples (3 epochs, TPU v4-8)
   ```bash
   bash scripts/run_fft_tpu.sh
   ```

3. **Eval FFT**: Baseline HumanEval/HumanEval+ pass@1
   ```bash
   bash scripts/run_eval_fft.sh
   ```

4. **Compute SCI**: Calculate influence scores on 999 holdout examples
   ```bash
   bash scripts/run_sci.sh
   ```

5. **Apply Mask**: Zero out top 5% SCI-positive parameters in target layers
   ```bash
   bash scripts/run_mask.sh
   ```

6. **Eval Masked**: HumanEval/HumanEval+ pass@1 on masked model
   ```bash
   bash scripts/eval_masked.sh
   ```

7. **Compare Results**: Generate comparison table
   ```bash
   python3 scripts/compare_results.py
   ```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For HumanEval evaluation
pip install human-eval
```

## Datasets

- **Evol CodeAlpaca** (10k train, 333 gradient subset)
- **Code-Alpaca** (10k train, 333 gradient subset)
- **Tulu 3 Persona Python** (10k train, 333 gradient subset)

Total: 30k training examples, 999 gradient examples for SCI computation

## Evaluation

- **HumanEval**: 164 coding problems
- **HumanEval+**: Extended test suite with additional test cases
- Metric: Pass@1 (percentage of problems solved on first attempt)

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
