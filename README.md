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

### Option A: Using Pre-trained Code Model (Recommended)

Use Qwen2.5-Coder-0.5B-Instruct (already trained on code) to validate the SCI masking pipeline:

1. **Download Code Model**: Use as baseline instead of FFT training
   ```bash
   python3 scripts/download_code_model.py
   ```

2. **Eval Baseline**: HumanEval/HumanEval+ pass@1 on code model
   ```bash
   bash scripts/run_eval_fft.sh
   ```

3. **Compute SCI**: Calculate influence scores on gradient subset
   ```bash
   bash scripts/run_sci.sh
   ```

4. **Apply Mask**: Zero out top 5% SCI-positive parameters
   ```bash
   bash scripts/run_mask.sh
   ```

5. **Eval Masked**: HumanEval/HumanEval+ pass@1 on masked model
   ```bash
   bash scripts/eval_masked.sh
   ```

6. **Compare Results**: Generate comparison table
   ```bash
   python3 scripts/compare_results.py
   ```

**Estimated time: ~2 hours**

### Option B: With FFT Training (If TPU resources available)

1. **Data Prep**: Load 3 coding datasets
   ```bash
   python3 scripts/prepare_data.py
   ```

2. **FFT**: Full fine-tune Qwen2-0.5B on coding data
   ```bash
   bash scripts/run_fft_tpu.sh
   ```

3. Continue with steps 2-6 from Option A

**Estimated time: 6-8 hours**

## Quick Start (TPU)

**3 commands to run the complete pipeline:**

```bash
# 1. SSH into your TPU VM
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE

# 2. Setup (clears old files, clones repo, installs deps)
curl -sSL https://raw.githubusercontent.com/chrisfrancisque/mft-qwen2/main/scripts/setup_tpu.sh | bash

# 3. Run full pipeline
cd mft-qwen2
bash scripts/run_full_pipeline.sh
```

**Estimated time:** 4-6 hours on TPU v4-8

For detailed TPU setup instructions, troubleshooting, and monitoring, see [TPU_SETUP.md](TPU_SETUP.md)

## Manual Setup

```bash
# Clone repository
git clone https://github.com/chrisfrancisque/mft-qwen2.git
cd mft-qwen2

# Install dependencies
pip install -r requirements.txt
pip install human-eval

# Create directories
mkdir -p data_raw data_processed checkpoints logs/results
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
