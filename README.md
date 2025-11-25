# MFT-Qwen2: Sign-Corrected Influence Masking for Code Generation

Implementation of Sign-Corrected Influence (SCI) based parameter masking on Qwen2.5-Coder-0.5B-Instruct for code generation tasks.

## Overview

This project implements a pipeline to:
1. Compute SCI (Sign-Corrected Influence) scores on a gradient subset
2. Mask parameters with high positive SCI scores
3. Evaluate the impact on code generation benchmarks (HumanEval/HumanEval+)

**Model:** Qwen2.5-Coder-0.5B-Instruct (24 layers, 494M parameters)
**Masking Target:** Layers 15-17 (middle layers)
**Evaluation:** HumanEval and HumanEval+ (pass@1 metric)

## Quick Start

### Prerequisites

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

### Running the Pipeline

**Step 1: Download Base Model**

Download Qwen2.5-Coder-0.5B-Instruct to use as baseline:

```bash
python3 scripts/download_code_model.py
```

This saves the model to `checkpoints/fft/final/`.

**Step 2: Evaluate Baseline**

Evaluate the unmodified model on HumanEval/HumanEval+:

```bash
export USE_CPU=1  # Use CPU for stability
bash scripts/run_eval_fft_cpu.sh
```

Results saved to `logs/results/fft_eval/eval_metrics.json`.

**Step 3: Compute SCI Scores**

Calculate Sign-Corrected Influence scores on a gradient subset:

```bash
# First prepare gradient data (if not already done)
python3 scripts/prepare_data.py

# Compute SCI scores (targets layers 15-17)
bash scripts/run_sci.sh
```

This computes gradients on 256 examples and selects parameters to mask based on SCI scores.
Results saved to `logs/results/sci/`.

**Step 4: Apply Mask**

Zero out selected parameters:

```bash
bash scripts/run_mask.sh
```

Masked model saved to `checkpoints/fft_plus_sci_mask/final/`.

**Step 5: Evaluate Masked Model**

Evaluate the masked model:

```bash
export USE_CPU=1
bash scripts/eval_masked_cpu.sh
```

This automatically compares masked vs baseline results.

## Configuration

Edit `configs/sci_config.json` to adjust:

```json
{
  "sci": {
    "mask_fraction": 0.01,        // Fraction of each parameter to mask (1%)
    "max_grad_examples": 256,     // Number of gradient examples
    "layer_band": {
      "start": 15,                // First layer to mask
      "end": 17                   // Last layer to mask
    }
  }
}
```

## Data Preparation

The gradient subset uses samples from:
- Evol CodeAlpaca
- Code-Alpaca
- Tulu 3 Persona Python

To prepare data:

```bash
python3 scripts/prepare_data.py
```

This creates:
- `data_processed/train_combined.jsonl` (training data, if doing FFT)
- `data_processed/gradient_subset.jsonl` (for SCI computation)

## Evaluation Details

**Benchmarks:**
- HumanEval: 164 hand-written programming problems
- HumanEval+: Extended test suite with additional test cases

**Metric:** Pass@1 (percentage of problems solved correctly on first attempt)

**Generation Settings:**
- Temperature: 0.0 (greedy decoding)
- Max tokens: 512

## Project Structure

```
mft-qwen2/
├── configs/
│   └── sci_config.json           # Configuration
├── src/
│   ├── sci_gradients.py          # SCI computation
│   ├── apply_mask.py             # Mask application
│   ├── eval_code.py              # HumanEval evaluation
│   └── ...
├── scripts/
│   ├── download_code_model.py    # Download base model
│   ├── prepare_data.py           # Prepare datasets
│   ├── run_sci.sh               # Run SCI computation
│   ├── run_mask.sh              # Apply mask
│   ├── run_eval_fft_cpu.sh      # Evaluate baseline
│   └── eval_masked_cpu.sh       # Evaluate masked model
├── checkpoints/
│   ├── fft/final/               # Baseline model
│   └── fft_plus_sci_mask/final/ # Masked model
└── logs/results/
    ├── fft_eval/                # Baseline results
    ├── fft_plus_sci_mask_eval/  # Masked model results
    └── sci/                     # SCI computation results
```

## TPU Usage

The code supports TPU but CPU is recommended for stability. To use CPU:

```bash
export USE_CPU=1
# Then run any script
```

For TPU setup instructions, see [TPU_SETUP.md](TPU_SETUP.md).

## Troubleshooting

**XLA Compilation Issues:**
- Set `USE_CPU=1` to bypass TPU
- Ensure all gradient examples have fixed length (padding to `max_length=1024`)

**Evaluation Taking Too Long:**
- Evaluation takes 2-3 hours on CPU
- Run in screen session: `screen -S eval` then `Ctrl+A, D` to detach
- Check progress: `screen -r eval`

**Out of Memory:**
- Reduce `max_grad_examples` in config (default: 256)
- Use smaller batch size in SCI computation

## Results Structure

**Baseline Results** (`logs/results/fft_eval/eval_metrics.json`):
```json
{
  "humaneval": {
    "dataset": "HumanEval",
    "total_problems": 164,
    "passed": 23,
    "pass@1": 0.14
  },
  "humaneval_plus": {
    "dataset": "HumanEval+",
    "total_problems": 164,
    "passed": 21,
    "pass@1": 0.128
  }
}
```

**Masked Results** (`logs/results/fft_plus_sci_mask_eval/eval_metrics.json`):
Similar structure with comparison to baseline.

## Reference

This implementation is inspired by the Mask Fine-Tuning (MFT) paper:
[arXiv:2503.22764](https://arxiv.org/abs/2503.22764)

Key differences:
- Uses SCI-based selection instead of learned binary masks
- Applied to Qwen2.5-Coder-0.5B instead of LLaMA2-7B
- Focuses on code generation instead of general NLP
