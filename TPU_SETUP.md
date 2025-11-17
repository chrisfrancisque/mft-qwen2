# TPU Setup and Deployment Guide

## Prerequisites

- TPU v4-8 VM already created
- SSH access to TPU VM

## Quick Start (3 commands)

```bash
# 1. SSH into your TPU VM
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE

# 2. Run setup script (clears old files, clones repo, installs deps)
curl -sSL https://raw.githubusercontent.com/chrisfrancisque/mft-qwen2/main/scripts/setup_tpu.sh | bash

# 3. Run full pipeline
cd mft-qwen2
bash scripts/run_full_pipeline.sh
```

## Detailed Instructions

### Step 1: Connect to TPU

```bash
# List your TPUs
gcloud compute tpus tpu-vm list

# SSH into TPU
gcloud compute tpus tpu-vm ssh YOUR_TPU_NAME --zone YOUR_ZONE
```

### Step 2: Clean and Setup

```bash
# Remove old directories
rm -rf mft-qwen2 mft_coding_replication

# Clone repository
git clone https://github.com/chrisfrancisque/mft-qwen2.git
cd mft-qwen2

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install human-eval

# Create directories
mkdir -p data_raw data_processed checkpoints logs/results
```

Or use the automated setup script:

```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/chrisfrancisque/mft-qwen2/main/scripts/setup_tpu.sh | bash
cd mft-qwen2
```

### Step 3: Run Pipeline

**Option A: Run full pipeline (recommended)**

```bash
bash scripts/run_full_pipeline.sh
```

This runs all 7 stages automatically. Estimated time: 4-6 hours.

**Option B: Run stages individually**

```bash
# Stage 1: Data preparation (~10 min)
python3 scripts/prepare_data.py

# Stage 2: FFT training (~2-3 hours)
bash scripts/run_fft_tpu.sh

# Stage 3: Evaluate FFT baseline (~30 min)
bash scripts/run_eval_fft.sh

# Stage 4: Compute SCI scores (~20 min)
bash scripts/run_sci.sh

# Stage 5: Apply mask (~2 min)
bash scripts/run_mask.sh

# Stage 6: Evaluate masked model (~30 min)
bash scripts/eval_masked.sh

# Stage 7: Compare results (~1 min)
python3 scripts/compare_results.py
```

## Monitoring Progress

### Check training progress

```bash
# View FFT training logs
tail -f logs/fft_training.log

# View latest logs
ls -lt logs/
```

### Check for errors

```bash
# Check for XLA errors
grep -i "error\|xla\|crash" logs/*.log

# Check GPU/TPU memory
nvidia-smi  # or
python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_memory_info(xm.xla_device()))"
```

### Monitor disk space

```bash
df -h
du -sh checkpoints/* data_processed/*
```

## Troubleshooting

### Issue: "No module named 'torch_xla'"

```bash
pip install torch-xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html
```

### Issue: "CUDA out of memory" or XLA compilation errors

This shouldn't happen on TPU v4-8, but if it does:
- Reduce batch size in `configs/fft_qwen2_0.5b.json`
- Reduce max_length from 1024 to 512

### Issue: Dataset download fails

```bash
# Pre-download datasets
python3 -c "
from datasets import load_dataset
load_dataset('nickrosh/Evol-Instruct-Code-80k-v1')
load_dataset('lucasmccabe-lmi/CodeAlpaca-20k')
load_dataset('allenai/tulu-3-sft-personas-python-v0.3')
"
```

### Issue: Out of disk space

```bash
# Clean up old checkpoints
rm -rf checkpoints/fft/checkpoint-*

# Keep only final checkpoints
find checkpoints -name "checkpoint-*" -type d -exec rm -rf {} +
```

## File Transfer (if needed)

### Download results from TPU

```bash
# From your local machine
gcloud compute tpus tpu-vm scp YOUR_TPU_NAME:~/mft-qwen2/logs/results . --zone YOUR_ZONE --recurse

# Or specific files
gcloud compute tpus tpu-vm scp YOUR_TPU_NAME:~/mft-qwen2/logs/results/*/results.json . --zone YOUR_ZONE
```

### Upload custom configs

```bash
# From your local machine
gcloud compute tpus tpu-vm scp configs/sci_config.json YOUR_TPU_NAME:~/mft-qwen2/configs/ --zone YOUR_ZONE
```

## Expected Timeline

Based on TPU v4-8:

| Stage | Task | Estimated Time |
|-------|------|----------------|
| 1 | Data prep | 10 min |
| 2 | FFT training (3 epochs, 30k examples) | 2-3 hours |
| 3 | FFT evaluation (HumanEval + HumanEval+) | 30 min |
| 4 | SCI computation (999 examples) | 20 min |
| 5 | Apply mask | 2 min |
| 6 | Masked evaluation | 30 min |
| 7 | Compare results | 1 min |
| **Total** | | **4-6 hours** |

## Expected Disk Usage

- `data_processed/`: ~500 MB
- `checkpoints/fft/`: ~1 GB
- `checkpoints/fft_plus_sci_mask/`: ~1 GB
- `logs/`: ~100 MB

**Total: ~2.5-3 GB**

## Configuration Changes

### Change target layers (15-17 vs 20-23)

Edit `configs/sci_config.json`:

```json
"layer_band": {
  "start": 20,  // Change from 15
  "end": 23,    // Change from 17
  "note": "Deep layers for Qwen2-0.5B"
}
```

### Change masking ratio (5% vs 2%)

Edit `configs/sci_config.json`:

```json
"mask_fraction": 0.02  // Change from 0.05 for 2%
```

### Reduce training time (for testing)

Edit `configs/fft_qwen2_0.5b.json`:

```json
"num_epochs": 1  // Change from 3 for quick test
```

## Next Steps After Completion

1. **View results:**
   ```bash
   python3 scripts/compare_results.py
   ```

2. **Download checkpoints** (if you want to run inference locally):
   ```bash
   gcloud compute tpus tpu-vm scp YOUR_TPU_NAME:~/mft-qwen2/checkpoints . --zone YOUR_ZONE --recurse
   ```

3. **Analyze logs:**
   ```bash
   grep "pass@1" logs/results/*/results.json
   grep "loss" logs/fft_training.log | tail -20
   ```

4. **Clean up TPU** (to avoid charges):
   ```bash
   gcloud compute tpus tpu-vm delete YOUR_TPU_NAME --zone YOUR_ZONE
   ```
