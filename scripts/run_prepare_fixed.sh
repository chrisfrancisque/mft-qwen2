#!/bin/bash
# Prepare fixed-length tokenized dataset for TPU training
# Run this ONCE before training

set -e

echo "=========================================="
echo "Preparing Fixed-Length Dataset for TPU"
echo "=========================================="

cd "$(dirname "$0")/.."

# Create output directory
mkdir -p data_processed

# Run preparation script
python scripts/prepare_fixed_data.py

echo ""
echo "Dataset preparation complete!"
echo "Output: data_processed/train_1k_balanced.pt"
echo ""
echo "Next step: Run training with ./scripts/run_train_tpu_fixed.sh"
