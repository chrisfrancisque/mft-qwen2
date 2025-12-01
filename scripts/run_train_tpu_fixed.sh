#!/bin/bash
# TPU-Optimized Training with Fixed Shapes
#
# This script trains with all XLA recompilation fixes:
# - Fixed sequence length (512)
# - drop_last=True for constant batch sizes
# - Dynamic warmup calculation
# - Frequent checkpointing
# - XLA metrics logging

set -e

echo "=========================================="
echo "TPU Training with Fixed Shapes"
echo "=========================================="

cd "$(dirname "$0")/.."

# Check if data exists
if [ ! -f "data_processed/train_1k_balanced.pt" ]; then
    echo "ERROR: Pre-tokenized data not found!"
    echo "Run ./scripts/run_prepare_fixed.sh first"
    exit 1
fi

# Create output directory
mkdir -p checkpoints/fft_tpu_fixed
mkdir -p logs

# Set environment variables for TPU
export XLA_USE_BF16=1
export XRT_TPU_CONFIG="localservice;0;localhost:51011"

# Run training
echo ""
echo "Starting TPU training..."
echo "Checkpoints will be saved at steps: 1, 5, 10, 15, 20, 25, 30, 31"
echo ""

python scripts/train_tpu_fixed.py 2>&1 | tee logs/train_tpu_fixed_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training complete!"
echo "Checkpoints: checkpoints/fft_tpu_fixed/"
