#!/bin/bash
# CPU Training with Fixed Shapes (for local testing)
#
# Use this to verify the training loop works before deploying to TPU

set -e

echo "=========================================="
echo "CPU Training with Fixed Shapes (Test Mode)"
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

# Force CPU mode
export USE_CPU=1

echo ""
echo "Starting CPU training (test mode)..."
echo "This will be slow but verifies the training loop works"
echo ""

python scripts/train_tpu_fixed.py 2>&1 | tee logs/train_cpu_fixed_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Training complete!"
