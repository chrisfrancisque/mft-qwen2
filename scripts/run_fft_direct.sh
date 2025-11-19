#!/bin/bash
#
# Launch Full Fine-Tuning on TPU v4-8 (Direct execution without xla_multiprocessing)
#
# Usage:
#   bash scripts/run_fft_direct.sh
#

set -eo pipefail

echo "======================================================================"
echo "MFT-Qwen2: Full Fine-Tuning on TPU v4-8 (Direct)"
echo "======================================================================"

# Set environment variables for TPU
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1

# Paths
CONFIG_PATH="configs/fft_qwen2_0.5b.json"
DATA_DIR="data_processed"
OUTPUT_DIR="checkpoints/fft"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs/results"

# Run training directly
echo ""
echo "Starting FFT training..."
echo "Config: ${CONFIG_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python3 scripts/run_fft.py \
    --config "${CONFIG_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/fft_training.log"

echo ""
echo "======================================================================"
echo "FFT Training Complete!"
echo "======================================================================"
echo "Checkpoint saved to: ${OUTPUT_DIR}/final"
echo "Logs saved to: logs/fft_training.log"
