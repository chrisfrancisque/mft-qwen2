#!/bin/bash
#
# Compute SCI scores and select top parameters to mask
#
# Usage:
#   bash scripts/run_sci.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: SCI Gradient Computation"
echo "======================================================================"

# Paths
CONFIG_PATH="configs/sci_config.json"
CHECKPOINT_DIR="checkpoints/fft/final"
GRAD_DATA="data_processed/grad_subset.jsonl"
OUTPUT_DIR="logs/results/sci"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: FFT checkpoint not found at ${CHECKPOINT_DIR}"
    echo "Please run FFT training first: bash scripts/run_fft_tpu.sh"
    exit 1
fi

# Check if gradient data exists
if [ ! -f "${GRAD_DATA}" ]; then
    echo "Error: Gradient subset not found at ${GRAD_DATA}"
    echo "Please run data preparation first: python3 scripts/prepare_data.py"
    exit 1
fi

# Run SCI computation
echo ""
echo "Computing SCI scores..."
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Gradient data: ${GRAD_DATA}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python3 scripts/run_sci.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --grad_data "${GRAD_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/sci_computation.log"

echo ""
echo "======================================================================"
echo "SCI Computation Complete!"
echo "======================================================================"
echo "Mask indices saved to: ${OUTPUT_DIR}/mask_indices.json"
echo "Statistics saved to: ${OUTPUT_DIR}/sci_stats.json"
echo "Logs saved to: logs/sci_computation.log"
