#!/bin/bash
#
# Apply SCI mask to FFT checkpoint
#
# Usage:
#   bash scripts/run_mask.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: Apply SCI Mask"
echo "======================================================================"

# Paths
CONFIG_PATH="configs/sci_config.json"
FFT_CHECKPOINT="checkpoints/fft/final"
MASK_INDICES="logs/results/sci/mask_indices.json"
OUTPUT_DIR="checkpoints/fft_plus_sci_mask/final"

# Create output directory
mkdir -p "$(dirname "${OUTPUT_DIR}")"

# Check if FFT checkpoint exists
if [ ! -d "${FFT_CHECKPOINT}" ]; then
    echo "Error: FFT checkpoint not found at ${FFT_CHECKPOINT}"
    echo "Please run FFT training first: bash scripts/run_fft_tpu.sh"
    exit 1
fi

# Check if mask indices exist
if [ ! -f "${MASK_INDICES}" ]; then
    echo "Error: Mask indices not found at ${MASK_INDICES}"
    echo "Please run SCI computation first: bash scripts/run_sci.sh"
    exit 1
fi

# Apply mask
echo ""
echo "Applying SCI mask to FFT checkpoint..."
echo "FFT checkpoint: ${FFT_CHECKPOINT}"
echo "Mask indices: ${MASK_INDICES}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python3 scripts/run_mask.py \
    --config "${CONFIG_PATH}" \
    --fft_checkpoint "${FFT_CHECKPOINT}" \
    --mask_indices "${MASK_INDICES}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/apply_mask.log"

echo ""
echo "======================================================================"
echo "Masking Complete!"
echo "======================================================================"
echo "Masked checkpoint saved to: ${OUTPUT_DIR}"
echo "Statistics saved to: checkpoints/fft_plus_sci_mask/mask_stats.json"
echo "Logs saved to: logs/apply_mask.log"
echo ""
echo "Next: Evaluate masked model with bash scripts/eval_masked.sh"
