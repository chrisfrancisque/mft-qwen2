#!/bin/bash
#
# Compute SCI scores and select top parameters to mask (CPU version)
#
# Usage:
#   bash scripts/run_sci_cpu.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: SCI Gradient Computation (CPU)"
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
    echo "Please download the model first: python3 scripts/download_code_model.py"
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
echo "Running SCI computation on CPU (TPU disabled due to permission errors)"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Gradient data: ${GRAD_DATA} (976 examples)"
echo "Target layers: 15-17 (Qwen2-0.5B)"
echo "Mask fraction: 5%"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Note: CPU gradient computation will be slower but more reliable."
echo "Expected time: ~1-2 hours for 976 examples"
echo ""

# Force CPU usage
export USE_CPU=1

python3 scripts/run_sci.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --grad_data "${GRAD_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/sci_computation_cpu.log"

echo ""
echo "======================================================================"
echo "SCI Computation Complete!"
echo "======================================================================"
echo "Mask indices saved to: ${OUTPUT_DIR}/mask_indices.json"
echo "Statistics saved to: ${OUTPUT_DIR}/sci_stats.json"
echo "Logs saved to: logs/sci_computation_cpu.log"
