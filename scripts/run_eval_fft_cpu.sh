#!/bin/bash
#
# Evaluate FFT baseline on HumanEval and HumanEval+ (CPU-only version)
#
# Usage:
#   bash scripts/run_eval_fft_cpu.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: Evaluate FFT Baseline (CPU)"
echo "======================================================================"

# Paths
CONFIG_PATH="configs/sci_config.json"
CHECKPOINT_DIR="checkpoints/fft/final"
OUTPUT_DIR="logs/results/fft_eval"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_DIR}"
    echo "Please download the model first: python3 scripts/download_code_model.py"
    exit 1
fi

# Run evaluation on CPU (disable TPU)
echo ""
echo "Running evaluation on CPU (TPU disabled for stability)"
echo "Evaluating checkpoint: ${CHECKPOINT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Note: CPU eval will be slower but more reliable than TPU."
echo "Expected time: ~2-3 hours for both HumanEval and HumanEval+"
echo ""

# Disable TPU by unsetting PJRT_DEVICE
unset PJRT_DEVICE

python3 scripts/eval_humaneval.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/fft_eval_cpu.log"

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: ${OUTPUT_DIR}/eval_metrics.json"
echo "Logs saved to: logs/fft_eval_cpu.log"
