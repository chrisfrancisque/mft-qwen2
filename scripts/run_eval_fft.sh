#!/bin/bash
#
# Evaluate FFT baseline on HumanEval and HumanEval+
#
# Usage:
#   bash scripts/run_eval_fft.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: Evaluate FFT Baseline"
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
    echo "Please run FFT training first: bash scripts/run_fft_tpu.sh"
    exit 1
fi

# Run evaluation
echo ""
echo "Evaluating checkpoint: ${CHECKPOINT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python3 scripts/eval_humaneval.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "logs/fft_eval.log"

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: ${OUTPUT_DIR}/eval_metrics.json"
echo "Logs saved to: logs/fft_eval.log"
