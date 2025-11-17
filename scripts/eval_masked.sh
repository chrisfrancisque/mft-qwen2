#!/bin/bash
#
# Evaluate FFT+SCI masked model on HumanEval and HumanEval+
#
# Usage:
#   bash scripts/eval_masked.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: Evaluate FFT+SCI Masked Model"
echo "======================================================================"

# Paths
CONFIG_PATH="configs/sci_config.json"
CHECKPOINT_DIR="checkpoints/fft_plus_sci_mask/final"
OUTPUT_DIR="logs/results/fft_plus_sci_mask_eval"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if checkpoint exists
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Error: Masked checkpoint not found at ${CHECKPOINT_DIR}"
    echo "Please apply SCI mask first: bash scripts/run_mask.sh"
    exit 1
fi

# Run evaluation
echo ""
echo "Evaluating masked model on HumanEval and HumanEval+..."
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

python3 scripts/eval_humaneval.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# Display results if available
if [ -f "${OUTPUT_DIR}/humaneval_results.json" ]; then
    echo "HumanEval Results:"
    python3 -c "import json; results = json.load(open('${OUTPUT_DIR}/humaneval_results.json')); print(f\"  Pass@1: {results['pass@1']:.2%}\")"
fi

if [ -f "${OUTPUT_DIR}/humaneval_plus_results.json" ]; then
    echo "HumanEval+ Results:"
    python3 -c "import json; results = json.load(open('${OUTPUT_DIR}/humaneval_plus_results.json')); print(f\"  Pass@1: {results['pass@1']:.2%}\")"
fi

echo ""
echo "Compare with FFT baseline in logs/results/fft_eval/"
