#!/bin/bash
#
# Evaluate FFT+SCI masked model on HumanEval and HumanEval+ (CPU version)
#
# Usage:
#   bash scripts/eval_masked_cpu.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: Evaluate FFT+SCI Masked Model (CPU)"
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
echo "Evaluating masked model on HumanEval and HumanEval+ (CPU)..."
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Note: CPU eval will be slower but more reliable than TPU."
echo "Expected time: ~2-3 hours for both HumanEval and HumanEval+"
echo ""

# Force CPU usage
export USE_CPU=1

python3 scripts/eval_humaneval.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/eval_cpu.log"

echo ""
echo "======================================================================"
echo "Evaluation Complete!"
echo "======================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# Display results if available
if [ -f "${OUTPUT_DIR}/eval_metrics.json" ]; then
    echo "Results:"
    python3 << 'EOF'
import json

with open('logs/results/fft_plus_sci_mask_eval/eval_metrics.json') as f:
    masked_results = json.load(f)

with open('logs/results/fft_eval/eval_metrics.json') as f:
    baseline_results = json.load(f)

print("="*80)
print("COMPARISON: Baseline vs Masked Model")
print("="*80)

print("\nHumanEval:")
print(f"  Baseline:  {baseline_results['humaneval']['pass@1']:.1%} ({baseline_results['humaneval']['passed']}/{baseline_results['humaneval']['total_problems']})")
print(f"  Masked:    {masked_results['humaneval']['pass@1']:.1%} ({masked_results['humaneval']['passed']}/{masked_results['humaneval']['total_problems']})")
delta_he = masked_results['humaneval']['pass@1'] - baseline_results['humaneval']['pass@1']
print(f"  Delta:     {delta_he:+.1%}")

print("\nHumanEval+:")
print(f"  Baseline:  {baseline_results['humaneval_plus']['pass@1']:.1%} ({baseline_results['humaneval_plus']['passed']}/{baseline_results['humaneval_plus']['total_problems']})")
print(f"  Masked:    {masked_results['humaneval_plus']['pass@1']:.1%} ({masked_results['humaneval_plus']['passed']}/{masked_results['humaneval_plus']['total_problems']})")
delta_hep = masked_results['humaneval_plus']['pass@1'] - baseline_results['humaneval_plus']['pass@1']
print(f"  Delta:     {delta_hep:+.1%}")
EOF
fi
