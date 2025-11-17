#!/bin/bash
#
# Run the complete MFT-Qwen2 pipeline
#
# This script runs all stages sequentially:
# 1. Data preparation
# 2. FFT training
# 3. FFT evaluation
# 4. SCI computation
# 5. Apply mask
# 6. Masked model evaluation
# 7. Results comparison
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# To run individual stages, see scripts/run_*.sh
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "${PROJECT_DIR}"

echo "======================================================================"
echo "MFT-Qwen2: Full Pipeline"
echo "======================================================================"
echo ""
echo "This will run all stages:"
echo "  1. Data preparation"
echo "  2. FFT training (TPU v4-8)"
echo "  3. FFT evaluation"
echo "  4. SCI gradient computation"
echo "  5. Apply SCI mask"
echo "  6. Masked model evaluation"
echo "  7. Results comparison"
echo ""
echo "Estimated time: 4-6 hours (depending on TPU availability)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Log file
LOG_FILE="logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo ""
echo "======================================================================"
echo "Stage 1/7: Data Preparation"
echo "======================================================================"
python3 scripts/prepare_data.py 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 2/7: Full Fine-Tuning (FFT)"
echo "======================================================================"
bash scripts/run_fft_tpu.sh 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 3/7: Evaluate FFT Baseline"
echo "======================================================================"
bash scripts/run_eval_fft.sh 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 4/7: Compute SCI Scores"
echo "======================================================================"
bash scripts/run_sci.sh 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 5/7: Apply SCI Mask"
echo "======================================================================"
bash scripts/run_mask.sh 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 6/7: Evaluate Masked Model"
echo "======================================================================"
bash scripts/eval_masked.sh 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "Stage 7/7: Compare Results"
echo "======================================================================"
python3 scripts/compare_results.py 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Full log saved to: ${LOG_FILE}"
echo ""
echo "Results:"
echo "  - FFT baseline: logs/results/fft_eval/"
echo "  - SCI scores: logs/results/sci/"
echo "  - Masked model: logs/results/fft_plus_sci_mask_eval/"
echo ""
echo "Checkpoints:"
echo "  - FFT: checkpoints/fft/final/"
echo "  - FFT+SCI: checkpoints/fft_plus_sci_mask/final/"
