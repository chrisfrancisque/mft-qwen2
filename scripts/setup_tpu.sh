#!/bin/bash
#
# Setup script to run on TPU VM
# This clears old files, pulls the repo, and installs dependencies
#
# Usage (run on TPU):
#   bash setup_tpu.sh
#

set -e

echo "======================================================================"
echo "MFT-Qwen2: TPU Setup"
echo "======================================================================"

# Clean up old directories if they exist
echo ""
echo "Cleaning up old files..."
if [ -d "mft-qwen2" ]; then
    echo "Removing old mft-qwen2 directory..."
    rm -rf mft-qwen2
fi

if [ -d "mft_coding_replication" ]; then
    echo "Removing old mft_coding_replication directory..."
    rm -rf mft_coding_replication
fi

# Clone the repository
echo ""
echo "Cloning repository..."
git clone https://github.com/chrisfrancisque/mft-qwen2.git
cd mft-qwen2

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data_raw
mkdir -p data_processed
mkdir -p checkpoints/fft
mkdir -p checkpoints/fft_plus_sci_mask
mkdir -p logs/results/fft_eval
mkdir -p logs/results/sci
mkdir -p logs/results/fft_plus_sci_mask_eval

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install human-eval

# Verify TPU access
echo ""
echo "Verifying TPU access..."
python3 -c "import torch_xla; import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.xla_device()}')" || {
    echo "Warning: TPU not detected. This script should be run on a TPU VM."
}

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. cd mft-qwen2"
echo "2. Run data preparation: python3 scripts/prepare_data.py"
echo "3. Run full pipeline: bash scripts/run_full_pipeline.sh"
echo ""
echo "Or run stages individually (see README.md)"
