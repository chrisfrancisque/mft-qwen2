"""
Entry point for applying SCI mask to FFT checkpoint.

Loads FFT checkpoint, applies mask by zeroing parameters, and saves masked model.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.apply_mask import apply_sci_mask


def main():
    parser = argparse.ArgumentParser(description="Apply SCI mask to Qwen2 FFT checkpoint")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SCI config JSON"
    )

    parser.add_argument(
        "--fft_checkpoint",
        type=str,
        required=True,
        help="Path to FFT checkpoint directory"
    )

    parser.add_argument(
        "--mask_indices",
        type=str,
        required=True,
        help="Path to mask_indices.json from SCI computation"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save masked checkpoint"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    sci_config = config["sci"]

    # Get target layers for verification
    layer_band = sci_config["layer_band"]
    target_layers = list(range(layer_band["start"], layer_band["end"] + 1))

    # Paths
    fft_checkpoint_dir = Path(args.fft_checkpoint)
    mask_indices_path = Path(args.mask_indices)
    output_dir = Path(args.output_dir)

    # Apply mask
    results = apply_sci_mask(
        fft_checkpoint_dir=fft_checkpoint_dir,
        mask_indices_path=mask_indices_path,
        output_dir=output_dir,
        target_layers=target_layers
    )

    # Print summary
    if results:
        mask_stats = results["mask_stats"]
        verification = results["verification"]

        print("\n" + "=" * 80)
        print("MASKING SUMMARY")
        print("=" * 80)

        print(f"\nMasking statistics:")
        print(f"  Total model parameters: {mask_stats['total_model_params']:,}")
        print(f"  Parameters masked: {mask_stats['total_params_masked']:,}")
        print(f"  Masking ratio: {mask_stats['masking_ratio']:.4%}")
        print(f"  Parameters modified: {mask_stats['params_modified']}")

        print(f"\nVerification (target layers {verification['target_layers']}):")
        print(f"  Target layer parameters: {verification['target_total']:,}")
        print(f"  Target layer zeros: {verification['target_zeros']:,}")
        print(f"  Target layer zero ratio: {verification['target_zero_ratio']:.4%}")

        if verification['non_target_zeros'] > 0:
            print(f"\nWarning: Non-target layers have {verification['non_target_zeros']:,} zeros")


if __name__ == "__main__":
    main()
