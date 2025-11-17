"""
Entry point for SCI gradient computation.

Computes Sign-Corrected Influence scores and selects top parameters to mask.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sci_gradients import compute_sci_and_select


def main():
    parser = argparse.ArgumentParser(description="Compute SCI scores for Qwen2")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SCI config JSON (contains layer_band, mask_fraction, etc.)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FFT checkpoint directory"
    )

    parser.add_argument(
        "--grad_data",
        type=str,
        required=True,
        help="Path to grad_subset.jsonl"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save SCI results and mask indices"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    sci_config = config["sci"]

    # Paths
    checkpoint_dir = Path(args.checkpoint)
    grad_data_path = Path(args.grad_data)
    output_dir = Path(args.output_dir)

    # Run SCI computation
    results = compute_sci_and_select(
        checkpoint_dir=checkpoint_dir,
        grad_data_path=grad_data_path,
        output_dir=output_dir,
        sci_config=sci_config
    )

    # Print summary
    if results["stats"]:
        stats = results["stats"]

        print("\n" + "=" * 80)
        print("SCI COMPUTATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal parameters in target layers: {stats['total_params']:,}")
        print(f"Mask fraction: {stats['mask_fraction']:.1%}")
        print(f"Parameters selected for masking: {stats['num_selected']:,}")
        print(f"Parameters with positive SCI: {stats['num_positive']:,}")

        if stats['max_score'] is not None:
            print(f"\nSCI score range:")
            print(f"  Min: {stats['min_score']:.6f}")
            print(f"  Max: {stats['max_score']:.6f}")
            print(f"  Mean: {stats['mean_score']:.6f}")


if __name__ == "__main__":
    main()
