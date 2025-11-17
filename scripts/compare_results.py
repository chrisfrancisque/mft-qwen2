"""
Compare FFT baseline vs FFT+SCI masked model results.

Generates a comparison table showing HumanEval and HumanEval+ pass@1 scores.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional


def load_results(results_dir: Path) -> Optional[Dict]:
    """Load evaluation results from directory."""
    humaneval_path = results_dir / "humaneval_results.json"
    humaneval_plus_path = results_dir / "humaneval_plus_results.json"

    if not humaneval_path.exists() or not humaneval_plus_path.exists():
        return None

    with open(humaneval_path, 'r') as f:
        humaneval_results = json.load(f)

    with open(humaneval_plus_path, 'r') as f:
        humaneval_plus_results = json.load(f)

    return {
        "humaneval": humaneval_results,
        "humaneval_plus": humaneval_plus_results
    }


def print_comparison_table(
    fft_results: Optional[Dict],
    masked_results: Optional[Dict],
    mask_stats: Optional[Dict] = None
):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON: FFT vs FFT+SCI Masked")
    print("=" * 100)

    # Extract metrics
    if fft_results:
        fft_he = fft_results["humaneval"]["pass@1"] * 100
        fft_he_plus = fft_results["humaneval_plus"]["pass@1"] * 100
    else:
        fft_he = None
        fft_he_plus = None

    if masked_results:
        masked_he = masked_results["humaneval"]["pass@1"] * 100
        masked_he_plus = masked_results["humaneval_plus"]["pass@1"] * 100
    else:
        masked_he = None
        masked_he_plus = None

    # Get masking info
    if mask_stats:
        masking_ratio = mask_stats["masking"]["masking_ratio"] * 100
        target_layers = mask_stats["verification"]["target_layers"]
        layer_range = f"{target_layers[0]}–{target_layers[-1]}"
    else:
        masking_ratio = 5.0  # Default
        layer_range = "20–23"

    # Print table
    print("\n┌─────────────────────────┬──────────────────┬──────────┬─────────┬────────────┬──────────────┐")
    print("│ Model                   │ Domain Data      │ Masking  │ Layers  │ HumanEval  │ HumanEval+   │")
    print("│                         │                  │ Ratio    │         │ Pass@1     │ Pass@1       │")
    print("├─────────────────────────┼──────────────────┼──────────┼─────────┼────────────┼──────────────┤")

    # FFT row
    fft_he_str = f"{fft_he:5.2f}%" if fft_he is not None else "    –"
    fft_he_plus_str = f"{fft_he_plus:5.2f}%" if fft_he_plus is not None else "    –"

    print(f"│ Qwen2-0.5B FFT          │ Evol+CA+Tulu3    │   0.0%   │    –    │ {fft_he_str:10s} │ {fft_he_plus_str:12s} │")
    print(f"│                         │ (10k ea)         │          │         │            │              │")

    # Masked row
    masked_he_str = f"{masked_he:5.2f}%" if masked_he is not None else "    –"
    masked_he_plus_str = f"{masked_he_plus:5.2f}%" if masked_he_plus is not None else "    –"

    print("├─────────────────────────┼──────────────────┼──────────┼─────────┼────────────┼──────────────┤")
    print(f"│ Qwen2-0.5B FFT+SCI      │ Evol+CA+Tulu3    │  {masking_ratio:4.1f}%   │ {layer_range:7s} │ {masked_he_str:10s} │ {masked_he_plus_str:12s} │")
    print(f"│                         │ (10k ea)         │          │ (linear)│            │              │")
    print("└─────────────────────────┴──────────────────┴──────────┴─────────┴────────────┴──────────────┘")

    # Calculate deltas if both available
    if fft_he is not None and masked_he is not None:
        delta_he = masked_he - fft_he
        delta_he_plus = masked_he_plus - fft_he_plus

        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│ Delta (FFT+SCI - FFT)                                                       │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        print(f"│ HumanEval Pass@1:   {delta_he:+6.2f} percentage points                              │")
        print(f"│ HumanEval+ Pass@1:  {delta_he_plus:+6.2f} percentage points                              │")
        print("└─────────────────────────────────────────────────────────────────────────────┘")

        # Interpretation
        print("\n" + "=" * 100)
        print("INTERPRETATION")
        print("=" * 100)

        if delta_he > 0:
            print("\n✓ SCI masking IMPROVED performance on HumanEval")
            print(f"  Masking {masking_ratio:.1f}% of parameters in layers {layer_range} increased pass@1 by {delta_he:.2f}pp")
        elif delta_he < -2.0:
            print("\n✗ SCI masking DEGRADED performance on HumanEval")
            print(f"  Masking {masking_ratio:.1f}% of parameters decreased pass@1 by {abs(delta_he):.2f}pp")
            print("\n  Possible mitigations:")
            print("  - Reduce masking ratio (try 2%)")
            print("  - Mask only MLP weights (exclude attention)")
            print("  - Try different layer range")
        else:
            print("\n≈ SCI masking had MINIMAL IMPACT on HumanEval")
            print(f"  Change in pass@1: {delta_he:.2f}pp (within noise margin)")


def main():
    parser = argparse.ArgumentParser(description="Compare FFT vs FFT+SCI results")

    parser.add_argument(
        "--fft_results",
        type=str,
        default="logs/results/fft_eval",
        help="Path to FFT evaluation results directory"
    )

    parser.add_argument(
        "--masked_results",
        type=str,
        default="logs/results/fft_plus_sci_mask_eval",
        help="Path to masked model evaluation results directory"
    )

    parser.add_argument(
        "--mask_stats",
        type=str,
        default="checkpoints/fft_plus_sci_mask/mask_stats.json",
        help="Path to mask statistics JSON"
    )

    args = parser.parse_args()

    # Load results
    fft_results_dir = Path(args.fft_results)
    masked_results_dir = Path(args.masked_results)
    mask_stats_path = Path(args.mask_stats)

    fft_results = load_results(fft_results_dir)
    masked_results = load_results(masked_results_dir)

    mask_stats = None
    if mask_stats_path.exists():
        with open(mask_stats_path, 'r') as f:
            mask_stats = json.load(f)

    # Check availability
    if fft_results is None:
        print(f"Warning: FFT results not found at {fft_results_dir}")
        print("Run: bash scripts/run_eval_fft.sh")

    if masked_results is None:
        print(f"Warning: Masked model results not found at {masked_results_dir}")
        print("Run: bash scripts/eval_masked.sh")

    if fft_results is None and masked_results is None:
        print("\nNo results available to compare. Please run evaluations first.")
        return

    # Print comparison
    print_comparison_table(fft_results, masked_results, mask_stats)

    print("\n")


if __name__ == "__main__":
    main()
