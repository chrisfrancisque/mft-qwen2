"""
Entry point for HumanEval evaluation.

Evaluates a model checkpoint on HumanEval and HumanEval+.
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval_code import evaluate_model_on_code_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2 on HumanEval")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SCI config JSON (contains eval settings)"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    eval_config = config["evaluation"]

    # Paths
    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    # Run evaluation
    metrics = evaluate_model_on_code_benchmarks(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        eval_config=eval_config
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    for benchmark, results in metrics.items():
        print(f"\n{benchmark.upper()}:")
        print(f"  Pass@1: {results['pass@1']:.1%}")
        print(f"  Passed: {results['passed']}/{results['total_problems']}")


if __name__ == "__main__":
    main()
