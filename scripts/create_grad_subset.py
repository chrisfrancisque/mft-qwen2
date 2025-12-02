"""
Create gradient subset for SCI computation.

Takes 1000 samples from the training data for computing SCI scores.
"""

import sys
import json
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Paths
    input_path = Path("data_processed/train_1k_balanced.jsonl")
    output_path = Path("data_processed/grad_subset.jsonl")

    # Number of samples for gradient computation
    n_samples = 1000
    seed = 42

    print("=" * 60)
    print("Creating Gradient Subset for SCI")
    print("=" * 60)

    # Load training data
    print(f"\nLoading data from {input_path}...")
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"  Loaded {len(examples)} examples")

    # Sample
    random.seed(seed)
    if len(examples) > n_samples:
        sampled = random.sample(examples, n_samples)
    else:
        sampled = examples

    print(f"  Sampled {len(sampled)} examples for gradient computation")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in sampled:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {output_path}")

    # Show source distribution
    source_counts = {}
    for ex in sampled:
        src = ex.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("\nSource distribution:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
