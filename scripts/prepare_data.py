"""
Data Preparation Script

Loads, normalizes, splits, and formats all 3 coding datasets.

Output:
- data_processed/train_mixed.jsonl (30k examples)
- data_processed/grad_subset.jsonl (999 examples)
- data_processed/val_mixed.jsonl (optional, ~2k examples)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    load_and_normalize_dataset,
    split_dataset,
    apply_formatting,
    save_dataset_jsonl
)
from src.tokenization import load_qwen2_tokenizer, filter_by_length
from datasets import concatenate_datasets


def main():
    print("=" * 80)
    print("DATA PREPARATION FOR MFT-QWEN2")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "fft_qwen2_0.5b.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_config = config["data"]
    seed = data_config["seed"]
    max_length = config["model"]["max_length"]

    # Output paths
    output_dir = Path(__file__).parent.parent / "data_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_mixed.jsonl"
    grad_path = output_dir / "grad_subset.jsonl"
    val_path = output_dir / "val_mixed.jsonl"

    # Load tokenizer (for length filtering)
    print("\nLoading tokenizer...")
    tokenizer = load_qwen2_tokenizer(config["model"]["name"])

    # Process each dataset
    all_train = []
    all_grad = []
    all_val = []

    for dataset_config in data_config["datasets"]:
        dataset_name = dataset_config["name"]
        hf_path = dataset_config["hf_path"]
        split = dataset_config["split"]
        train_size = dataset_config["train_samples"]
        grad_size = dataset_config["grad_samples"]

        # Optional validation size (proportional)
        val_size = int(train_size * 0.07)  # ~7% for validation (~700 per dataset)

        print(f"\n{'='*80}")
        print(f"Processing {dataset_name}")
        print(f"{'='*80}")

        # Load and normalize
        dataset = load_and_normalize_dataset(
            dataset_name=dataset_name,
            hf_path=hf_path,
            split=split,
            seed=seed
        )

        # Split
        print(f"\nSplitting dataset:")
        print(f"  Train: {train_size}")
        print(f"  Gradient: {grad_size}")
        print(f"  Validation: {val_size}")

        train_ds, grad_ds, val_ds = split_dataset(
            dataset,
            train_size=train_size,
            grad_size=grad_size,
            val_size=val_size
        )

        # Apply formatting
        print(f"\nApplying prompt formatting...")

        train_formatted = train_ds.map(apply_formatting, desc="Formatting train")
        grad_formatted = grad_ds.map(apply_formatting, desc="Formatting grad")
        val_formatted = val_ds.map(apply_formatting, desc="Formatting val") if val_ds else None

        # Convert to list of dicts
        all_train.extend(list(train_formatted))
        all_grad.extend(list(grad_formatted))
        if val_formatted:
            all_val.extend(list(val_formatted))

        print(f"  Train: {len(train_formatted)} examples")
        print(f"  Grad: {len(grad_formatted)} examples")
        if val_formatted:
            print(f"  Val: {val_formatted} examples")

    # Combine all datasets
    print(f"\n{'='*80}")
    print("Combining datasets")
    print(f"{'='*80}")
    print(f"  Total train: {len(all_train)}")
    print(f"  Total grad: {len(all_grad)}")
    print(f"  Total val: {len(all_val)}")

    # Optional: Filter by length
    print(f"\n{'='*80}")
    print("Filtering by length")
    print(f"{'='*80}")

    all_train = filter_by_length(all_train, tokenizer, max_length=max_length)
    all_grad = filter_by_length(all_grad, tokenizer, max_length=max_length)
    if all_val:
        all_val = filter_by_length(all_val, tokenizer, max_length=max_length)

    # Save
    print(f"\n{'='*80}")
    print("Saving processed datasets")
    print(f"{'='*80}")

    # Save train
    with open(train_path, 'w') as f:
        for example in all_train:
            f.write(json.dumps(example) + '\n')
    print(f"Saved {len(all_train)} train examples to {train_path}")

    # Save grad
    with open(grad_path, 'w') as f:
        for example in all_grad:
            f.write(json.dumps(example) + '\n')
    print(f"Saved {len(all_grad)} grad examples to {grad_path}")

    # Save val
    if all_val:
        with open(val_path, 'w') as f:
            for example in all_val:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(all_val)} val examples to {val_path}")

    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*80}")

    # Print summary
    print("\nSummary:")
    print(f"  Train: {train_path} ({len(all_train)} examples)")
    print(f"  Grad:  {grad_path} ({len(all_grad)} examples)")
    if all_val:
        print(f"  Val:   {val_path} ({len(all_val)} examples)")


if __name__ == "__main__":
    main()
