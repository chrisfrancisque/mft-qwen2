#!/usr/bin/env python3
"""
Prepare 5k training dataset in messages format (matching MFT repo).

This script creates a balanced dataset from 3 coding datasets:
- Evol CodeAlpaca
- Code-Alpaca
- Tulu 3 Persona Python

Output format uses 'messages' field with role/content structure,
compatible with tokenizer.apply_chat_template() for proper label masking.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from src.dataset import (
    normalize_evol_codealpaca,
    normalize_code_alpaca,
    normalize_tulu3_persona_python
)


def load_and_normalize(dataset_name: str, hf_path: str, max_samples: int, seed: int = 42):
    """Load dataset from HuggingFace and normalize to messages format."""
    print(f"\nLoading {dataset_name} from {hf_path}...")

    try:
        dataset = load_dataset(hf_path, split="train")
    except Exception as e:
        print(f"Error loading {hf_path}: {e}")
        raise

    print(f"  Loaded {len(dataset)} examples")

    # Shuffle and select
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Choose normalization function
    if dataset_name == "evol_codealpaca":
        normalize_fn = normalize_evol_codealpaca
    elif dataset_name == "code_alpaca":
        normalize_fn = normalize_code_alpaca
    elif dataset_name == "tulu3_persona_python":
        normalize_fn = normalize_tulu3_persona_python
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize to messages format
    normalized = []
    for ex in tqdm(dataset, desc=f"Normalizing {dataset_name}"):
        normalized.append(normalize_fn(ex))

    print(f"  Normalized {len(normalized)} examples")
    return normalized


def filter_by_message_length(examples: list, min_chars: int = 50, max_chars: int = 8000):
    """Filter examples by total message character length."""
    filtered = []
    for ex in examples:
        total_len = sum(len(m["content"]) for m in ex["messages"])
        if min_chars <= total_len <= max_chars:
            filtered.append(ex)

    print(f"  Filtered: {len(filtered)}/{len(examples)} examples within [{min_chars}, {max_chars}] chars")
    return filtered


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare 5k training dataset")
    parser.add_argument("--output_dir", type=str, default="data_processed",
                       help="Output directory")
    parser.add_argument("--train_samples", type=int, default=5000,
                       help="Number of training samples")
    parser.add_argument("--grad_samples", type=int, default=1000,
                       help="Number of gradient subset samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    print("=" * 80)
    print("PREPARING 5K TRAINING DATASET (MESSAGES FORMAT)")
    print("=" * 80)

    # Dataset configs
    datasets_config = [
        ("evol_codealpaca", "theblackcat102/evol-codealpaca-v1"),
        ("code_alpaca", "sahil2801/CodeAlpaca-20k"),
        ("tulu3_persona_python", "allenai/tulu-3-sft-personas-code"),
    ]

    # Calculate samples per dataset (balanced)
    train_per_dataset = args.train_samples // len(datasets_config) + 500  # Extra for filtering
    grad_per_dataset = args.grad_samples // len(datasets_config) + 200

    all_train = []
    all_grad = []

    for dataset_name, hf_path in datasets_config:
        # Load more than needed to account for filtering
        total_needed = train_per_dataset + grad_per_dataset
        examples = load_and_normalize(dataset_name, hf_path, total_needed, args.seed)

        # Filter by length
        examples = filter_by_message_length(examples)

        # Split into train and grad
        split_idx = min(train_per_dataset, len(examples) - grad_per_dataset)
        train_examples = examples[:split_idx]
        grad_examples = examples[split_idx:split_idx + grad_per_dataset]

        all_train.extend(train_examples)
        all_grad.extend(grad_examples)

        print(f"  {dataset_name}: {len(train_examples)} train, {len(grad_examples)} grad")

    # Shuffle combined datasets
    import random
    random.seed(args.seed)
    random.shuffle(all_train)
    random.shuffle(all_grad)

    # Trim to exact sizes
    all_train = all_train[:args.train_samples]
    all_grad = all_grad[:args.grad_samples]

    print(f"\nFinal dataset sizes:")
    print(f"  Training: {len(all_train)} examples")
    print(f"  Gradient subset: {len(all_grad)} examples")

    # Save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_5k.jsonl"
    grad_path = output_dir / "grad_subset_5k.jsonl"

    print(f"\nSaving to {output_dir}...")

    with open(train_path, 'w') as f:
        for ex in tqdm(all_train, desc="Writing train"):
            f.write(json.dumps(ex) + '\n')

    with open(grad_path, 'w') as f:
        for ex in tqdm(all_grad, desc="Writing grad"):
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved:")
    print(f"  {train_path}")
    print(f"  {grad_path}")

    # Print sample
    print("\n" + "=" * 80)
    print("SAMPLE EXAMPLE")
    print("=" * 80)
    sample = all_train[0]
    print(f"Source: {sample['source']}")
    print(f"Messages:")
    for msg in sample['messages']:
        content_preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
        print(f"  [{msg['role']}]: {content_preview}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
