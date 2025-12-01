"""
Prepare Fixed-Length Tokenized Dataset for TPU Training.

This script creates a pre-tokenized dataset with FIXED sequence lengths
to prevent XLA recompilation on TPU.

Key features:
- Fixed max_length=512 with padding="max_length"
- pad_to_multiple_of=64 for additional shape stability
- Saves pre-tokenized tensors to .pt file
- Balanced sampling: 334/333/333 from 3 datasets = 1000 total

Output:
- data_processed/train_1k_balanced.pt (pre-tokenized tensors)
- data_processed/train_1k_balanced.jsonl (raw text for reference)
"""

import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from transformers import AutoTokenizer


# Dataset configurations with balanced sampling
DATASETS = [
    {
        "name": "evol_codealpaca",
        "hf_path": "theblackcat102/evol-codealpaca-v1",
        "split": "train",
        "samples": 334,
    },
    {
        "name": "code_alpaca",
        "hf_path": "sahil2801/CodeAlpaca-20k",
        "split": "train",
        "samples": 333,
    },
    {
        "name": "tulu3_persona_python",
        "hf_path": "allenai/tulu-3-sft-personas-code",
        "split": "train",
        "samples": 333,
    },
]

# Fixed tokenization parameters (TPU-friendly)
MAX_LENGTH = 512
PAD_TO_MULTIPLE_OF = 64
SEED = 42


def normalize_example(example: dict, source: str) -> dict:
    """Normalize example to common schema based on source dataset."""

    if source == "tulu3_persona_python":
        # Tulu 3 uses messages format
        if "messages" in example:
            messages = example["messages"]
            user_msg = ""
            assistant_msg = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_msg = msg.get("content", "")
            return {
                "instruction": user_msg,
                "input": "",
                "output": assistant_msg,
                "source": source,
            }

    # Standard format for evol_codealpaca and code_alpaca
    return {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", ""),
        "source": source,
    }


def format_to_text(example: dict) -> str:
    """
    Format normalized example to training text.

    Uses SIMPLE code completion format (no chat template) to match HumanEval's
    evaluation format. The model learns to complete code directly without
    adding extra indentation or chat-style responses.
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # Simple completion format - no chat markers
    # This matches how HumanEval prompts the model (just code, continue it)
    if input_text and input_text.strip():
        # Include input context as a comment or docstring
        text = (
            f"# Task: {instruction}\n"
            f"# Input: {input_text}\n\n"
            f"{output}<|endoftext|>"
        )
    else:
        # For pure code tasks, just instruction + code
        text = (
            f"# Task: {instruction}\n\n"
            f"{output}<|endoftext|>"
        )

    return text


def load_and_sample_dataset(config: dict, seed: int) -> list:
    """Load dataset from HuggingFace and sample specified number of examples."""
    name = config["name"]
    hf_path = config["hf_path"]
    split = config["split"]
    n_samples = config["samples"]

    print(f"\nLoading {name} from {hf_path}...")
    dataset = load_dataset(hf_path, split=split)
    print(f"  Total available: {len(dataset)}")

    # Shuffle and sample
    dataset = dataset.shuffle(seed=seed)
    sampled = dataset.select(range(min(n_samples, len(dataset))))
    print(f"  Sampled: {len(sampled)}")

    # Normalize and format
    examples = []
    for ex in tqdm(sampled, desc=f"Processing {name}"):
        normalized = normalize_example(ex, name)
        text = format_to_text(normalized)
        examples.append({
            "text": text,
            "source": name,
        })

    return examples


def tokenize_fixed_length(
    examples: list,
    tokenizer: AutoTokenizer,
    max_length: int,
    pad_to_multiple_of: int,
) -> dict:
    """
    Tokenize all examples to FIXED length for TPU stability.

    This is the KEY fix for XLA recompilation:
    - All sequences padded to exactly max_length
    - Using pad_to_multiple_of for additional shape consistency
    - Returns stacked tensors ready for DataLoader
    """
    print(f"\nTokenizing {len(examples)} examples to fixed length {max_length}...")
    print(f"  pad_to_multiple_of: {pad_to_multiple_of}")

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    skipped = 0

    for ex in tqdm(examples, desc="Tokenizing"):
        text = ex["text"]

        # Tokenize with FIXED padding (the critical fix)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # KEY: pad to exact max_length
            pad_to_multiple_of=pad_to_multiple_of,  # Additional shape stability
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)  # [seq_len]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [seq_len]

        # Create labels: copy input_ids, mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    # Stack into single tensors [N, seq_len]
    result = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels),
    }

    print(f"\nTokenization complete:")
    print(f"  Shape: {result['input_ids'].shape}")
    print(f"  Skipped (too short): {skipped}")

    # Verify all shapes are identical (no recompilation triggers)
    assert result["input_ids"].shape == result["attention_mask"].shape == result["labels"].shape
    print(f"  All shapes verified identical: {result['input_ids'].shape}")

    return result


def main():
    print("=" * 80)
    print("PREPARING FIXED-LENGTH TOKENIZED DATASET FOR TPU")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Pad to multiple of: {PAD_TO_MULTIPLE_OF}")
    print(f"  Seed: {SEED}")
    print(f"  Total samples: {sum(d['samples'] for d in DATASETS)}")

    # Output paths
    output_dir = Path(__file__).parent.parent / "data_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / "train_1k_balanced.pt"
    jsonl_path = output_dir / "train_1k_balanced.jsonl"

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # Load and sample from all datasets
    all_examples = []
    for dataset_config in DATASETS:
        examples = load_and_sample_dataset(dataset_config, SEED)
        all_examples.extend(examples)

    print(f"\n{'='*80}")
    print(f"Total examples collected: {len(all_examples)}")
    print(f"{'='*80}")

    # Shuffle combined dataset
    import random
    random.seed(SEED)
    random.shuffle(all_examples)

    # Save raw JSONL for reference
    print(f"\nSaving raw JSONL to {jsonl_path}...")
    with open(jsonl_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"  Saved {len(all_examples)} examples")

    # Tokenize to fixed length
    tokenized = tokenize_fixed_length(
        all_examples,
        tokenizer,
        MAX_LENGTH,
        PAD_TO_MULTIPLE_OF,
    )

    # Save pre-tokenized tensors
    print(f"\nSaving pre-tokenized tensors to {pt_path}...")
    torch.save(tokenized, pt_path)

    # Print final summary
    print(f"\n{'='*80}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs:")
    print(f"  Tensors: {pt_path}")
    print(f"    - input_ids: {tokenized['input_ids'].shape}")
    print(f"    - attention_mask: {tokenized['attention_mask'].shape}")
    print(f"    - labels: {tokenized['labels'].shape}")
    print(f"  Raw JSONL: {jsonl_path}")

    # Distribution check
    print(f"\nSource distribution:")
    source_counts = {}
    for ex in all_examples:
        src = ex["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")

    print(f"\nThis dataset is ready for TPU training with ZERO XLA recompilation!")
    print(f"All sequences have identical shape: {tokenized['input_ids'].shape}")


if __name__ == "__main__":
    main()
