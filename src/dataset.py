"""
Dataset loading and normalization for coding tasks.

Handles 3 datasets:
- Evol CodeAlpaca
- Code-Alpaca
- Tulu 3 Persona Python

Normalizes to messages format (matching MFT repo) for proper label masking.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset
from tqdm import tqdm


def normalize_evol_codealpaca(example: dict) -> dict:
    """Normalize Evol CodeAlpaca to messages format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Build user content
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_content = instruction

    return {
        "source": "evol_codealpaca",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def normalize_code_alpaca(example: dict) -> dict:
    """Normalize Code-Alpaca to messages format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Build user content
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        user_content = instruction

    return {
        "source": "code_alpaca",
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def normalize_tulu3_persona_python(example: dict) -> dict:
    """
    Normalize Tulu 3 Persona Python to messages format.

    Tulu 3 already uses 'messages' format with role/content.
    """
    if "messages" in example:
        # Already in messages format, just pass through
        return {
            "source": "tulu3_persona_python",
            "messages": example["messages"]
        }
    else:
        # Fallback for other formats
        instruction = example.get("instruction", example.get("prompt", ""))
        output = example.get("output", example.get("completion", ""))

        return {
            "source": "tulu3_persona_python",
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
        }


def load_and_normalize_dataset(
    dataset_name: str,
    hf_path: str,
    split: str = "train",
    seed: int = 42
) -> Dataset:
    """
    Load a dataset from HuggingFace and normalize to common schema.

    Args:
        dataset_name: One of 'evol_codealpaca', 'code_alpaca', 'tulu3_persona_python'
        hf_path: HuggingFace dataset path
        split: Dataset split to load
        seed: Random seed for shuffling

    Returns:
        Normalized HuggingFace Dataset
    """
    print(f"\nLoading {dataset_name} from {hf_path}...")

    # Load dataset
    try:
        dataset = load_dataset(hf_path, split=split)
    except Exception as e:
        print(f"Error loading {hf_path}: {e}")
        raise

    print(f"  Loaded {len(dataset)} examples")

    # Choose normalization function
    if dataset_name == "evol_codealpaca":
        normalize_fn = normalize_evol_codealpaca
    elif dataset_name == "code_alpaca":
        normalize_fn = normalize_code_alpaca
    elif dataset_name == "tulu3_persona_python":
        normalize_fn = normalize_tulu3_persona_python
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize
    print(f"  Normalizing {dataset_name}...")
    normalized = dataset.map(
        normalize_fn,
        desc=f"Normalizing {dataset_name}"
    )

    # Shuffle with seed
    normalized = normalized.shuffle(seed=seed)

    return normalized


def split_dataset(
    dataset: Dataset,
    train_size: int,
    grad_size: int,
    val_size: int = 0
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, gradient subset, and optional validation.

    Args:
        dataset: Shuffled normalized dataset
        train_size: Number of training examples
        grad_size: Number of gradient subset examples
        val_size: Number of validation examples (optional)

    Returns:
        (train_dataset, grad_dataset, val_dataset)
    """
    total_needed = train_size + grad_size + val_size

    if len(dataset) < total_needed:
        raise ValueError(
            f"Dataset has {len(dataset)} examples but need {total_needed} "
            f"(train={train_size}, grad={grad_size}, val={val_size})"
        )

    # Split indices
    train_end = train_size
    grad_end = train_size + grad_size
    val_end = train_size + grad_size + val_size

    train_dataset = dataset.select(range(0, train_end))
    grad_dataset = dataset.select(range(train_end, grad_end))

    if val_size > 0:
        val_dataset = dataset.select(range(grad_end, val_end))
    else:
        val_dataset = None

    return train_dataset, grad_dataset, val_dataset




def save_dataset_jsonl(dataset: Dataset, output_path: Path):
    """Save dataset to JSONL format."""
    print(f"Saving {len(dataset)} examples to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for example in tqdm(dataset, desc="Writing JSONL"):
            f.write(json.dumps(example) + '\n')

    print(f"  Saved to {output_path}")


def load_dataset_jsonl(input_path: Path) -> List[dict]:
    """Load dataset from JSONL format."""
    print(f"Loading dataset from {input_path}...")

    examples = []
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Reading JSONL"):
            examples.append(json.loads(line))

    print(f"  Loaded {len(examples)} examples")
    return examples
