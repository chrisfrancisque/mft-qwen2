"""
Tokenization utilities for Qwen2 model.

Handles tokenizer loading, padding, and sequence length management.
"""

from typing import Dict, List
import torch
from transformers import AutoTokenizer


def load_qwen2_tokenizer(model_name: str = "Qwen/Qwen2-0.5B") -> AutoTokenizer:
    """
    Load Qwen2 tokenizer with proper configuration.

    Args:
        model_name: HuggingFace model name

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    # Ensure special tokens are set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def tokenize_example(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    return_full_text: bool = False
) -> dict:
    """
    Tokenize a single example.

    Args:
        example: Dict with 'text' field (full prompt + output)
        tokenizer: Qwen2 tokenizer
        max_length: Maximum sequence length
        return_full_text: Whether to keep 'text' field in output

    Returns:
        Dict with 'input_ids', 'attention_mask', and optionally 'text'
    """
    # Tokenize full text
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Don't pad individual examples
        return_tensors=None
    )

    result = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

    if return_full_text:
        result["text"] = example["text"]

    return result


def tokenize_dataset(
    examples: List[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    show_progress: bool = True
) -> List[dict]:
    """
    Tokenize a list of examples.

    Args:
        examples: List of dicts with 'text' field
        tokenizer: Qwen2 tokenizer
        max_length: Maximum sequence length
        show_progress: Show tqdm progress bar

    Returns:
        List of tokenized examples
    """
    from tqdm import tqdm

    tokenized = []

    iterator = tqdm(examples, desc="Tokenizing") if show_progress else examples

    for example in iterator:
        tokenized.append(tokenize_example(example, tokenizer, max_length))

    return tokenized


def collate_fn(
    batch: List[dict],
    pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of tokenized examples with padding.

    Args:
        batch: List of dicts with 'input_ids' and 'attention_mask'
        pad_token_id: Padding token ID

    Returns:
        Batched and padded tensors
    """
    # Find max length in batch
    max_len = max(len(ex["input_ids"]) for ex in batch)

    # Pad each example
    input_ids = []
    attention_mask = []

    for ex in batch:
        seq_len = len(ex["input_ids"])
        padding_len = max_len - seq_len

        # Pad input_ids
        padded_input = ex["input_ids"] + [pad_token_id] * padding_len
        input_ids.append(padded_input)

        # Pad attention_mask
        padded_mask = ex["attention_mask"] + [0] * padding_len
        attention_mask.append(padded_mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }


def filter_by_length(
    examples: List[dict],
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    min_length: int = 10
) -> List[dict]:
    """
    Filter examples by tokenized length.

    Args:
        examples: List of examples with 'text' field
        tokenizer: Qwen2 tokenizer
        max_length: Maximum allowed length
        min_length: Minimum allowed length

    Returns:
        Filtered list of examples
    """
    from tqdm import tqdm

    filtered = []

    for example in tqdm(examples, desc="Filtering by length"):
        tokenized = tokenizer(example["text"], add_special_tokens=True)
        length = len(tokenized["input_ids"])

        if min_length <= length <= max_length:
            filtered.append(example)

    print(f"Filtered: {len(filtered)}/{len(examples)} examples within [{min_length}, {max_length}] tokens")

    return filtered
