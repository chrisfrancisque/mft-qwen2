"""
Tokenization utilities for Qwen2 model.

Handles tokenizer loading, padding, sequence length management,
and SFT encoding with proper label masking (matching MFT repo).
"""

from typing import Dict, List
import torch
from transformers import AutoTokenizer

# Tulu chat template (from MFT repo)
# This template:
# - Adds <|user|>, <|assistant|>, <|system|> markers
# - Adds EOS token after assistant responses
# - Supports add_generation_prompt for inference
TULU_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_qwen2_tokenizer(
    model_name: str = "Qwen/Qwen2-0.5B",
    use_tulu_template: bool = True
) -> AutoTokenizer:
    """
    Load Qwen2 tokenizer with proper configuration.

    Args:
        model_name: HuggingFace model name
        use_tulu_template: Whether to use Tulu chat template (matching MFT repo)

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

    # Set Tulu chat template (matching MFT repo)
    if use_tulu_template:
        tokenizer.chat_template = TULU_CHAT_TEMPLATE

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


def encode_sft_example(
    example: dict,
    tokenizer: AutoTokenizer,
    max_seq_length: int = 1024
) -> Dict[str, torch.Tensor]:
    """
    Encode a single SFT example with proper label masking.

    This function encodes examples matching MFT repo's approach:
    - Uses apply_chat_template for tokenization
    - Masks non-assistant tokens with -100 in labels
    - Only computes loss on assistant responses

    Args:
        example: Dict with 'messages' field (list of {"role": "user"|"assistant", "content": str})
        tokenizer: Qwen2 tokenizer with chat template set
        max_seq_length: Maximum sequence length

    Returns:
        Dict with 'input_ids', 'labels', 'attention_mask' (all as 1D tensors)
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # Tokenize full conversation
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )

    labels = input_ids.clone()

    # Mask the non-assistant parts for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # Calculate start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]

            # Calculate end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # For messages followed by assistant, use add_generation_prompt=True
                # to exclude the assistant prefix (e.g., '<|assistant|>') from the loss
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # For last message or message not followed by assistant
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]

            # Set labels to -100 for non-assistant tokens
            labels[:, message_start_idx:message_end_idx] = -100

            if max_seq_length and message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def collate_sft_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """
    Collate a batch of SFT examples with padding.

    Pads from the RIGHT (standard for training).

    Args:
        batch: List of dicts with 'input_ids', 'labels', 'attention_mask'
        pad_token_id: Padding token ID

    Returns:
        Batched and padded tensors
    """
    # Find max length in batch
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for ex in batch:
        seq_len = len(ex["input_ids"])
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        padded_input = torch.cat([
            ex["input_ids"],
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        # Pad labels with -100 (ignored in loss)
        padded_labels = torch.cat([
            ex["labels"],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)

        # Pad attention_mask with 0
        padded_mask = torch.cat([
            ex["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(padded_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
