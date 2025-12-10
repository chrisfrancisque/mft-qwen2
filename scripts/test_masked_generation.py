"""
Test code generation with masked model.

Loads model, applies mask from mask_indices.json, and generates sample outputs.
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_mask_to_model(model, mask_indices):
    """Apply mask by zeroing selected parameters."""
    total_masked = 0
    for name, indices in mask_indices.items():
        for pname, param in model.named_parameters():
            if pname == name:
                flat_param = param.data.view(-1)
                for idx in indices:
                    if idx < len(flat_param):
                        flat_param[idx] = 0.0
                        total_masked += 1
                break
    return total_masked


def generate_code(model, tokenizer, prompt, max_new_tokens=256):
    """Generate code completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the completion
    if generated.startswith(prompt):
        completion = generated[len(prompt):]
    else:
        completion = generated

    return completion.strip()


# Test prompts (HumanEval-style)
TEST_PROMPTS = [
    '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
    '''def sum_list(numbers: list) -> int:
    """Return the sum of all numbers in the list.
    >>> sum_list([1, 2, 3])
    6
    >>> sum_list([])
    0
    """
''',
    '''def fibonacci(n: int) -> int:
    """Return the nth fibonacci number.
    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(10)
    55
    """
''',
]


def main():
    parser = argparse.ArgumentParser(description="Test masked model generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mask_indices", type=str, default=None, help="Path to mask_indices.json (optional)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    device = torch.device("cpu")  # Use CPU to avoid XLA issues
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()
    print("Model loaded")

    # Apply mask if provided
    if args.mask_indices:
        print(f"\nLoading mask from {args.mask_indices}...")
        with open(args.mask_indices, 'r') as f:
            mask_indices = json.load(f)

        total_masked = apply_mask_to_model(model, mask_indices)
        print(f"Applied mask: {total_masked:,} parameters zeroed")
    else:
        print("\nNo mask applied (unmasked model)")

    # Generate for each prompt
    print("\n" + "=" * 80)
    print("GENERATION RESULTS")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- Prompt {i} ---")
        print(prompt.strip()[:100] + "...")

        completion = generate_code(model, tokenizer, prompt, args.max_tokens)

        print(f"\n--- Completion {i} ---")
        print(completion[:500] if len(completion) > 500 else completion)
        print("-" * 40)


if __name__ == "__main__":
    main()
