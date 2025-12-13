"""
Evaluate perplexity of fine-tuned vs masked models on TPU.

Perplexity = exp(average cross-entropy loss)

Usage:
    # Compare FFT model vs masked model
    python scripts/eval_perplexity.py \
        --fft_checkpoint /path/to/fft/checkpoint \
        --masked_checkpoint /path/to/masked/checkpoint \
        --eval_data /path/to/eval_data.jsonl \
        --max_examples 100

    # Just evaluate one model
    python scripts/eval_perplexity.py \
        --fft_checkpoint /path/to/checkpoint \
        --eval_data /path/to/eval_data.jsonl
"""

import sys
import json
import argparse
import math
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils_xla import get_device, is_master, print_once, is_tpu_available, mark_step


def load_eval_examples(data_path: Path, max_examples: int = None) -> list:
    """Load evaluation examples from jsonl file."""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Support both 'text' and 'prompt'/'completion' formats
            if 'text' in item:
                examples.append(item)
            elif 'prompt' in item and 'completion' in item:
                examples.append({'text': item['prompt'] + item['completion']})
            elif 'input' in item and 'output' in item:
                examples.append({'text': item['input'] + item['output']})

            if max_examples and len(examples) >= max_examples:
                break

    return examples


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 1,
    use_cpu: bool = False
) -> dict:
    """
    Compute perplexity over examples.

    Returns:
        Dict with perplexity, loss, and per-example stats
    """
    if use_cpu:
        print_once("Using CPU for evaluation")
        eval_device = torch.device('cpu')
        model = model.to(eval_device).float()
    else:
        eval_device = device

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    per_example_ppl = []

    print_once(f"\nEvaluating on {len(examples)} examples...")

    with torch.no_grad():
        for i, ex in enumerate(tqdm(examples, desc="Computing perplexity", disable=not is_master())):
            text = ex['text']

            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=False
            ).to(eval_device)

            # Skip very short sequences
            if inputs['input_ids'].shape[1] < 2:
                continue

            # Prepare labels (shift happens inside model)
            labels = inputs['input_ids'].clone()

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()

            # Count tokens (excluding first token which has no prediction)
            num_tokens = inputs['input_ids'].shape[1] - 1

            # Accumulate
            total_loss += loss * num_tokens
            total_tokens += num_tokens

            # Per-example perplexity
            example_ppl = math.exp(loss)
            per_example_ppl.append({
                'index': i,
                'loss': loss,
                'perplexity': example_ppl,
                'num_tokens': num_tokens
            })

            # Mark step for XLA
            if is_tpu_available() and not use_cpu:
                mark_step()

            # Progress update
            if (i + 1) % 20 == 0:
                running_ppl = math.exp(total_loss / total_tokens)
                print_once(f"  [{i+1}/{len(examples)}] Running perplexity: {running_ppl:.2f}")

    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
        'num_examples': len(per_example_ppl),
        'per_example': per_example_ppl
    }


def load_model(checkpoint_path: str, device: torch.device, use_cpu: bool = False):
    """Load model and tokenizer from checkpoint."""
    print_once(f"Loading model from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    if use_cpu:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    else:
        dtype = torch.bfloat16 if str(device).startswith('xla') else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model = model.to(device)

    print_once(f"Model loaded: {model.config.num_hidden_layers} layers, {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of models")

    parser.add_argument(
        "--fft_checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned (FFT) model checkpoint"
    )

    parser.add_argument(
        "--masked_checkpoint",
        type=str,
        default=None,
        help="Path to masked model checkpoint (optional, for comparison)"
    )

    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation data (jsonl with 'text' field)"
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate (default: 100)"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (optional)"
    )

    parser.add_argument(
        "--cpu_eval",
        action="store_true",
        help="Use CPU for evaluation (avoids XLA compilation overhead)"
    )

    args = parser.parse_args()

    # Get device
    device = get_device()
    print_once(f"Device: {device}")

    # Load evaluation data
    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {eval_path}")

    examples = load_eval_examples(eval_path, args.max_examples)
    print_once(f"Loaded {len(examples)} evaluation examples")

    results = {
        'timestamp': datetime.now().isoformat(),
        'eval_data': str(eval_path),
        'max_examples': args.max_examples,
        'max_length': args.max_length,
    }

    # =========================================================================
    # Evaluate FFT model
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("EVALUATING FFT MODEL")
    print_once("=" * 80)

    model_fft, tokenizer = load_model(args.fft_checkpoint, device, args.cpu_eval)

    fft_results = compute_perplexity(
        model_fft, tokenizer, examples, device,
        max_length=args.max_length,
        use_cpu=args.cpu_eval
    )

    print_once(f"\nFFT Model Results:")
    print_once(f"  Perplexity: {fft_results['perplexity']:.2f}")
    print_once(f"  Avg Loss: {fft_results['avg_loss']:.4f}")
    print_once(f"  Total Tokens: {fft_results['total_tokens']:,}")

    results['fft'] = {
        'checkpoint': args.fft_checkpoint,
        'perplexity': fft_results['perplexity'],
        'avg_loss': fft_results['avg_loss'],
        'total_tokens': fft_results['total_tokens'],
        'num_examples': fft_results['num_examples']
    }

    # Free memory
    del model_fft
    if is_tpu_available():
        mark_step()

    # =========================================================================
    # Evaluate masked model (if provided)
    # =========================================================================
    if args.masked_checkpoint:
        print_once("\n" + "=" * 80)
        print_once("EVALUATING MASKED MODEL")
        print_once("=" * 80)

        model_masked, _ = load_model(args.masked_checkpoint, device, args.cpu_eval)

        masked_results = compute_perplexity(
            model_masked, tokenizer, examples, device,
            max_length=args.max_length,
            use_cpu=args.cpu_eval
        )

        print_once(f"\nMasked Model Results:")
        print_once(f"  Perplexity: {masked_results['perplexity']:.2f}")
        print_once(f"  Avg Loss: {masked_results['avg_loss']:.4f}")
        print_once(f"  Total Tokens: {masked_results['total_tokens']:,}")

        results['masked'] = {
            'checkpoint': args.masked_checkpoint,
            'perplexity': masked_results['perplexity'],
            'avg_loss': masked_results['avg_loss'],
            'total_tokens': masked_results['total_tokens'],
            'num_examples': masked_results['num_examples']
        }

        # Comparison
        ppl_diff = masked_results['perplexity'] - fft_results['perplexity']
        ppl_pct = (ppl_diff / fft_results['perplexity']) * 100

        results['comparison'] = {
            'perplexity_diff': ppl_diff,
            'perplexity_pct_change': ppl_pct,
            'loss_diff': masked_results['avg_loss'] - fft_results['avg_loss']
        }

        del model_masked

    # =========================================================================
    # Summary
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("SUMMARY")
    print_once("=" * 80)

    print_once(f"\nFFT Model:    PPL = {results['fft']['perplexity']:.2f}")

    if args.masked_checkpoint:
        print_once(f"Masked Model: PPL = {results['masked']['perplexity']:.2f}")
        print_once(f"\nDifference:   {ppl_diff:+.2f} ({ppl_pct:+.1f}%)")

        if ppl_diff > 0:
            print_once("=> Masked model has HIGHER perplexity (worse)")
        else:
            print_once("=> Masked model has LOWER perplexity (better)")

    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "perplexity_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print_once(f"\nResults saved to {results_path}")

    print_once(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
