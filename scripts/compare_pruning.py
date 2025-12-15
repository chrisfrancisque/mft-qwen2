"""
Compare SCI vs WANDA pruning approaches.

This script:
1. Loads both checkpoints (SCI-masked and WANDA-pruned)
2. Analyzes the sparsity patterns
3. Checks overlap between pruned weights
4. Evaluates loss/perplexity on each model
5. Generates code samples from each to compare quality
"""

import argparse
import json
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(module: nn.Module, layers: List = [nn.Linear], name: str = '') -> Dict[str, nn.Module]:
    """Recursively find layers of specified types."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_zero_mask(model: AutoModelForCausalLM) -> Dict[str, torch.Tensor]:
    """Get boolean mask of zero weights for each linear layer."""
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) == 2:  # Linear layer weights
            masks[name] = (param.data == 0)
    return masks


def analyze_sparsity(model: AutoModelForCausalLM, name: str) -> Dict:
    """Analyze sparsity patterns in a model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print('='*60)

    layers = model.model.layers
    total_zeros = 0
    total_params = 0

    layer_stats = []

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_zeros = 0
        layer_params = 0

        for lname in subset:
            W = subset[lname].weight.data
            zeros = (W == 0).sum().item()
            params = W.numel()
            layer_zeros += zeros
            layer_params += params

        total_zeros += layer_zeros
        total_params += layer_params
        sparsity = layer_zeros / layer_params if layer_params > 0 else 0
        layer_stats.append({
            'layer': i,
            'zeros': layer_zeros,
            'total': layer_params,
            'sparsity': sparsity
        })

        if sparsity > 0:
            print(f"  Layer {i:2d}: {sparsity*100:6.2f}% sparse ({layer_zeros:,} / {layer_params:,})")

    overall = total_zeros / total_params if total_params > 0 else 0
    print(f"\n  Overall: {overall*100:.2f}% sparse ({total_zeros:,} / {total_params:,})")

    return {
        'name': name,
        'total_zeros': total_zeros,
        'total_params': total_params,
        'overall_sparsity': overall,
        'layer_stats': layer_stats
    }


def compare_masks(mask1: Dict[str, torch.Tensor], mask2: Dict[str, torch.Tensor],
                  name1: str, name2: str) -> Dict:
    """Compare which weights are zeroed in each model."""
    print(f"\n{'='*60}")
    print(f"Comparing zero masks: {name1} vs {name2}")
    print('='*60)

    common_keys = set(mask1.keys()) & set(mask2.keys())

    total_only1 = 0
    total_only2 = 0
    total_both = 0
    total_neither = 0

    for key in sorted(common_keys):
        m1 = mask1[key]
        m2 = mask2[key]

        only1 = (m1 & ~m2).sum().item()  # Zero only in model1
        only2 = (~m1 & m2).sum().item()  # Zero only in model2
        both = (m1 & m2).sum().item()     # Zero in both
        neither = (~m1 & ~m2).sum().item() # Non-zero in both

        total_only1 += only1
        total_only2 += only2
        total_both += both
        total_neither += neither

    total = total_only1 + total_only2 + total_both + total_neither

    print(f"\n  Zero only in {name1}: {total_only1:,} ({total_only1/total*100:.4f}%)")
    print(f"  Zero only in {name2}: {total_only2:,} ({total_only2/total*100:.4f}%)")
    print(f"  Zero in both:         {total_both:,} ({total_both/total*100:.4f}%)")
    print(f"  Non-zero in both:     {total_neither:,} ({total_neither/total*100:.4f}%)")

    # Calculate overlap coefficient (Jaccard similarity of zero sets)
    if total_only1 + total_only2 + total_both > 0:
        jaccard = total_both / (total_only1 + total_only2 + total_both)
    else:
        jaccard = 0

    print(f"\n  Jaccard similarity of zero masks: {jaccard:.4f}")

    return {
        f'only_{name1}': total_only1,
        f'only_{name2}': total_only2,
        'both': total_both,
        'neither': total_neither,
        'jaccard': jaccard
    }


@torch.no_grad()
def evaluate_perplexity(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                        device: torch.device, num_samples: int = 50) -> Dict:
    """Evaluate model perplexity on wikitext-2."""
    print("  Loading evaluation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Tokenize samples
    eval_texts = [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]

    total_loss = 0.0
    total_tokens = 0

    model.eval()

    for text in tqdm(eval_texts, desc="  Evaluating", leave=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)

        if input_ids.shape[1] < 2:
            continue

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

        seq_len = input_ids.shape[1]
        total_loss += loss * (seq_len - 1)
        total_tokens += seq_len - 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens
    }


def generate_sample(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                   prompt: str, max_new_tokens: int = 150) -> str:
    """Generate a code sample from the model."""
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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


def main():
    parser = argparse.ArgumentParser(description="Compare SCI vs WANDA pruning")
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline (unpruned) checkpoint")
    parser.add_argument("--sci", type=str, required=True,
                       help="Path to SCI-masked checkpoint")
    parser.add_argument("--wanda", type=str, required=True,
                       help="Path to WANDA-pruned checkpoint")
    parser.add_argument("--output", type=str, default="pruning_comparison.json",
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu recommended to avoid memory issues)")
    parser.add_argument("--generate", action="store_true",
                       help="Generate code samples from each model")

    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16

    print("="*60)
    print("SCI vs WANDA PRUNING COMPARISON")
    print("="*60)

    # Load models
    print("\nLoading models...")

    print(f"  Loading baseline from {args.baseline}...")
    tokenizer = AutoTokenizer.from_pretrained(args.baseline, trust_remote_code=True)
    baseline = AutoModelForCausalLM.from_pretrained(
        args.baseline, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    print(f"  Loading SCI from {args.sci}...")
    sci_model = AutoModelForCausalLM.from_pretrained(
        args.sci, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    print(f"  Loading WANDA from {args.wanda}...")
    wanda_model = AutoModelForCausalLM.from_pretrained(
        args.wanda, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # Analyze sparsity
    baseline_stats = analyze_sparsity(baseline, "Baseline (FFT)")
    sci_stats = analyze_sparsity(sci_model, "SCI Masked")
    wanda_stats = analyze_sparsity(wanda_model, "WANDA Pruned")

    # Get zero masks
    print("\nExtracting zero masks...")
    baseline_mask = get_zero_mask(baseline)
    sci_mask = get_zero_mask(sci_model)
    wanda_mask = get_zero_mask(wanda_model)

    # Compare masks
    sci_vs_wanda = compare_masks(sci_mask, wanda_mask, "SCI", "WANDA")
    sci_vs_baseline = compare_masks(sci_mask, baseline_mask, "SCI", "Baseline")
    wanda_vs_baseline = compare_masks(wanda_mask, baseline_mask, "WANDA", "Baseline")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print('='*60)

    print(f"\n{'Metric':<30} {'Baseline':>12} {'SCI':>12} {'WANDA':>12}")
    print("-"*66)
    print(f"{'Total zeros':<30} {baseline_stats['total_zeros']:>12,} {sci_stats['total_zeros']:>12,} {wanda_stats['total_zeros']:>12,}")
    print(f"{'Sparsity %':<30} {baseline_stats['overall_sparsity']*100:>11.4f}% {sci_stats['overall_sparsity']*100:>11.4f}% {wanda_stats['overall_sparsity']*100:>11.4f}%")

    # Calculate ratio
    if sci_stats['total_zeros'] > 0:
        wanda_to_sci_ratio = wanda_stats['total_zeros'] / sci_stats['total_zeros']
        print(f"\nWANDA removes {wanda_to_sci_ratio:.1f}x more weights than SCI")

    # Evaluate perplexity on each model
    print(f"\n{'='*60}")
    print("PERPLEXITY EVALUATION (WikiText-2)")
    print('='*60)

    print("\nEvaluating baseline...")
    baseline_ppl = evaluate_perplexity(baseline, tokenizer, device)
    print(f"  Baseline - Loss: {baseline_ppl['loss']:.4f}, PPL: {baseline_ppl['perplexity']:.2f}")

    print("\nEvaluating SCI...")
    sci_ppl = evaluate_perplexity(sci_model, tokenizer, device)
    print(f"  SCI      - Loss: {sci_ppl['loss']:.4f}, PPL: {sci_ppl['perplexity']:.2f}")

    print("\nEvaluating WANDA...")
    wanda_ppl = evaluate_perplexity(wanda_model, tokenizer, device)
    print(f"  WANDA    - Loss: {wanda_ppl['loss']:.4f}, PPL: {wanda_ppl['perplexity']:.2f}")

    # Print perplexity comparison table
    print(f"\n{'='*60}")
    print("PERPLEXITY COMPARISON")
    print('='*60)
    print(f"\n{'Model':<20} {'Loss':>12} {'Perplexity':>15} {'PPL Increase':>15}")
    print("-"*62)
    print(f"{'Baseline':<20} {baseline_ppl['loss']:>12.4f} {baseline_ppl['perplexity']:>15.2f} {'-':>15}")

    sci_ppl_increase = (sci_ppl['perplexity'] / baseline_ppl['perplexity'] - 1) * 100 if baseline_ppl['perplexity'] > 0 else 0
    wanda_ppl_increase = (wanda_ppl['perplexity'] / baseline_ppl['perplexity'] - 1) * 100 if baseline_ppl['perplexity'] > 0 else 0

    print(f"{'SCI':<20} {sci_ppl['loss']:>12.4f} {sci_ppl['perplexity']:>15.2f} {f'+{sci_ppl_increase:.1f}%':>15}")
    print(f"{'WANDA':<20} {wanda_ppl['loss']:>12.4f} {wanda_ppl['perplexity']:>15.2f} {f'+{wanda_ppl_increase:.1f}%':>15}")

    # Generate samples if requested
    generation_results = {}
    if args.generate:
        print(f"\n{'='*60}")
        print("CODE GENERATION COMPARISON")
        print('='*60)

        prompts = [
            "def fibonacci(n):\n    \"\"\"Return the nth fibonacci number.\"\"\"\n",
            "def is_prime(n):\n    \"\"\"Check if n is a prime number.\"\"\"\n",
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt[:50]}...")

            print("\n  Baseline:")
            baseline_out = generate_sample(baseline, tokenizer, prompt)
            print(f"    {baseline_out[:200]}...")

            print("\n  SCI:")
            sci_out = generate_sample(sci_model, tokenizer, prompt)
            print(f"    {sci_out[:200]}...")

            print("\n  WANDA:")
            wanda_out = generate_sample(wanda_model, tokenizer, prompt)
            print(f"    {wanda_out[:200]}...")

            generation_results[prompt[:30]] = {
                'baseline': baseline_out,
                'sci': sci_out,
                'wanda': wanda_out
            }

    # Save results
    results = {
        'baseline': baseline_stats,
        'sci': sci_stats,
        'wanda': wanda_stats,
        'comparisons': {
            'sci_vs_wanda': sci_vs_wanda,
            'sci_vs_baseline': sci_vs_baseline,
            'wanda_vs_baseline': wanda_vs_baseline
        },
        'perplexity': {
            'baseline': baseline_ppl,
            'sci': sci_ppl,
            'wanda': wanda_ppl,
            'sci_ppl_increase_pct': sci_ppl_increase,
            'wanda_ppl_increase_pct': wanda_ppl_increase
        },
        'summary': {
            'wanda_to_sci_ratio': wanda_stats['total_zeros'] / max(1, sci_stats['total_zeros']),
            'wanda_sparsity': wanda_stats['overall_sparsity'],
            'sci_sparsity': sci_stats['overall_sparsity'],
            'baseline_ppl': baseline_ppl['perplexity'],
            'sci_ppl': sci_ppl['perplexity'],
            'wanda_ppl': wanda_ppl['perplexity'],
        }
    }

    if args.generate:
        results['generation'] = generation_results

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print('='*60)


if __name__ == "__main__":
    main()
