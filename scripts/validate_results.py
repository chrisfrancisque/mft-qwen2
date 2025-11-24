"""
Deep validation of masked model results.
Investigates whether the improvement is real and what's causing it.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import numpy as np

def load_eval_results(results_dir: Path):
    """Load evaluation results."""
    with open(results_dir / "eval_metrics.json") as f:
        return json.load(f)

def compare_predictions(baseline_dir: Path, masked_dir: Path):
    """Compare prediction details between baseline and masked."""

    # Load both results
    baseline_results = load_eval_results(baseline_dir)
    masked_results = load_eval_results(masked_dir)

    print("\n" + "="*80)
    print("DETAILED COMPARISON: Baseline vs 1% Masked Model")
    print("="*80)

    # Overall metrics
    print("\n1. OVERALL METRICS:")
    print("-" * 80)
    for benchmark in ["humaneval", "humanevalplus"]:
        if benchmark in baseline_results and benchmark in masked_results:
            b_pass = baseline_results[benchmark].get("pass@1", 0)
            m_pass = masked_results[benchmark].get("pass@1", 0)
            print(f"\n{benchmark.upper()}:")
            print(f"  Baseline:  {b_pass:.1f}%")
            print(f"  Masked:    {m_pass:.1f}%")
            print(f"  Delta:     {m_pass - b_pass:+.1f}%")

    # Check if we have detailed predictions
    baseline_preds_file = baseline_dir / "humaneval_predictions.jsonl"
    masked_preds_file = masked_dir / "humaneval_predictions.jsonl"

    if baseline_preds_file.exists() and masked_preds_file.exists():
        print("\n2. PROBLEM-BY-PROBLEM ANALYSIS:")
        print("-" * 80)

        # Load predictions
        baseline_preds = {}
        with open(baseline_preds_file) as f:
            for line in f:
                item = json.loads(line)
                baseline_preds[item["task_id"]] = item

        masked_preds = {}
        with open(masked_preds_file) as f:
            for line in f:
                item = json.loads(line)
                masked_preds[item["task_id"]] = item

        # Categorize outcomes
        both_correct = []
        both_wrong = []
        baseline_only = []  # Baseline correct, masked wrong
        masked_only = []    # Masked correct, baseline wrong

        for task_id in baseline_preds.keys():
            if task_id not in masked_preds:
                continue

            b_passed = baseline_preds[task_id].get("passed", False)
            m_passed = masked_preds[task_id].get("passed", False)

            if b_passed and m_passed:
                both_correct.append(task_id)
            elif not b_passed and not m_passed:
                both_wrong.append(task_id)
            elif b_passed and not m_passed:
                baseline_only.append(task_id)
            elif not b_passed and m_passed:
                masked_only.append(task_id)

        print(f"\nBoth models correct: {len(both_correct)} problems")
        print(f"Both models wrong: {len(both_wrong)} problems")
        print(f"Only baseline correct: {len(baseline_only)} problems")
        print(f"Only masked correct: {len(masked_only)} problems")

        print(f"\nNet improvement: {len(masked_only) - len(baseline_only)} problems")

        # Show examples of improvements
        if masked_only:
            print(f"\n3. PROBLEMS FIXED BY MASKING (sample):")
            print("-" * 80)
            for task_id in masked_only[:5]:  # Show first 5
                print(f"\n{task_id}:")
                print(f"  Baseline: FAILED")
                print(f"  Masked:   PASSED")

                # Show code samples if available
                if "completion" in masked_preds[task_id]:
                    print(f"\n  Masked model's solution (first 200 chars):")
                    completion = masked_preds[task_id]["completion"]
                    print(f"  {completion[:200]}...")

        # Show examples of regressions
        if baseline_only:
            print(f"\n4. PROBLEMS BROKEN BY MASKING (sample):")
            print("-" * 80)
            for task_id in baseline_only[:5]:  # Show first 5
                print(f"\n{task_id}:")
                print(f"  Baseline: PASSED")
                print(f"  Masked:   FAILED")

    return baseline_results, masked_results

def analyze_masked_parameters(mask_indices_path: Path, model_path: Path):
    """Analyze which parameters were masked and their patterns."""

    print("\n" + "="*80)
    print("MASKED PARAMETERS ANALYSIS")
    print("="*80)

    # Load mask indices
    with open(mask_indices_path) as f:
        mask_indices = json.load(f)

    print(f"\nTotal parameters masked: {sum(len(indices) for indices in mask_indices.values())}")

    # Analyze by parameter type
    param_type_counts = defaultdict(int)
    for param_name, indices in mask_indices.items():
        # Extract parameter type (q_proj, k_proj, v_proj, etc.)
        if "q_proj" in param_name:
            param_type_counts["q_proj"] += len(indices)
        elif "k_proj" in param_name:
            param_type_counts["k_proj"] += len(indices)
        elif "v_proj" in param_name:
            param_type_counts["v_proj"] += len(indices)
        elif "o_proj" in param_name:
            param_type_counts["o_proj"] += len(indices)
        elif "gate_proj" in param_name:
            param_type_counts["gate_proj"] += len(indices)
        elif "up_proj" in param_name:
            param_type_counts["up_proj"] += len(indices)
        elif "down_proj" in param_name:
            param_type_counts["down_proj"] += len(indices)

    print("\nMasking by parameter type:")
    print("-" * 80)
    for param_type, count in sorted(param_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param_type}: {count:,} parameters masked")

    # Load model to check actual masking
    print(f"\nLoading model from {model_path} to verify masking...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print("\nVerifying masking per layer:")
    print("-" * 80)
    for layer_idx in [15, 16, 17]:
        layer_zeros = 0
        layer_total = 0

        for name, param in model.named_parameters():
            if f"layers.{layer_idx}." in name:
                zeros = (param == 0).sum().item()
                total = param.numel()
                layer_zeros += zeros
                layer_total += total

                if zeros > 0:
                    print(f"  Layer {layer_idx} - {name.split('.')[-2]}.{name.split('.')[-1]}: "
                          f"{zeros/total*100:.2f}% zeros ({zeros:,}/{total:,})")

        if layer_total > 0:
            print(f"\n  Layer {layer_idx} total: {layer_zeros/layer_total*100:.2f}% zeros\n")

def check_model_quality(model_path: Path, tokenizer_path: Path):
    """Check basic model generation quality."""

    print("\n" + "="*80)
    print("MODEL GENERATION QUALITY CHECK")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Simple test prompts
    test_prompts = [
        "def fibonacci(n):\n    ",
        "def is_prime(n):\n    ",
        "def reverse_string(s):\n    "
    ]

    print("\nGenerating sample completions:")
    print("-" * 80)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt.strip()}")

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from completion
        completion = completion[len(prompt):]

        print(f"Completion: {completion[:200]}")

        # Check for degenerate patterns
        if len(set(completion.split())) < 5:
            print("  ⚠️  WARNING: Very low vocabulary diversity")
        if "|||" in completion or "assistant" in completion.lower():
            print("  ⚠️  WARNING: Degenerate repetition detected")

def main():
    # Paths
    baseline_dir = Path("logs/results/fft_eval")
    masked_dir = Path("logs/results/fft_plus_sci_mask_eval")
    mask_indices_path = Path("logs/results/sci/mask_indices.json")
    masked_model_path = Path("checkpoints/fft_plus_sci_mask/final")
    baseline_model_path = Path("checkpoints/fft_qwen2_0.5b/final")

    print("Starting deep validation of 1% masked model results...")

    # 1. Compare predictions in detail
    compare_predictions(baseline_dir, masked_dir)

    # 2. Analyze masked parameters
    analyze_masked_parameters(mask_indices_path, masked_model_path)

    # 3. Check generation quality
    print("\n\nBaseline model generation quality:")
    check_model_quality(baseline_model_path, baseline_model_path)

    print("\n\n1% Masked model generation quality:")
    check_model_quality(masked_model_path, masked_model_path)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
