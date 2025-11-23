"""
SCI (Sign-Corrected Influence) gradient computation.

Computes SCI scores for parameters in target layers:
    SCI_i = sign(θ_i) × ∂L/∂θ_i

Where L is the average loss over gradient subset examples.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tokenization import load_qwen2_tokenizer
from .utils_xla import (
    get_device,
    is_master,
    print_once,
    prepare_labels_for_clm,
    mark_step,
    is_tpu_available
)


def load_gradient_examples(
    data_path: Path,
    tokenizer: AutoTokenizer,
    max_length: int = 1024,
    max_examples: int = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Load and tokenize gradient subset examples.

    Args:
        data_path: Path to grad_subset.jsonl
        tokenizer: Qwen2 tokenizer
        max_length: Max sequence length (all examples padded to this length)
        max_examples: Optional limit on number of examples to load

    Returns:
        List of tokenized examples (all with same sequence length)
    """
    print_once(f"Loading gradient examples from {data_path}...")

    examples = []

    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]

            # Tokenize with FIXED padding to max_length
            # This ensures all examples have identical shape for XLA
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',  # Pad to max_length (not dynamic)
                return_tensors="pt"
            )

            examples.append({
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            })

            # Limit number of examples if specified
            if max_examples is not None and len(examples) >= max_examples:
                break

    print_once(f"Loaded {len(examples)} gradient examples")
    print_once(f"All examples padded to fixed length: {max_length} tokens")

    return examples


def get_target_parameters(
    model: AutoModelForCausalLM,
    target_layers: List[int],
    include_patterns: List[str]
) -> Dict[str, torch.nn.Parameter]:
    """
    Get parameters in target layers matching include patterns.

    Args:
        model: Qwen2 model
        target_layers: List of layer indices to target (e.g., [20, 21, 22, 23])
        include_patterns: Parameter name patterns to include

    Returns:
        Dict mapping parameter names to parameters
    """
    target_params = {}

    for name, param in model.named_parameters():
        # Check if in target layer
        is_target_layer = False
        for layer_idx in target_layers:
            if f"layers.{layer_idx}." in name:
                is_target_layer = True
                break

        if not is_target_layer:
            continue

        # Check if matches include patterns
        matches_pattern = False
        for pattern in include_patterns:
            if pattern in name:
                matches_pattern = True
                break

        if matches_pattern and param.requires_grad:
            target_params[name] = param

    print_once(f"Found {len(target_params)} target parameters")
    print_once(f"Target layers: {target_layers}")
    print_once(f"Include patterns: {include_patterns}")

    # Print example parameter names
    if is_master() and target_params:
        print_once("\nExample target parameters:")
        for i, name in enumerate(list(target_params.keys())[:5]):
            print_once(f"  {name}")
        if len(target_params) > 5:
            print_once(f"  ... and {len(target_params) - 5} more")

    return target_params


def accumulate_gradients(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[Dict[str, torch.Tensor]],
    target_params: Dict[str, torch.nn.Parameter],
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Accumulate gradients over all examples.

    Args:
        model: Qwen2 model
        tokenizer: Qwen2 tokenizer
        examples: List of tokenized examples
        target_params: Dict of target parameters
        batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        device: Device to run on

    Returns:
        Dict mapping parameter names to accumulated gradients
    """
    if device is None:
        device = get_device()

    print_once("=" * 80)
    print_once("ACCUMULATING GRADIENTS")
    print_once("=" * 80)

    # Set model to eval mode but enable gradients
    model.eval()

    # Zero all gradients
    model.zero_grad()

    # Accumulate gradients
    total_loss = 0.0
    num_examples = 0

    effective_batch_size = batch_size * gradient_accumulation_steps

    print_once(f"\nGradient accumulation settings:")
    print_once(f"  Batch size per device: {batch_size}")
    print_once(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print_once(f"  Effective batch size: {effective_batch_size}")
    print_once(f"  Total examples: {len(examples)}")

    # Process in batches
    import time
    for batch_idx, i in enumerate(tqdm(
        range(0, len(examples), batch_size),
        desc="Accumulating gradients",
        disable=not is_master()
    )):
        batch_start = time.time()
        batch = examples[i:i + batch_size]

        # Debug: print timing for first batch
        if batch_idx == 0:
            print_once(f"\n[DEBUG] Starting first batch (XLA compilation expected)...", flush=True)

        # Stack batch (all examples already padded to same length)
        input_ids_list = [ex["input_ids"] for ex in batch]
        attention_mask_list = [ex["attention_mask"] for ex in batch]

        input_ids = torch.stack(input_ids_list).to(device)
        attention_mask = torch.stack(attention_mask_list).to(device)

        if batch_idx == 0:
            print_once(f"[DEBUG] Batch shape: {input_ids.shape} (all batches will have same shape)", flush=True)

        # Prepare labels
        labels = prepare_labels_for_clm(input_ids, tokenizer.pad_token_id)

        # Forward pass
        if batch_idx == 0:
            print_once(f"[DEBUG] Starting forward pass (batch shape: {input_ids.shape})...", flush=True)

        forward_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        if batch_idx == 0:
            print_once(f"[DEBUG] Forward pass took {time.time() - forward_start:.2f}s", flush=True)

        loss = outputs.loss
        total_loss += loss.item() * len(batch)
        num_examples += len(batch)

        # Backward pass
        if batch_idx == 0:
            print_once(f"[DEBUG] Starting backward pass...", flush=True)

        backward_start = time.time()
        loss.backward()

        if batch_idx == 0:
            print_once(f"[DEBUG] Backward pass took {time.time() - backward_start:.2f}s", flush=True)
            print_once(f"[DEBUG] Total batch time: {time.time() - batch_start:.2f}s", flush=True)
            print_once(f"[DEBUG] First batch complete! Subsequent batches should be faster.\n", flush=True)

        # Mark step for XLA
        if is_tpu_available():
            mark_step()

    # Average loss
    avg_loss = total_loss / num_examples if num_examples > 0 else 0.0

    print_once(f"\nAverage loss over gradient examples: {avg_loss:.4f}")

    # Extract gradients for target parameters
    accumulated_grads = {}

    for name, param in target_params.items():
        if param.grad is not None:
            # Clone gradient to CPU
            if is_tpu_available():
                grad = param.grad.cpu().clone()
            else:
                grad = param.grad.clone()

            accumulated_grads[name] = grad
        else:
            print_once(f"Warning: No gradient for parameter {name}")

    print_once(f"\nExtracted gradients for {len(accumulated_grads)} parameters")

    return accumulated_grads


def compute_sci_scores(
    model: AutoModelForCausalLM,
    accumulated_grads: Dict[str, torch.Tensor],
    target_params: Dict[str, torch.nn.Parameter]
) -> Dict[str, torch.Tensor]:
    """
    Compute SCI scores: score_i = sign(θ_i) × ∂L/∂θ_i

    Args:
        model: Qwen2 model
        accumulated_grads: Dict of accumulated gradients
        target_params: Dict of target parameters

    Returns:
        Dict mapping parameter names to SCI scores
    """
    print_once("=" * 80)
    print_once("COMPUTING SCI SCORES")
    print_once("=" * 80)

    sci_scores = {}

    for name, param in target_params.items():
        if name not in accumulated_grads:
            print_once(f"Warning: No gradient for {name}, skipping")
            continue

        grad = accumulated_grads[name]

        # Get parameter values (move to CPU if needed)
        if is_tpu_available():
            param_values = param.cpu().detach()
        else:
            param_values = param.detach()

        # Compute SCI: sign(θ) × grad
        sci = torch.sign(param_values) * grad

        sci_scores[name] = sci

    print_once(f"\nComputed SCI scores for {len(sci_scores)} parameters")

    return sci_scores


def select_top_parameters(
    sci_scores: Dict[str, torch.Tensor],
    mask_fraction: float = 0.05
) -> Tuple[List[Tuple[str, int]], Dict[str, any]]:
    """
    Select top k% parameters with highest positive SCI scores PER PARAMETER.

    This ensures balanced masking - each parameter gets exactly k% masked,
    preventing any single parameter from being destroyed.

    Args:
        sci_scores: Dict of SCI scores
        mask_fraction: Fraction of parameters to mask per parameter (default 0.05 = 5%)

    Returns:
        Tuple of:
            - List of (param_name, flat_index) tuples for top parameters
            - Dict with selection statistics
    """
    print_once("=" * 80)
    print_once("SELECTING TOP PARAMETERS (PER-PARAMETER MASKING)")
    print_once("=" * 80)

    top_params = []
    total_params = 0
    total_selected = 0
    per_param_stats = {}

    # Select top k% from EACH parameter separately
    for name, scores in sci_scores.items():
        flat_scores = scores.flatten()
        param_size = flat_scores.numel()
        total_params += param_size

        # Select top k% for this parameter
        k = int(param_size * mask_fraction)

        # Get all (index, score) pairs
        param_scores = [(idx, score.item()) for idx, score in enumerate(flat_scores)]

        # Sort by score descending
        param_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top k with positive scores
        selected_for_param = []
        for idx, score in param_scores[:k]:
            if score > 0:
                top_params.append((name, idx))
                selected_for_param.append(score)
            else:
                break

        total_selected += len(selected_for_param)

        # Track per-parameter stats
        if selected_for_param:
            per_param_stats[name] = {
                "param_size": param_size,
                "selected": len(selected_for_param),
                "mask_ratio": len(selected_for_param) / param_size,
                "min_score": min(selected_for_param),
                "max_score": max(selected_for_param),
                "mean_score": sum(selected_for_param) / len(selected_for_param)
            }

    print_once(f"\nTotal parameters in target layers: {total_params:,}")
    print_once(f"Mask fraction per parameter: {mask_fraction:.1%}")
    print_once(f"Total parameters selected for masking: {total_selected:,}")
    print_once(f"Overall masking ratio: {total_selected / total_params:.2%}")

    # Print per-parameter breakdown
    print_once(f"\nPer-parameter masking breakdown:")
    for name, pstats in per_param_stats.items():
        print_once(f"  {name}:")
        print_once(f"    Selected: {pstats['selected']:,}/{pstats['param_size']:,} ({pstats['mask_ratio']:.2%})")
        print_once(f"    SCI range: [{pstats['min_score']:.4f}, {pstats['max_score']:.4f}]")

    # Overall statistics
    all_scores = []
    for name, idx in top_params:
        # Get the actual score (need to look it up)
        param_scores = sci_scores[name].flatten()
        all_scores.append(param_scores[idx].item())

    stats = {
        "total_params": total_params,
        "mask_fraction": mask_fraction,
        "num_selected": total_selected,
        "num_positive": total_selected,  # All selected are positive by construction
        "overall_mask_ratio": total_selected / total_params,
        "per_param_stats": per_param_stats
    }

    if all_scores:
        stats["min_score"] = min(all_scores)
        stats["max_score"] = max(all_scores)
        stats["mean_score"] = sum(all_scores) / len(all_scores)
    else:
        stats["min_score"] = None
        stats["max_score"] = None
        stats["mean_score"] = None

    return top_params, stats


def compute_sci_and_select(
    checkpoint_dir: Path,
    grad_data_path: Path,
    output_dir: Path,
    sci_config: dict
) -> Dict[str, any]:
    """
    Main function to compute SCI scores and select top parameters.

    Args:
        checkpoint_dir: Path to FFT checkpoint
        grad_data_path: Path to grad_subset.jsonl
        output_dir: Directory to save results
        sci_config: SCI configuration dict

    Returns:
        Dict with results
    """
    print_once("=" * 80)
    print_once("SCI GRADIENT COMPUTATION")
    print_once("=" * 80)

    # Get device
    device = get_device()

    # Load model and tokenizer
    print_once(f"\nLoading model from {checkpoint_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = model.to(device)

    print_once(f"Model loaded on {device}")

    # Load gradient examples
    examples = load_gradient_examples(
        data_path=grad_data_path,
        tokenizer=tokenizer,
        max_length=sci_config.get("max_length", 1024),
        max_examples=sci_config.get("max_grad_examples", None)
    )

    # Get target parameters
    layer_band = sci_config["layer_band"]
    target_layers = list(range(layer_band["start"], layer_band["end"] + 1))
    include_patterns = sci_config["include_patterns"]

    target_params = get_target_parameters(
        model=model,
        target_layers=target_layers,
        include_patterns=include_patterns
    )

    if not target_params:
        raise ValueError("No target parameters found! Check layer_band and include_patterns.")

    # Accumulate gradients
    accumulated_grads = accumulate_gradients(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        target_params=target_params,
        batch_size=sci_config.get("batch_size", 4),
        gradient_accumulation_steps=sci_config.get("gradient_accumulation_steps", 8),
        device=device
    )

    # Compute SCI scores
    sci_scores = compute_sci_scores(
        model=model,
        accumulated_grads=accumulated_grads,
        target_params=target_params
    )

    # Select top parameters
    top_params, stats = select_top_parameters(
        sci_scores=sci_scores,
        mask_fraction=sci_config.get("mask_fraction", 0.05)
    )

    # Save results
    if is_master():
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save top parameter indices
        indices_path = output_dir / "mask_indices.json"

        # Convert to serializable format
        mask_indices = {}
        for name, idx in top_params:
            if name not in mask_indices:
                mask_indices[name] = []
            mask_indices[name].append(int(idx))

        with open(indices_path, 'w') as f:
            json.dump(mask_indices, f, indent=2)

        print_once(f"\nMask indices saved to {indices_path}")

        # Save statistics
        stats_path = output_dir / "sci_stats.json"

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print_once(f"SCI statistics saved to {stats_path}")

        # Save full SCI scores (optional, can be large)
        # Uncomment if you want to save all scores
        # scores_path = output_dir / "sci_scores.pt"
        # torch.save(sci_scores, scores_path)
        # print_once(f"Full SCI scores saved to {scores_path}")

    print_once("\n" + "=" * 80)
    print_once("SCI COMPUTATION COMPLETE")
    print_once("=" * 80)

    return {
        "mask_indices": mask_indices if is_master() else None,
        "stats": stats
    }
