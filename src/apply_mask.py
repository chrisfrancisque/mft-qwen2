"""
Apply SCI mask to model by zeroing out selected parameters.

This module:
1. Loads the FFT checkpoint fresh
2. Loads the mask indices from SCI computation
3. Zeros out the selected parameters
4. Saves the masked model checkpoint

No further training - just mask and save.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils_xla import is_master, print_once


def load_mask_indices(mask_path: Path) -> Dict[str, List[int]]:
    """
    Load mask indices from JSON file.

    Args:
        mask_path: Path to mask_indices.json

    Returns:
        Dict mapping parameter names to lists of flat indices to zero
    """
    print_once(f"Loading mask indices from {mask_path}...")

    with open(mask_path, 'r') as f:
        mask_indices = json.load(f)

    total_indices = sum(len(indices) for indices in mask_indices.values())

    print_once(f"Loaded mask indices for {len(mask_indices)} parameters")
    print_once(f"Total indices to mask: {total_indices:,}")

    return mask_indices


def apply_mask_to_model(
    model: AutoModelForCausalLM,
    mask_indices: Dict[str, List[int]]
) -> Dict[str, any]:
    """
    Zero out parameters according to mask indices.

    Args:
        model: Model to apply mask to
        mask_indices: Dict mapping parameter names to flat indices

    Returns:
        Dict with masking statistics
    """
    print_once("=" * 80)
    print_once("APPLYING MASK TO MODEL")
    print_once("=" * 80)

    total_params_masked = 0
    total_model_params = 0
    params_modified = 0

    # Get all model parameters
    model_params = dict(model.named_parameters())

    # Count total parameters
    for param in model_params.values():
        total_model_params += param.numel()

    print_once(f"\nTotal model parameters: {total_model_params:,}")

    # Apply mask
    with torch.no_grad():
        for param_name, indices in mask_indices.items():
            if param_name not in model_params:
                print_once(f"Warning: Parameter {param_name} not found in model, skipping")
                continue

            param = model_params[param_name]

            # Flatten parameter
            flat_param = param.view(-1)

            # Zero out indices
            for idx in indices:
                if idx < flat_param.numel():
                    flat_param[idx] = 0.0
                else:
                    print_once(f"Warning: Index {idx} out of bounds for {param_name} (size {flat_param.numel()})")

            total_params_masked += len(indices)
            params_modified += 1

            # Print progress for first few parameters
            if params_modified <= 3:
                print_once(f"  Masked {len(indices):,} indices in {param_name}")

    print_once(f"\nMasking summary:")
    print_once(f"  Parameters modified: {params_modified}")
    print_once(f"  Total indices masked: {total_params_masked:,}")
    print_once(f"  Masking ratio: {total_params_masked / total_model_params:.4%}")

    stats = {
        "total_model_params": total_model_params,
        "total_params_masked": total_params_masked,
        "masking_ratio": total_params_masked / total_model_params,
        "params_modified": params_modified
    }

    return stats


def verify_mask(
    model: AutoModelForCausalLM,
    mask_indices: Dict[str, List[int]],
    target_layers: List[int]
) -> Dict[str, any]:
    """
    Verify that mask was applied correctly.

    Args:
        model: Masked model
        mask_indices: Mask indices that were applied
        target_layers: Target layer indices

    Returns:
        Dict with verification results
    """
    print_once("=" * 80)
    print_once("VERIFYING MASK")
    print_once("=" * 80)

    model_params = dict(model.named_parameters())

    # Count zeros in target layers
    target_zeros = 0
    target_total = 0

    # Count zeros outside target layers
    non_target_zeros = 0
    non_target_total = 0

    for name, param in model_params.items():
        # Check if in target layers
        is_target = False
        for layer_idx in target_layers:
            if f"layers.{layer_idx}." in name:
                is_target = True
                break

        # Count zeros
        num_zeros = (param == 0).sum().item()
        num_params = param.numel()

        if is_target:
            target_zeros += num_zeros
            target_total += num_params
        else:
            non_target_zeros += num_zeros
            non_target_total += num_params

    print_once(f"\nTarget layers {target_layers}:")
    print_once(f"  Total parameters: {target_total:,}")
    print_once(f"  Zero parameters: {target_zeros:,}")
    print_once(f"  Zero ratio: {target_zeros / target_total:.4%}")

    print_once(f"\nNon-target layers:")
    print_once(f"  Total parameters: {non_target_total:,}")
    print_once(f"  Zero parameters: {non_target_zeros:,}")
    print_once(f"  Zero ratio: {non_target_zeros / non_target_total:.4%}")

    # Verify no parameters outside target layers were modified
    if non_target_zeros > 0:
        print_once(f"\nWarning: Found {non_target_zeros:,} zeros in non-target layers!")
        print_once("This may indicate masking outside intended layers.")

    verification = {
        "target_layers": target_layers,
        "target_total": target_total,
        "target_zeros": target_zeros,
        "target_zero_ratio": target_zeros / target_total if target_total > 0 else 0,
        "non_target_total": non_target_total,
        "non_target_zeros": non_target_zeros,
        "non_target_zero_ratio": non_target_zeros / non_target_total if non_target_total > 0 else 0
    }

    return verification


def apply_sci_mask(
    fft_checkpoint_dir: Path,
    mask_indices_path: Path,
    output_dir: Path,
    target_layers: List[int]
) -> Dict[str, any]:
    """
    Main function to apply SCI mask and save masked model.

    Args:
        fft_checkpoint_dir: Path to FFT checkpoint
        mask_indices_path: Path to mask_indices.json
        output_dir: Directory to save masked checkpoint
        target_layers: Target layer indices for verification

    Returns:
        Dict with results
    """
    print_once("=" * 80)
    print_once("APPLYING SCI MASK")
    print_once("=" * 80)

    # Load mask indices
    mask_indices = load_mask_indices(mask_indices_path)

    # Load FFT checkpoint fresh
    print_once(f"\nLoading FFT checkpoint from {fft_checkpoint_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(
        fft_checkpoint_dir,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        fft_checkpoint_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    print_once("Model loaded successfully")

    # Apply mask
    mask_stats = apply_mask_to_model(model, mask_indices)

    # Verify mask
    verification = verify_mask(model, mask_indices, target_layers)

    # Save masked model
    if is_master():
        output_dir.mkdir(parents=True, exist_ok=True)

        print_once(f"\nSaving masked model to {output_dir}...")

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print_once("Masked model saved successfully")

        # Save statistics
        stats_path = output_dir.parent / "mask_stats.json"

        combined_stats = {
            "masking": mask_stats,
            "verification": verification
        }

        with open(stats_path, 'w') as f:
            json.dump(combined_stats, f, indent=2)

        print_once(f"Statistics saved to {stats_path}")

    print_once("\n" + "=" * 80)
    print_once("MASKING COMPLETE")
    print_once("=" * 80)

    return {
        "mask_stats": mask_stats,
        "verification": verification
    }
