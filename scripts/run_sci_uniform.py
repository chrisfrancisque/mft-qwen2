"""
Run SCI masking with uniformity constraint and loss tracking.

This script:
1. Computes SCI scores for target layers
2. Selects parameters with per-row cap (uniformity constraint)
3. Evaluates validation loss before masking
4. Applies the mask
5. Evaluates validation loss after masking
6. Saves detailed metrics including degree analysis
"""

import sys
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.sci_gradients import (
    load_gradient_examples,
    get_target_parameters,
    accumulate_gradients,
    compute_sci_scores
)
from src.uniform_selection import (
    select_top_parameters_uniform,
    compute_degree_stats
)
from src.utils_xla import get_device, is_master, print_once


def evaluate_validation_loss(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list,
    device: torch.device,
    max_examples: int = 50
) -> float:
    """
    Evaluate validation loss on examples.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        examples: List of examples with 'text' field
        device: Device to run on
        max_examples: Maximum examples to evaluate

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, ex in enumerate(examples[:max_examples]):
            text = ex['text']
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(device)

            labels = inputs['input_ids'].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()

            # Count non-padding tokens
            tokens = (labels != -100).sum().item()
            total_loss += loss * tokens
            total_tokens += tokens

            if (i + 1) % 10 == 0:
                print_once(f"  Evaluated {i+1}/{min(len(examples), max_examples)}...")

    return total_loss / max(1, total_tokens)


def apply_mask_to_model(
    model: AutoModelForCausalLM,
    mask_indices: dict
) -> dict:
    """
    Apply mask by zeroing selected parameters.

    Args:
        model: Model to mask
        mask_indices: Dict mapping param names to list of flat indices

    Returns:
        Stats about masking
    """
    total_masked = 0
    params_modified = 0

    for name, indices in mask_indices.items():
        for pname, param in model.named_parameters():
            if pname == name:
                # Zero out the selected indices
                flat_param = param.data.view(-1)
                for idx in indices:
                    if idx < len(flat_param):
                        flat_param[idx] = 0.0
                        total_masked += 1
                params_modified += 1
                break

    return {
        'total_masked': total_masked,
        'params_modified': params_modified
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SCI masking with uniformity constraint and loss tracking"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SCI config JSON"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--grad_data",
        type=str,
        required=True,
        help="Path to grad_subset.jsonl"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )

    parser.add_argument(
        "--cap_multiplier",
        type=int,
        default=3,
        help="Row cap multiplier for uniformity (default: 3)"
    )

    parser.add_argument(
        "--select_negative",
        action="store_true",
        help="Select most negative SCI scores (detrimental params) instead of positive"
    )

    parser.add_argument(
        "--use_col_cap",
        action="store_true",
        help="Also apply column caps for uniformity"
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the masked model to output_dir/masked_model"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = json.load(f)

    sci_config = config["sci"]

    # Paths
    checkpoint_dir = Path(args.checkpoint)
    grad_data_path = Path(args.grad_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"sci_uniform_cap{args.cap_multiplier}"
    if args.select_negative:
        run_name += "_negative"

    print_once("=" * 80)
    print_once("SCI MASKING WITH UNIFORMITY CONSTRAINT")
    print_once("=" * 80)
    print_once(f"Run: {run_name}")
    print_once(f"Started: {datetime.now().isoformat()}")
    print_once(f"Cap multiplier: {args.cap_multiplier}")
    print_once(f"Select negative: {args.select_negative}")

    # Get device
    device = get_device()
    print_once(f"\nDevice: {device}")

    # Load model and tokenizer
    print_once(f"\nLoading model from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if str(device).startswith('xla') else torch.float32,
        trust_remote_code=True
    )
    model = model.to(device)
    print_once("Model loaded")

    # Load validation examples
    print_once(f"\nLoading validation examples from {grad_data_path}...")
    val_examples = []
    with open(grad_data_path, 'r') as f:
        for line in f:
            val_examples.append(json.loads(line))
    print_once(f"Loaded {len(val_examples)} examples")

    # =========================================================================
    # STEP 1: Evaluate loss BEFORE masking
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 1: EVALUATING LOSS BEFORE MASKING")
    print_once("=" * 80)

    loss_before = evaluate_validation_loss(
        model, tokenizer, val_examples, device, max_examples=50
    )
    print_once(f"\nValidation loss BEFORE masking: {loss_before:.4f}")

    # =========================================================================
    # STEP 2: Compute SCI scores
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 2: COMPUTING SCI SCORES")
    print_once("=" * 80)

    # Load gradient examples
    grad_examples = load_gradient_examples(
        data_path=grad_data_path,
        tokenizer=tokenizer,
        max_length=sci_config.get("max_length", 1024),
        max_examples=sci_config.get("max_grad_examples", 256)
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
        raise ValueError("No target parameters found!")

    # Accumulate gradients
    accumulated_grads = accumulate_gradients(
        model=model,
        tokenizer=tokenizer,
        examples=grad_examples,
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

    # =========================================================================
    # STEP 3: Select parameters with uniformity constraint
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 3: SELECTING PARAMETERS WITH UNIFORMITY CONSTRAINT")
    print_once("=" * 80)

    selected_params, selection_stats = select_top_parameters_uniform(
        sci_scores=sci_scores,
        mask_fraction=sci_config.get("mask_fraction", 0.001),
        cap_multiplier=args.cap_multiplier,
        select_negative=args.select_negative,
        use_col_cap=args.use_col_cap
    )

    # Convert to mask indices format
    mask_indices = {}
    for name, idx in selected_params:
        if name not in mask_indices:
            mask_indices[name] = []
        mask_indices[name].append(int(idx))

    # Save mask indices
    mask_indices_path = output_dir / "mask_indices.json"
    with open(mask_indices_path, 'w') as f:
        json.dump(mask_indices, f, indent=2)
    print_once(f"\nMask indices saved to {mask_indices_path}")

    # =========================================================================
    # STEP 4: Apply mask
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 4: APPLYING MASK")
    print_once("=" * 80)

    mask_stats = apply_mask_to_model(model, mask_indices)
    print_once(f"Masked {mask_stats['total_masked']:,} parameters in {mask_stats['params_modified']} tensors")

    # =========================================================================
    # STEP 5: Evaluate loss AFTER masking
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 5: EVALUATING LOSS AFTER MASKING")
    print_once("=" * 80)

    loss_after = evaluate_validation_loss(
        model, tokenizer, val_examples, device, max_examples=50
    )
    print_once(f"\nValidation loss AFTER masking: {loss_after:.4f}")

    loss_change = loss_after - loss_before
    loss_change_pct = (loss_change / loss_before) * 100 if loss_before > 0 else 0

    print_once(f"\nLoss change: {loss_change:+.4f} ({loss_change_pct:+.1f}%)")

    if loss_change > 0:
        print_once("⚠️  Loss INCREASED after masking (model got worse)")
    else:
        print_once("✓ Loss DECREASED after masking (model may have improved)")

    # =========================================================================
    # STEP 6: Save results
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("STEP 6: SAVING RESULTS")
    print_once("=" * 80)

    # Compile metrics
    metrics = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "checkpoint": str(checkpoint_dir),
            "mask_fraction": sci_config.get("mask_fraction", 0.001),
            "target_layers": target_layers,
            "cap_multiplier": args.cap_multiplier,
            "select_negative": args.select_negative,
            "use_col_cap": args.use_col_cap
        },
        "validation": {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_change": loss_change,
            "loss_change_pct": loss_change_pct,
            "num_examples": min(50, len(val_examples))
        },
        "masking": {
            "total_masked": mask_stats['total_masked'],
            "params_modified": mask_stats['params_modified']
        },
        "uniformity": {
            "avg_top1_row_share": selection_stats['avg_top1_row_share'],
            "avg_gini_row": selection_stats['avg_gini_row'],
            "per_tensor": selection_stats['per_tensor']
        }
    }

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print_once(f"Metrics saved to {metrics_path}")

    # Save masked model if requested
    if args.save_model and is_master():
        model_path = output_dir / "masked_model"
        model_path.mkdir(parents=True, exist_ok=True)

        print_once(f"\nSaving masked model to {model_path}...")
        # Move to CPU for saving
        model_cpu = model.cpu()
        model_cpu.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print_once("Masked model saved")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_once("\n" + "=" * 80)
    print_once("SUMMARY")
    print_once("=" * 80)
    print_once(f"\nRun: {run_name}")
    print_once(f"Target layers: {target_layers}")
    print_once(f"Mask fraction: {sci_config.get('mask_fraction', 0.001):.1%}")
    print_once(f"Cap multiplier: {args.cap_multiplier}")
    print_once(f"Select negative: {args.select_negative}")
    print_once(f"\nValidation Loss:")
    print_once(f"  Before: {loss_before:.4f}")
    print_once(f"  After:  {loss_after:.4f}")
    print_once(f"  Change: {loss_change:+.4f} ({loss_change_pct:+.1f}%)")
    print_once(f"\nUniformity:")
    print_once(f"  Avg top1_row_share: {selection_stats['avg_top1_row_share']:.1%}")
    print_once(f"  Avg gini_row: {selection_stats['avg_gini_row']:.3f}")
    print_once(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
