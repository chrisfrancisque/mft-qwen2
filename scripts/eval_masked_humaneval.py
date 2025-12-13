"""
Evaluate masked model on HumanEval.

Loads model, applies mask, and runs HumanEval evaluation.
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.eval_code import evaluate_humaneval


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate masked model on HumanEval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mask_indices", type=str, default=None, help="Path to mask_indices.json (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU (slower but avoids XLA issues)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.use_cpu:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            dtype = torch.bfloat16
        except:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)
    model.eval()
    print("Model loaded")

    # Apply mask if provided
    mask_name = "unmasked"
    if args.mask_indices:
        print(f"\nLoading mask from {args.mask_indices}...")
        with open(args.mask_indices, 'r') as f:
            mask_indices = json.load(f)

        total_masked = apply_mask_to_model(model, mask_indices)
        print(f"Applied mask: {total_masked:,} parameters zeroed")
        mask_name = Path(args.mask_indices).parent.name
    else:
        print("\nNo mask applied (unmasked model)")

    # Run HumanEval
    print("\n" + "=" * 80)
    print("RUNNING HUMANEVAL")
    print("=" * 80)

    results = evaluate_humaneval(
        model=model,
        tokenizer=tokenizer,
        humaneval_plus=False,
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        device=device,
        output_path=output_dir / f"humaneval_results_{mask_name}.json"
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Mask: {mask_name}")
    print(f"Pass@1: {results['pass@1']:.1%}")
    print(f"Passed: {results['passed']}/{results['total_problems']}")

    # Save summary
    summary = {
        "mask_name": mask_name,
        "mask_indices_path": args.mask_indices,
        "results": results
    }
    with open(output_dir / f"summary_{mask_name}.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
