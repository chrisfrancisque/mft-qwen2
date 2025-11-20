"""
Download Qwen2.5-Coder-0.5B-Instruct to use as the FFT checkpoint.

This is used instead of actual FFT training when training fails or
to validate the SCI masking pipeline on a pre-trained code model.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    output_dir = Path("checkpoints/fft/final")

    print("=" * 80)
    print("DOWNLOADING QWEN2.5-CODER-0.5B-INSTRUCT")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"Output: {output_dir}")
    print("\nThis model is:")
    print("  - Same architecture as Qwen2-0.5B (494M params, 24 layers)")
    print("  - Trained on 6 trillion tokens of code data")
    print("  - Instruction-tuned for coding tasks")
    print("  - Will be used as 'FFT checkpoint' for SCI masking experiment")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    print("\nDownloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Summary
    print("\n" + "=" * 80)
    print("✓ DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\nModel parameters: {model.num_parameters():,}")
    print(f"Saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. bash scripts/run_eval_fft.sh     # Baseline evaluation")
    print("  2. bash scripts/run_sci.sh          # Compute SCI scores")
    print("  3. bash scripts/run_mask.sh         # Apply mask")
    print("  4. bash scripts/eval_masked.sh      # Evaluate masked model")
    print("  5. python3 scripts/compare_results.py  # Compare results")


if __name__ == "__main__":
    main()
