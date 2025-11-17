"""
Entry point for FFT training with torch_xla.distributed.xla_multiprocessing.

This script is spawned on each TPU core.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_fft import train_fft


def main():
    parser = argparse.ArgumentParser(description="Full Fine-Tuning for Qwen2-0.5B")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing processed data (JSONL files)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints"
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint directory to resume from"
    )

    args = parser.parse_args()

    # Convert to Path objects
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    resume_from = Path(args.resume_from) if args.resume_from else None

    # Run training
    train_fft(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        resume_from=resume_from
    )


if __name__ == "__main__":
    main()
