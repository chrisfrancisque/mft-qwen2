"""
FFT training launcher using xmp.spawn() for multi-core TPU.

This works with PJRT runtime (torch_xla 2.x+).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _mp_fn(index, config_path, data_dir, output_dir, resume_from):
    """Function to run on each TPU core."""
    # Import here to avoid issues before spawn
    from src.train_fft import train_fft

    train_fft(
        config_path=Path(config_path),
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        resume_from=Path(resume_from) if resume_from else None
    )


def main():
    import argparse
    import torch_xla.distributed.xla_multiprocessing as xmp

    parser = argparse.ArgumentParser(description="Full Fine-Tuning for Qwen2-0.5B (Multi-core)")

    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint to resume from")

    args = parser.parse_args()

    # Spawn training on all TPU cores
    # For PJRT, nprocs must be None (auto-detect) or 1
    xmp.spawn(
        _mp_fn,
        args=(args.config, args.data_dir, args.output_dir, args.resume_from),
        nprocs=None,  # Auto-detect all available devices
        start_method='spawn'  # Use 'spawn' instead of 'fork' for PJRT
    )


if __name__ == "__main__":
    main()
