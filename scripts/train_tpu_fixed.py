"""
TPU-Optimized Training Script with Fixed Shapes.

This script addresses XLA recompilation issues by:
1. Loading pre-tokenized tensors with FIXED shapes
2. Using drop_last=True for constant batch sizes
3. Dynamic warmup calculation (never exceeds total_steps)
4. Frequent checkpointing during training
5. XLA metrics logging for recompilation detection

Usage:
    python scripts/train_tpu_fixed.py
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# Import XLA utilities
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.debug.metrics as met
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    print("WARNING: torch_xla not available, will use CPU/CUDA")


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "Qwen/Qwen2-0.5B",
    "data_path": "data_processed/train_1k_balanced.pt",
    "output_dir": "checkpoints/fft_tpu_fixed",

    # Training hyperparameters
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "num_epochs": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # Checkpointing - frequent saves for debugging
    "checkpoint_steps": [1, 5, 10, 15, 20, 25, 30, 31],  # Save at these steps
    "logging_steps": 1,  # Log every step for debugging

    # Hardware
    "dtype": "bfloat16",
}


# ============================================================================
# Device utilities
# ============================================================================

def get_device():
    """Get the appropriate device (TPU, CUDA, or CPU)."""
    if HAS_XLA:
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_world_size():
    """Get number of distributed workers."""
    if HAS_XLA:
        return xm.xrt_world_size()
    return 1


def get_rank():
    """Get current worker rank."""
    if HAS_XLA:
        return xm.get_ordinal()
    return 0


def is_master():
    """Check if current process is master."""
    return get_rank() == 0


def print_master(msg):
    """Print only on master process."""
    if is_master():
        print(msg, flush=True)


def mark_step():
    """Mark XLA step for graph execution."""
    if HAS_XLA:
        xm.mark_step()


def save_xla_checkpoint(state_dict, path):
    """Save checkpoint using XLA-safe method."""
    if HAS_XLA:
        xm.save(state_dict, path)
    else:
        torch.save(state_dict, path)


# ============================================================================
# Data loading with FIXED shapes
# ============================================================================

def load_fixed_dataset(data_path: str):
    """
    Load pre-tokenized dataset with fixed shapes.

    The .pt file contains:
    - input_ids: [N, seq_len]
    - attention_mask: [N, seq_len]
    - labels: [N, seq_len]

    All tensors have IDENTICAL shapes - no recompilation!
    """
    print_master(f"Loading pre-tokenized data from {data_path}...")

    data = torch.load(data_path)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels = data["labels"]

    print_master(f"  Loaded {len(input_ids)} examples")
    print_master(f"  Shape: {input_ids.shape}")
    print_master(f"  All shapes identical: {input_ids.shape == attention_mask.shape == labels.shape}")

    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    return dataset


def create_dataloader(dataset, batch_size: int, shuffle: bool = True):
    """
    Create DataLoader with FIXED batch sizes.

    Key settings:
    - drop_last=True: ensures every batch has exactly batch_size samples
    - This prevents shape variation at the end of epoch
    """
    world_size = get_world_size()
    rank = get_rank()

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=42,
            drop_last=True,  # CRITICAL: constant batch size
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        drop_last=True,  # CRITICAL: constant batch size
        num_workers=0,  # Avoid multiprocessing issues on TPU
        pin_memory=False,
    )

    return dataloader, sampler


# ============================================================================
# Training loop
# ============================================================================

def train():
    """Main training function with all TPU optimizations."""

    print_master("=" * 80)
    print_master("TPU-OPTIMIZED TRAINING WITH FIXED SHAPES")
    print_master("=" * 80)
    print_master(f"Started at: {datetime.now().isoformat()}")

    # Setup device
    device = get_device()
    world_size = get_world_size()
    rank = get_rank()

    print_master(f"\nDevice: {device}")
    print_master(f"World size: {world_size}")
    print_master(f"Rank: {rank}")
    print_master(f"XLA available: {HAS_XLA}")

    # Paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / CONFIG["data_path"]
    output_dir = base_dir / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (for pad_token_id)
    print_master(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print_master(f"Loading model: {CONFIG['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16 if CONFIG["dtype"] == "bfloat16" else torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_master(f"Total parameters: {total_params:,}")

    # Load pre-tokenized dataset with FIXED shapes
    dataset = load_fixed_dataset(data_path)

    # Create dataloader with FIXED batch sizes
    batch_size = CONFIG["batch_size"]
    grad_accum_steps = CONFIG["gradient_accumulation_steps"]

    dataloader, sampler = create_dataloader(dataset, batch_size, shuffle=True)

    # Calculate training steps DYNAMICALLY
    num_epochs = CONFIG["num_epochs"]
    micro_batches_per_epoch = len(dataloader)
    steps_per_epoch = micro_batches_per_epoch // grad_accum_steps
    total_steps = steps_per_epoch * num_epochs

    print_master(f"\nTraining configuration:")
    print_master(f"  Dataset size: {len(dataset)}")
    print_master(f"  Batch size: {batch_size}")
    print_master(f"  Gradient accumulation: {grad_accum_steps}")
    print_master(f"  Effective batch size: {batch_size * world_size * grad_accum_steps}")
    print_master(f"  Micro-batches per epoch: {micro_batches_per_epoch}")
    print_master(f"  Optimizer steps per epoch: {steps_per_epoch}")
    print_master(f"  Total optimizer steps: {total_steps}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=CONFIG["weight_decay"],
    )

    # Setup scheduler with DYNAMIC warmup (never exceeds total_steps)
    warmup_steps = min(3, max(1, int(0.1 * total_steps)))  # 10% or 3, whichever is smaller
    print_master(f"  Warmup steps: {warmup_steps} (dynamic, <= total_steps)")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Wrap dataloader for XLA if available
    if HAS_XLA:
        dataloader = pl.MpDeviceLoader(dataloader, device)

    # Training state
    model.train()
    global_step = 0
    optimizer.zero_grad()
    accumulated_loss = 0.0

    print_master(f"\n{'='*80}")
    print_master("STARTING TRAINING")
    print_master(f"{'='*80}")
    print_master(f"Checkpoint will be saved at steps: {CONFIG['checkpoint_steps']}")

    # Track step times for diagnostics
    step_start_time = time.time()

    for epoch in range(num_epochs):
        print_master(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        if sampler is not None:
            sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(dataloader):
            # Unpack batch (already on device if using MpDeviceLoader)
            if HAS_XLA:
                input_ids, attention_mask, labels = batch
            else:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            accumulated_loss += loss.item()

            # Optimizer step every grad_accum_steps
            if (micro_step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    CONFIG["max_grad_norm"],
                )

                # Optimizer step
                if HAS_XLA:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # Mark XLA step
                mark_step()

                global_step += 1

                # Calculate step time
                step_time = time.time() - step_start_time
                step_start_time = time.time()

                # Logging (every step for debugging)
                if global_step % CONFIG["logging_steps"] == 0:
                    avg_loss = accumulated_loss
                    lr = scheduler.get_last_lr()[0]

                    print_master(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {step_time:.1f}s"
                    )

                    # XLA metrics for recompilation detection
                    if HAS_XLA and is_master():
                        print_master("\n--- XLA Metrics (recompilation check) ---")
                        print_master(met.metrics_report())
                        print_master("--- End XLA Metrics ---\n")

                    accumulated_loss = 0.0

                # Checkpoint saving at specified steps
                if global_step in CONFIG["checkpoint_steps"]:
                    ckpt_path = output_dir / f"checkpoint_step_{global_step}.pt"
                    print_master(f"\nSaving checkpoint at step {global_step} to {ckpt_path}")

                    state_dict = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "config": CONFIG,
                    }
                    save_xla_checkpoint(state_dict, str(ckpt_path))
                    print_master(f"  Checkpoint saved!")

    # Save final checkpoint
    print_master(f"\n{'='*80}")
    print_master("TRAINING COMPLETE")
    print_master(f"{'='*80}")

    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    # Save model in HuggingFace format
    print_master(f"Saving final model to {final_path}")
    if is_master():
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

    print_master(f"\nCompleted at: {datetime.now().isoformat()}")
    print_master(f"Total steps completed: {global_step}")


if __name__ == "__main__":
    train()
