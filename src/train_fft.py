"""
Full Fine-Tuning (FFT) for Qwen2-0.5B on coding tasks.

Implements distributed training on TPU v4-8 with:
- Gradient accumulation
- Mixed precision (bfloat16)
- Cosine LR schedule with warmup
- Periodic checkpointing
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import DataLoader, DistributedSampler

from .utils_xla import (
    get_device,
    get_world_size,
    get_rank,
    is_master,
    print_once,
    mark_step,
    optimizer_step,
    reduce_gradients,
    save_checkpoint,
    prepare_labels_for_clm,
    MetricsTracker
)
from .tokenization import load_qwen2_tokenizer, collate_fn, encode_sft_example, collate_sft_batch


class CodingDataset(torch.utils.data.Dataset):
    """Simple dataset for pre-tokenized coding examples."""

    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_jsonl_data(file_path: Path) -> list:
    """Load JSONL data file."""
    import json

    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    return examples


def create_dataloader(
    data_path: Path,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    use_sft_encoding: bool = True
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        data_path: Path to JSONL file
        tokenizer: Qwen2 tokenizer
        batch_size: Batch size per device
        max_length: Maximum sequence length
        shuffle: Whether to shuffle (handled by DistributedSampler)
        num_workers: Number of data loading workers
        use_sft_encoding: Use SFT encoding with label masking (matching MFT repo)

    Returns:
        DataLoader
    """
    print_once(f"Loading data from {data_path}...")

    # Load examples
    examples = load_jsonl_data(data_path)

    print_once(f"Loaded {len(examples)} examples")

    # Tokenize examples
    print_once("Tokenizing examples...")

    tokenized = []
    for ex in tqdm(examples, desc="Tokenizing", disable=not is_master()):
        if use_sft_encoding and "messages" in ex:
            # Use SFT encoding with label masking (only train on assistant responses)
            tokenized_ex = encode_sft_example(ex, tokenizer, max_length)
            tokenized.append(tokenized_ex)
        else:
            # Legacy: tokenize full text without label masking
            tokenized_ex = tokenizer(
                ex["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
            tokenized.append({
                "input_ids": tokenized_ex["input_ids"],
                "attention_mask": tokenized_ex["attention_mask"]
            })

    # Create dataset
    dataset = CodingDataset(tokenized)

    # Create distributed sampler
    world_size = get_world_size()
    rank = get_rank()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=42
    )

    # Choose collate function based on encoding
    if use_sft_encoding and "messages" in examples[0]:
        collate = lambda batch: collate_sft_batch(batch, tokenizer.pad_token_id)
    else:
        collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True  # For consistent batch sizes
    )

    return dataloader, sampler


def train_fft(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    resume_from: Optional[Path] = None
):
    """
    Main FFT training function.

    Args:
        config_path: Path to config JSON
        data_dir: Directory containing processed data
        output_dir: Directory to save checkpoints
        resume_from: Optional checkpoint to resume from
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print_once("=" * 80)
    print_once("FULL FINE-TUNING (FFT) - Qwen2-0.5B")
    print_once("=" * 80)

    # Setup device
    device = get_device()
    world_size = get_world_size()
    rank = get_rank()

    print_once(f"\nDevice: {device}")
    print_once(f"World size: {world_size}")
    print_once(f"Rank: {rank}")

    # Load model and tokenizer
    model_name = config["model"]["name"]
    print_once(f"\nLoading model: {model_name}")

    tokenizer = load_qwen2_tokenizer(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_once(f"Total parameters: {total_params:,}")
    print_once(f"Trainable parameters: {trainable_params:,}")

    # Load data
    train_config = config["training"]
    batch_size = train_config["batch_size"]
    grad_accum_steps = train_config["gradient_accumulation_steps"]

    train_data_path = data_dir / "train_mixed.jsonl"

    train_loader, train_sampler = create_dataloader(
        data_path=train_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=config["model"]["max_length"],
        shuffle=True
    )

    # Calculate training steps
    num_epochs = train_config["num_epochs"]
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * num_epochs

    print_once(f"\nTraining configuration:")
    print_once(f"  Epochs: {num_epochs}")
    print_once(f"  Batch size per device: {batch_size}")
    print_once(f"  Gradient accumulation steps: {grad_accum_steps}")
    print_once(f"  Effective batch size: {batch_size * world_size * grad_accum_steps}")
    print_once(f"  Steps per epoch: {steps_per_epoch}")
    print_once(f"  Total steps: {total_steps}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_config["weight_decay"]
    )

    # Setup scheduler with dynamic warmup calculation
    # Ensure warmup_steps doesn't exceed total_steps
    config_warmup = train_config.get("warmup_steps", 100)
    warmup_steps = min(config_warmup, max(1, int(0.1 * total_steps)))

    if config_warmup > total_steps:
        print_once(f"  WARNING: Config warmup_steps ({config_warmup}) > total_steps ({total_steps})")
        print_once(f"  Adjusting warmup_steps to {warmup_steps} (10% of total or config value, whichever is smaller)")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print_once(f"\nOptimizer: AdamW")
    print_once(f"  Learning rate: {train_config['learning_rate']}")
    print_once(f"  Weight decay: {train_config['weight_decay']}")
    print_once(f"  Warmup steps: {warmup_steps}")

    # Setup metrics tracking
    metrics_tracker = MetricsTracker(output_dir.parent.parent / "logs" / "results")

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    print_once("\n" + "=" * 80)
    print_once("Starting training...")
    print_once("=" * 80)

    for epoch in range(num_epochs):
        print_once(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        step_loss = 0.0

        # Progress bar tracks optimizer steps, not micro-batches
        progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}",
            disable=not is_master(),
            unit="step"
        )

        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get labels - either from batch (SFT encoding) or prepare for CLM
            if "labels" in batch:
                # SFT encoding: labels already have non-assistant tokens masked
                labels = batch["labels"].to(device)
            else:
                # Legacy: prepare labels for CLM (masks padding only)
                labels = prepare_labels_for_clm(input_ids, tokenizer.pad_token_id)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            step_loss += loss.item()
            epoch_loss += loss.item()

            # Optimizer step every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config["max_grad_norm"]
                )

                # Reduce gradients across devices
                reduce_gradients(optimizer)

                # Optimizer step
                optimizer_step(optimizer, barrier=True)
                scheduler.step()
                optimizer.zero_grad()

                # Mark XLA step
                mark_step()

                global_step += 1

                # Update progress bar (one step per optimizer step)
                progress_bar.update(1)

                # Logging
                if global_step % train_config["logging_steps"] == 0:
                    avg_loss = step_loss / grad_accum_steps
                    current_lr = scheduler.get_last_lr()[0]

                    print_once(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )

                    metrics_tracker.update(global_step, avg_loss, current_lr)
                    step_loss = 0.0

                # Save checkpoint
                if global_step % train_config["save_steps"] == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, output_dir,
                        is_final=False
                    )

        # Close progress bar
        progress_bar.close()

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print_once(f"Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

    # Save final checkpoint
    print_once("\nTraining complete! Saving final checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler,
        global_step, output_dir,
        is_final=True
    )

    # Save metrics
    metrics_tracker.save()

    print_once("=" * 80)
    print_once("FFT TRAINING COMPLETE")
    print_once("=" * 80)


def train_fft_simple(
    train_file: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen2-0.5B",
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 1024,
    grad_accum_steps: int = 4,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    save_steps: int = 500,
):
    """
    Simplified FFT training function with direct arguments.
    """
    print_once("=" * 80)
    print_once("FULL FINE-TUNING (FFT) - Qwen2-0.5B")
    print_once("=" * 80)

    # Setup device
    device = get_device()
    world_size = get_world_size()
    rank = get_rank()

    print_once(f"\nDevice: {device}")
    print_once(f"World size: {world_size}")
    print_once(f"Rank: {rank}")

    # Load model and tokenizer
    print_once(f"\nLoading model: {model_name}")

    tokenizer = load_qwen2_tokenizer(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_once(f"Total parameters: {total_params:,}")
    print_once(f"Trainable parameters: {trainable_params:,}")

    # Load data
    train_loader, train_sampler = create_dataloader(
        data_path=train_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_seq_length,
        shuffle=True,
        use_sft_encoding=True
    )

    # Calculate training steps
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)

    print_once(f"\nTraining configuration:")
    print_once(f"  Train file: {train_file}")
    print_once(f"  Epochs: {num_epochs}")
    print_once(f"  Batch size per device: {batch_size}")
    print_once(f"  Gradient accumulation steps: {grad_accum_steps}")
    print_once(f"  Effective batch size: {batch_size * world_size * grad_accum_steps}")
    print_once(f"  Steps per epoch: {steps_per_epoch}")
    print_once(f"  Total steps: {total_steps}")
    print_once(f"  Warmup steps: {warmup_steps}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print_once(f"\nOptimizer: AdamW")
    print_once(f"  Learning rate: {learning_rate}")
    print_once(f"  Weight decay: {weight_decay}")

    # Setup metrics tracking
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_tracker = MetricsTracker(output_dir / "logs")

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    print_once("\n" + "=" * 80)
    print_once("Starting training...")
    print_once("=" * 80)

    for epoch in range(num_epochs):
        print_once(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        step_loss = 0.0

        progress_bar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}",
            disable=not is_master(),
            unit="step"
        )

        for step, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get labels
            if "labels" in batch:
                labels = batch["labels"].to(device)
            else:
                labels = prepare_labels_for_clm(input_ids, tokenizer.pad_token_id)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()

            step_loss += loss.item()
            epoch_loss += loss.item()

            # Optimizer step every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                reduce_gradients(optimizer)
                optimizer_step(optimizer, barrier=True)
                scheduler.step()
                optimizer.zero_grad()
                mark_step()

                global_step += 1
                progress_bar.update(1)

                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = step_loss / grad_accum_steps
                    current_lr = scheduler.get_last_lr()[0]

                    print_once(
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e}"
                    )

                    metrics_tracker.update(global_step, avg_loss, current_lr)
                    step_loss = 0.0

                # Save checkpoint
                if global_step % save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step, output_dir,
                        is_final=False
                    )

        progress_bar.close()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print_once(f"Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

    # Save final checkpoint
    print_once("\nTraining complete! Saving final checkpoint...")
    save_checkpoint(
        model, optimizer, scheduler,
        global_step, output_dir,
        is_final=True
    )

    metrics_tracker.save()

    print_once("=" * 80)
    print_once("FFT TRAINING COMPLETE")
    print_once("=" * 80)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full Fine-Tuning for Qwen2")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/fft", help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B", help="Model name")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint interval")

    # Legacy args (ignored but don't error)
    parser.add_argument("--config", type=str, default=None, help="(Legacy) Config file path")

    args = parser.parse_args()

    train_fft_simple(
        train_file=Path(args.train_file),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        grad_accum_steps=args.grad_accum_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
