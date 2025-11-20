"""
TPU/XLA utilities for distributed training.

Handles device setup, gradient synchronization, checkpointing, and logging.
"""

import os
import torch
from pathlib import Path
from typing import Optional


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False


def get_device():
    """Get appropriate device (TPU, CUDA, or CPU)."""
    # Check if user explicitly wants CPU (for bypassing TPU issues)
    if os.environ.get('USE_CPU', '').lower() in ('1', 'true', 'yes'):
        return torch.device('cpu')

    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_world_size() -> int:
    """Get number of devices in distributed setup."""
    if is_tpu_available():
        try:
            # PJRT (torch_xla 2.x+) uses runtime API
            import torch_xla.runtime as xr
            return xr.world_size()
        except Exception:
            pass

        try:
            # XRT (torch_xla 1.x) uses xla_model API
            import torch_xla.core.xla_model as xm
            if hasattr(xm, 'xrt_world_size'):
                return xm.xrt_world_size()
        except Exception:
            pass

        # Fall back to 1
        return 1
    elif torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_rank() -> int:
    """Get rank of current process."""
    if is_tpu_available():
        try:
            # PJRT (torch_xla 2.x+) uses runtime API
            import torch_xla.runtime as xr
            return xr.global_ordinal()
        except Exception:
            pass

        try:
            # XRT (torch_xla 1.x) uses xla_model API
            import torch_xla.core.xla_model as xm
            if hasattr(xm, 'get_ordinal'):
                return xm.get_ordinal()
        except Exception:
            pass

        # Fall back to 0
        return 0
    elif torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def is_master() -> bool:
    """Check if current process is master (rank 0)."""
    return get_rank() == 0


def print_once(msg: str, flush: bool = False):
    """Print only from master process."""
    if is_master():
        print(msg, flush=flush)


def mark_step():
    """Mark XLA step for TPU."""
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.mark_step()


def optimizer_step(optimizer, barrier: bool = True):
    """
    Perform optimizer step with XLA support.

    Args:
        optimizer: PyTorch optimizer
        barrier: Whether to add synchronization barrier (TPU only)
    """
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer, barrier=barrier)
    else:
        optimizer.step()


def reduce_gradients(optimizer):
    """
    All-reduce gradients across devices (TPU only).

    Args:
        optimizer: PyTorch optimizer with parameters
    """
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.reduce_gradients(optimizer)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    is_final: bool = False
):
    """
    Save checkpoint (only from master process).

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Global step number
        output_dir: Directory to save checkpoint
        is_final: Whether this is the final checkpoint
    """
    if not is_master():
        return

    if is_final:
        checkpoint_dir = output_dir / "final"
    else:
        checkpoint_dir = output_dir / f"step_{step}"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print_once(f"Saving checkpoint to {checkpoint_dir}...")

    # Force XLA sync before saving
    if is_tpu_available():
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()

    # Create checkpoint state
    checkpoint_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'step': step,
        'is_final': is_final
    }

    # Save using xm.save for XLA compatibility (handles XLA tensors properly)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    try:
        if is_tpu_available():
            import torch_xla.core.xla_model as xm
            xm.save(checkpoint_state, checkpoint_path)
        else:
            torch.save(checkpoint_state, checkpoint_path)

        print_once(f"Checkpoint state saved successfully to {checkpoint_path}")
    except Exception as e:
        print_once(f"Error saving checkpoint state: {e}")
        import traceback
        print_once(traceback.format_exc())
        return

    # Also save model config separately for easy loading
    try:
        model.config.save_pretrained(checkpoint_dir)
        print_once(f"Model config saved to {checkpoint_dir}")
    except Exception as e:
        print_once(f"Error saving model config: {e}")

    print_once(f"✓ Checkpoint saved at step {step}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir: Path
):
    """
    Load checkpoint.

    Args:
        model: Model to load into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        checkpoint_dir: Directory containing checkpoint

    Returns:
        Global step number
    """
    print_once(f"Loading checkpoint from {checkpoint_dir}...")

    # Load model
    model = model.from_pretrained(checkpoint_dir)

    # Load training state
    training_state_path = checkpoint_dir / "training_state.pt"
    if training_state_path.exists():
        state = torch.load(training_state_path)
        optimizer.load_state_dict(state['optimizer'])
        if scheduler and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])
        step = state['step']
    else:
        step = 0

    print_once(f"Loaded checkpoint from step {step}")

    return model, optimizer, scheduler, step


def prepare_labels_for_clm(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Prepare labels for causal language modeling.

    Shifts input_ids right and masks padding tokens with -100.

    Args:
        input_ids: Input token IDs [batch, seq_len]
        pad_token_id: Padding token ID to mask

    Returns:
        Labels tensor [batch, seq_len]
    """
    labels = input_ids.clone()

    # Mask padding tokens
    labels[labels == pad_token_id] = -100

    return labels


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'learning_rate': [],
            'step': []
        }

    def update(self, step: int, loss: float, lr: float):
        """Update metrics."""
        self.metrics['step'].append(step)
        self.metrics['train_loss'].append(loss)
        self.metrics['learning_rate'].append(lr)

    def save(self):
        """Save metrics to file (master only)."""
        if not is_master():
            return

        import json

        metrics_file = self.log_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print_once(f"Metrics saved to {metrics_file}")
