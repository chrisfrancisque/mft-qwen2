#!/usr/bin/env python3
"""
WANDA-style pruning for Qwen2 models.

WANDA (Weights AND Activations) prunes weights based on:
    importance = |W| * ||A||

where |W| is weight magnitude and ||A|| is the L2 norm of input activations.

This prunes the LOWEST importance weights (unlike SCI which targets highest scores).

Reference: "A Simple and Effective Pruning Approach for Large Language Models"
           Sun et al., 2023 (https://arxiv.org/abs/2306.11695)
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_layers(module: nn.Module, layers: List = [nn.Linear], name: str = '') -> Dict[str, nn.Module]:
    """Recursively find layers of specified types in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class ActivationWrapper:
    """Wraps a linear layer to collect activation statistics."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.device = layer.weight.device
        self.rows = layer.weight.data.shape[0]  # output features
        self.columns = layer.weight.data.shape[1]  # input features

        # Store squared L2 norm of activations per input feature
        self.scaler_row = torch.zeros((self.columns), device=self.device, dtype=torch.float32)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """Add a batch of activations to compute running statistics."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                # (batch, seq_len, hidden) -> (batch * seq_len, hidden)
                inp = inp.reshape((-1, inp.shape[-1]))
            # Transpose: (hidden, batch * seq_len)
            inp = inp.t()

        # Update running mean of squared L2 norms
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        inp = inp.type(torch.float32)
        # L2 norm squared per input feature (column), averaged over samples
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def load_calibration_data(
    tokenizer: AutoTokenizer,
    data_path: Path,
    nsamples: int = 128,
    seqlen: int = 2048,
    seed: int = 42
) -> List[torch.Tensor]:
    """Load calibration data from JSONL file."""
    print(f"Loading calibration data from {data_path}...")

    torch.manual_seed(seed)

    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if len(examples) >= nsamples * 2:  # Load extra for filtering
                break
            data = json.loads(line)
            text = data.get('text', data.get('content', ''))
            if text:
                examples.append(text)

    # Tokenize and filter by length
    calibration_data = []
    for text in examples:
        if len(calibration_data) >= nsamples:
            break

        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=seqlen,
            padding=False
        )

        # Only use examples with sufficient length
        if tokens.input_ids.shape[1] >= seqlen // 2:
            # Pad or truncate to exact seqlen
            if tokens.input_ids.shape[1] < seqlen:
                pad_len = seqlen - tokens.input_ids.shape[1]
                tokens.input_ids = torch.cat([
                    tokens.input_ids,
                    torch.full((1, pad_len), tokenizer.pad_token_id or tokenizer.eos_token_id)
                ], dim=1)
            else:
                tokens.input_ids = tokens.input_ids[:, :seqlen]

            calibration_data.append(tokens.input_ids)

    print(f"Loaded {len(calibration_data)} calibration examples (seqlen={seqlen})")
    return calibration_data


def prepare_calibration_input(
    model: AutoModelForCausalLM,
    calibration_data: List[torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare calibration inputs by running through embeddings and capturing layer inputs.
    """
    print("Preparing calibration inputs...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    nsamples = len(calibration_data)
    seqlen = calibration_data[0].shape[1]
    hidden_size = model.config.hidden_size
    dtype = next(iter(model.parameters())).dtype

    # Storage for layer inputs
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        """Catch inputs to first transformer layer."""
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError  # Stop forward pass

    # Replace first layer with catcher
    layers[0] = Catcher(layers[0])

    for i, input_ids in enumerate(calibration_data):
        try:
            model(input_ids.to(device))
        except ValueError:
            pass

    # Restore first layer
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def prune_wanda(
    model: AutoModelForCausalLM,
    calibration_data: List[torch.Tensor],
    sparsity_ratio: float,
    device: torch.device,
    target_layers: Optional[List[int]] = None,
    prune_n: int = 0,
    prune_m: int = 0
) -> Dict[str, any]:
    """
    Apply WANDA pruning to model.

    Args:
        model: Model to prune
        calibration_data: List of tokenized calibration examples
        sparsity_ratio: Fraction of weights to prune (0.5 = 50%)
        device: Device to run on
        target_layers: If specified, only prune these layer indices
        prune_n, prune_m: For structured N:M sparsity (0,0 = unstructured)

    Returns:
        Dict with pruning statistics
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Prepare calibration inputs
    inps, outs, attention_mask, position_ids = prepare_calibration_input(
        model, calibration_data, device
    )

    layers = model.model.layers
    nsamples = len(calibration_data)

    stats = {
        'layers_pruned': 0,
        'total_weights_pruned': 0,
        'total_weights': 0,
        'layer_stats': {}
    }

    print(f"\nPruning with sparsity ratio: {sparsity_ratio}")
    if target_layers:
        print(f"Target layers: {target_layers}")

    for i in tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]

        # Skip non-target layers if specified
        if target_layers is not None and i not in target_layers:
            # Still need to pass through layer to update inps
            for j in range(nsamples):
                with torch.no_grad():
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )[0]
            inps, outs = outs, inps
            continue

        # Find linear layers in this transformer layer
        subset = find_layers(layer)

        # Wrap layers to collect activation statistics
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = ActivationWrapper(subset[name])

        # Register hooks to collect activations
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Forward pass to collect activations
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

        # Remove hooks
        for h in handles:
            h.remove()

        # Prune each linear layer
        layer_pruned = 0
        layer_total = 0

        for name in subset:
            W = subset[name].weight.data

            # WANDA metric: |W| * sqrt(activation_norm)
            # scaler_row contains squared L2 norms, so we take sqrt
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Create mask (True = prune)
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                # Structured N:M sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True
                        )
            else:
                # Unstructured pruning: prune lowest scores per row
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            # Apply mask
            W[W_mask] = 0

            n_pruned = W_mask.sum().item()
            n_total = W.numel()
            layer_pruned += n_pruned
            layer_total += n_total

        stats['layers_pruned'] += 1
        stats['total_weights_pruned'] += layer_pruned
        stats['total_weights'] += layer_total
        stats['layer_stats'][i] = {
            'pruned': layer_pruned,
            'total': layer_total,
            'sparsity': layer_pruned / layer_total if layer_total > 0 else 0
        }

        # Update inputs for next layer
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache

    stats['overall_sparsity'] = stats['total_weights_pruned'] / stats['total_weights'] if stats['total_weights'] > 0 else 0

    return stats


def check_sparsity(model: AutoModelForCausalLM) -> float:
    """Check actual sparsity of model weights."""
    layers = model.model.layers
    total_zeros = 0
    total_params = 0

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        layer_zeros = 0
        layer_params = 0

        for name in subset:
            W = subset[name].weight.data
            layer_zeros += (W == 0).sum().item()
            layer_params += W.numel()

        total_zeros += layer_zeros
        total_params += layer_params
        print(f"Layer {i}: sparsity = {layer_zeros / layer_params:.4f}")

    overall = total_zeros / total_params
    print(f"\nOverall sparsity: {overall:.4f}")
    return overall


def main():
    parser = argparse.ArgumentParser(description="WANDA pruning for Qwen2")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="Path to calibration data (JSONL)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save pruned model"
    )
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        default=0.5,
        help="Sparsity ratio (default: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples (default: 128)"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length for calibration (default: 2048)"
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to prune (default: all layers)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("WANDA PRUNING FOR QWEN2")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Sparsity: {args.sparsity_ratio}")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Device: {args.device}")

    # Parse target layers
    target_layers = None
    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(',')]
        print(f"Target layers: {target_layers}")

    device = torch.device(args.device)

    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except:
        print("Tokenizer not in checkpoint, using Qwen/Qwen2-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.seqlen = args.seqlen  # Set seqlen attribute

    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # Load calibration data
    calibration_data = load_calibration_data(
        tokenizer,
        Path(args.calibration_data),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed
    )

    # Prune
    print("\n" + "=" * 80)
    print("STARTING WANDA PRUNING")
    print("=" * 80)

    stats = prune_wanda(
        model,
        calibration_data,
        sparsity_ratio=args.sparsity_ratio,
        device=device,
        target_layers=target_layers
    )

    # Verify sparsity
    print("\n" + "=" * 80)
    print("VERIFYING SPARSITY")
    print("=" * 80)
    actual_sparsity = check_sparsity(model)

    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving pruned model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save stats
    stats['actual_sparsity'] = actual_sparsity
    stats['config'] = {
        'model_path': args.model_path,
        'sparsity_ratio': args.sparsity_ratio,
        'nsamples': args.nsamples,
        'seqlen': args.seqlen,
        'target_layers': target_layers
    }

    stats_path = output_path.parent / f"wanda_stats_{output_path.name}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    print("\n" + "=" * 80)
    print("PRUNING COMPLETE")
    print("=" * 80)
    print(f"Layers pruned: {stats['layers_pruned']}")
    print(f"Weights pruned: {stats['total_weights_pruned']:,}")
    print(f"Overall sparsity: {stats['overall_sparsity']:.4f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
