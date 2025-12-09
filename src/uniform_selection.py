"""
Uniform mask selection with per-row/column caps.

Prevents hub concentration by limiting how many weights can be masked
per row (output neuron) or column (input feature).
"""

import math
import torch
from typing import Dict, List, Tuple, Any, Optional
from .utils_xla import print_once


def compute_degree_stats(mask: torch.BoolTensor) -> Dict[str, float]:
    """
    Compute degree statistics for a boolean mask.

    Args:
        mask: Boolean mask tensor (d_out, d_in)

    Returns:
        Dict with degree statistics
    """
    if mask.dim() != 2:
        return {}

    deg_row = mask.sum(dim=1).float()
    deg_col = mask.sum(dim=0).float()

    # Top-k share helper
    def topk_share(deg, pct):
        if deg.sum() == 0:
            return 0.0
        k = max(1, int(len(deg) * pct))
        vals, _ = torch.topk(deg, k)
        return (vals.sum() / deg.sum()).item()

    # Gini helper
    def gini(deg):
        x = deg.sort().values
        n = x.numel()
        if x.sum() == 0 or n == 0:
            return 0.0
        cumx = torch.cumsum(x, dim=0)
        return ((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n).item()

    return {
        'row_max': deg_row.max().item(),
        'row_mean': deg_row.mean().item(),
        'col_max': deg_col.max().item(),
        'col_mean': deg_col.mean().item(),
        'top1_row_share': topk_share(deg_row, 0.01),
        'top5_row_share': topk_share(deg_row, 0.05),
        'gini_row': gini(deg_row),
        'top1_col_share': topk_share(deg_col, 0.01),
        'top5_col_share': topk_share(deg_col, 0.05),
        'gini_col': gini(deg_col),
    }


def select_with_row_cap(
    scores_2d: torch.Tensor,
    k_total: int,
    cap_multiplier: int = 3,
    max_multiplier: int = 10,
    select_negative: bool = False
) -> Tuple[List[int], int, Dict[str, Any]]:
    """
    Select top-k indices with per-row cap to ensure uniform distribution.

    Args:
        scores_2d: Score tensor [out_dim, in_dim]
        k_total: Total number of weights to select
        cap_multiplier: Initial cap = ceil(cap_multiplier * (k_total / out_dim))
        max_multiplier: Maximum cap multiplier to try before giving up
        select_negative: If True, select most negative scores; else most positive

    Returns:
        Tuple of:
            - List of flat indices selected
            - Final cap used
            - Stats dict
    """
    out_dim, in_dim = scores_2d.shape

    # Sort candidates by score
    flat_scores = scores_2d.view(-1)
    if select_negative:
        # Sort ascending (most negative first)
        sorted_idx = torch.argsort(flat_scores, descending=False)
    else:
        # Sort descending (most positive first)
        sorted_idx = torch.argsort(flat_scores, descending=True)

    # Try increasing caps until we can select enough
    for m in range(cap_multiplier, max_multiplier + 1):
        avg_per_row = k_total / out_dim
        cap = int(math.ceil(m * avg_per_row))
        cap = max(cap, 1)  # At least 1

        row_count = torch.zeros(out_dim, dtype=torch.int32)
        chosen = []

        for flat_i in sorted_idx.tolist():
            r = flat_i // in_dim
            score = flat_scores[flat_i].item()

            # Check sign constraint
            if select_negative and score >= 0:
                continue
            if not select_negative and score <= 0:
                continue

            # Check row cap
            if row_count[r] >= cap:
                continue

            chosen.append(flat_i)
            row_count[r] += 1

            if len(chosen) >= k_total:
                # Compute stats for the selection
                mask = torch.zeros_like(scores_2d, dtype=torch.bool)
                for idx in chosen:
                    mask.view(-1)[idx] = True
                stats = compute_degree_stats(mask)
                stats['cap_multiplier_used'] = m
                stats['cap_used'] = cap
                stats['selected_count'] = len(chosen)
                return chosen, cap, stats

    # Return what we could get
    mask = torch.zeros_like(scores_2d, dtype=torch.bool)
    for idx in chosen:
        mask.view(-1)[idx] = True
    stats = compute_degree_stats(mask)
    stats['cap_multiplier_used'] = max_multiplier
    stats['cap_used'] = cap
    stats['selected_count'] = len(chosen)
    stats['warning'] = f"Could only select {len(chosen)}/{k_total}"

    return chosen, cap, stats


def select_with_row_and_col_cap(
    scores_2d: torch.Tensor,
    k_total: int,
    row_cap_multiplier: int = 3,
    col_cap_multiplier: int = 5,
    max_multiplier: int = 10,
    select_negative: bool = False
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Select top-k indices with both row and column caps.

    Args:
        scores_2d: Score tensor [out_dim, in_dim]
        k_total: Total number of weights to select
        row_cap_multiplier: Initial row cap multiplier
        col_cap_multiplier: Initial column cap multiplier
        max_multiplier: Maximum multiplier to try
        select_negative: If True, select most negative scores

    Returns:
        Tuple of:
            - List of flat indices selected
            - Stats dict
    """
    out_dim, in_dim = scores_2d.shape

    # Sort candidates by score
    flat_scores = scores_2d.view(-1)
    if select_negative:
        sorted_idx = torch.argsort(flat_scores, descending=False)
    else:
        sorted_idx = torch.argsort(flat_scores, descending=True)

    avg_per_row = k_total / out_dim
    avg_per_col = k_total / in_dim

    # Try increasing caps
    for rm in range(row_cap_multiplier, max_multiplier + 1):
        for cm in range(col_cap_multiplier, max_multiplier + 1):
            row_cap = max(1, int(math.ceil(rm * avg_per_row)))
            col_cap = max(1, int(math.ceil(cm * avg_per_col)))

            row_count = torch.zeros(out_dim, dtype=torch.int32)
            col_count = torch.zeros(in_dim, dtype=torch.int32)
            chosen = []

            for flat_i in sorted_idx.tolist():
                r = flat_i // in_dim
                c = flat_i % in_dim
                score = flat_scores[flat_i].item()

                # Check sign constraint
                if select_negative and score >= 0:
                    continue
                if not select_negative and score <= 0:
                    continue

                # Check caps
                if row_count[r] >= row_cap:
                    continue
                if col_count[c] >= col_cap:
                    continue

                chosen.append(flat_i)
                row_count[r] += 1
                col_count[c] += 1

                if len(chosen) >= k_total:
                    mask = torch.zeros_like(scores_2d, dtype=torch.bool)
                    for idx in chosen:
                        mask.view(-1)[idx] = True
                    stats = compute_degree_stats(mask)
                    stats['row_cap_multiplier'] = rm
                    stats['col_cap_multiplier'] = cm
                    stats['row_cap'] = row_cap
                    stats['col_cap'] = col_cap
                    stats['selected_count'] = len(chosen)
                    return chosen, stats

    # Return what we could get
    mask = torch.zeros_like(scores_2d, dtype=torch.bool)
    for idx in chosen:
        mask.view(-1)[idx] = True
    stats = compute_degree_stats(mask)
    stats['selected_count'] = len(chosen)
    stats['warning'] = f"Could only select {len(chosen)}/{k_total}"
    return chosen, stats


def select_top_parameters_uniform(
    sci_scores: Dict[str, torch.Tensor],
    mask_fraction: float = 0.001,
    cap_multiplier: int = 3,
    select_negative: bool = False,
    use_col_cap: bool = False
) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    """
    Select parameters with uniformity constraint (per-row cap).

    This prevents hub concentration by limiting how many weights
    can be masked per output neuron.

    Args:
        sci_scores: Dict mapping param names to SCI score tensors
        mask_fraction: Fraction of parameters to mask per tensor
        cap_multiplier: Row cap = ceil(cap_multiplier * avg_per_row)
        select_negative: If True, select most negative SCI (detrimental params)
        use_col_cap: If True, also apply column caps

    Returns:
        Tuple of:
            - List of (param_name, flat_index) tuples
            - Stats dict with per-tensor and aggregate statistics
    """
    mode = "NEGATIVE (detrimental)" if select_negative else "POSITIVE (beneficial)"
    print_once("=" * 80)
    print_once(f"UNIFORM PARAMETER SELECTION - {mode}")
    print_once(f"Cap multiplier: {cap_multiplier}")
    print_once("=" * 80)

    all_selected = []
    total_params = 0
    total_selected = 0
    per_tensor_stats = {}

    for name, scores in sci_scores.items():
        if scores.dim() != 2:
            print_once(f"  Skipping {name} (not 2D)")
            continue

        param_size = scores.numel()
        total_params += param_size

        k_total = int(param_size * mask_fraction)
        if k_total == 0:
            continue

        # Select with row cap
        if use_col_cap:
            chosen, stats = select_with_row_and_col_cap(
                scores, k_total,
                row_cap_multiplier=cap_multiplier,
                col_cap_multiplier=cap_multiplier * 2,
                select_negative=select_negative
            )
        else:
            chosen, cap, stats = select_with_row_cap(
                scores, k_total,
                cap_multiplier=cap_multiplier,
                select_negative=select_negative
            )

        # Add to results
        for idx in chosen:
            all_selected.append((name, idx))

        total_selected += len(chosen)

        # Store stats
        per_tensor_stats[name] = {
            'param_size': param_size,
            'k_requested': k_total,
            'k_selected': len(chosen),
            **stats
        }

        print_once(f"  {name}: {len(chosen)}/{k_total} selected, "
                   f"top1_row={stats['top1_row_share']:.1%}, "
                   f"gini={stats['gini_row']:.3f}")

    # Aggregate stats
    avg_top1_row = sum(s['top1_row_share'] for s in per_tensor_stats.values()) / max(1, len(per_tensor_stats))
    avg_gini_row = sum(s['gini_row'] for s in per_tensor_stats.values()) / max(1, len(per_tensor_stats))

    aggregate_stats = {
        'total_params': total_params,
        'total_selected': total_selected,
        'mask_fraction': mask_fraction,
        'cap_multiplier': cap_multiplier,
        'select_negative': select_negative,
        'avg_top1_row_share': avg_top1_row,
        'avg_gini_row': avg_gini_row,
        'per_tensor': per_tensor_stats
    }

    print_once(f"\nAggregate statistics:")
    print_once(f"  Total selected: {total_selected:,} / {total_params:,}")
    print_once(f"  Avg top1_row_share: {avg_top1_row:.1%}")
    print_once(f"  Avg gini_row: {avg_gini_row:.3f}")

    # Warnings
    if avg_top1_row > 0.10:
        print_once(f"\n⚠️  WARNING: top1_row_share ({avg_top1_row:.1%}) still high!")
        print_once(f"   Consider increasing cap_multiplier or using col_cap")
    else:
        print_once(f"\n✓ Uniformity looks good (top1_row_share < 10%)")

    return all_selected, aggregate_stats
