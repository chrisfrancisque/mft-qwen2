"""
Degree Analysis for SCI Masked Parameters.

Analyzes whether masked parameters are spread out or concentrated in hubs
(a small number of neurons/heads/rows/cols).

For each masked weight matrix W ∈ R^{d_out × d_in}, treats masked entries
as edges in a bipartite graph and computes degree statistics.
"""

import sys
import json
import re
import csv
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_mask_indices(mask_indices_path: Path) -> Dict[str, List[int]]:
    """Load mask indices from JSON file."""
    with open(mask_indices_path, 'r') as f:
        return json.load(f)


def get_tensor_shape(model, param_name: str) -> Tuple[int, ...]:
    """Get shape of a parameter tensor from model."""
    for name, param in model.named_parameters():
        if name == param_name:
            return tuple(param.shape)
    return None


def reconstruct_boolean_mask(
    flat_indices: List[int],
    shape: Tuple[int, int]
) -> torch.BoolTensor:
    """
    Reconstruct boolean mask from flat indices.

    Args:
        flat_indices: List of flat indices into the tensor
        shape: (d_out, d_in) shape of the weight matrix

    Returns:
        Boolean mask tensor of shape (d_out, d_in)
    """
    mask = torch.zeros(shape[0] * shape[1], dtype=torch.bool)
    for idx in flat_indices:
        if idx < len(mask):
            mask[idx] = True
    return mask.reshape(shape)


def topk_share(deg: torch.Tensor, pct: float) -> float:
    """
    Compute what fraction of total degree is held by top k% of nodes.

    Args:
        deg: 1D tensor of degrees
        pct: Percentile (e.g., 0.01 for top 1%)

    Returns:
        Fraction of total degree held by top k%
    """
    if deg.sum() == 0:
        return 0.0
    k = max(1, int(len(deg) * pct))
    vals, _ = torch.topk(deg.float(), k)
    return (vals.sum() / deg.float().sum()).item()


def gini(deg: torch.Tensor) -> float:
    """
    Compute Gini coefficient for degree distribution.

    Higher Gini = more concentrated (hub-like)
    Lower Gini = more spread out

    Args:
        deg: 1D tensor of degrees

    Returns:
        Gini coefficient (0-1)
    """
    x = deg.float().sort().values
    n = x.numel()
    if x.sum() == 0 or n == 0:
        return 0.0
    cumx = torch.cumsum(x, dim=0)
    return ((n + 1 - 2 * (cumx.sum() / cumx[-1])) / n).item()


def parse_param_name(param_name: str) -> Dict[str, Any]:
    """
    Parse parameter name to extract layer number and module type.

    Args:
        param_name: e.g., "model.layers.19.mlp.down_proj.weight"

    Returns:
        Dict with 'layer' and 'module' keys
    """
    # Extract layer number
    layer_match = re.search(r'layers\.(\d+)\.', param_name)
    layer = int(layer_match.group(1)) if layer_match else -1

    # Extract module type
    if 'self_attn.q_proj' in param_name:
        module = 'attention.q_proj'
        module_type = 'attention'
    elif 'self_attn.k_proj' in param_name:
        module = 'attention.k_proj'
        module_type = 'attention'
    elif 'self_attn.v_proj' in param_name:
        module = 'attention.v_proj'
        module_type = 'attention'
    elif 'self_attn.o_proj' in param_name:
        module = 'attention.o_proj'
        module_type = 'attention'
    elif 'mlp.gate_proj' in param_name:
        module = 'mlp.gate_proj'
        module_type = 'mlp'
    elif 'mlp.up_proj' in param_name:
        module = 'mlp.up_proj'
        module_type = 'mlp'
    elif 'mlp.down_proj' in param_name:
        module = 'mlp.down_proj'
        module_type = 'mlp'
    else:
        module = 'other'
        module_type = 'other'

    return {
        'layer': layer,
        'module': module,
        'module_type': module_type
    }


def compute_tensor_stats(
    param_name: str,
    mask: torch.BoolTensor
) -> Dict[str, Any]:
    """
    Compute degree statistics for a single masked tensor.

    Args:
        param_name: Name of the parameter
        mask: Boolean mask tensor (d_out, d_in)

    Returns:
        Dict with all statistics
    """
    if mask.dim() != 2:
        return None

    d_out, d_in = mask.shape
    masked_count = mask.sum().item()
    total_count = mask.numel()
    masked_frac = masked_count / total_count if total_count > 0 else 0

    # Compute degrees
    deg_row = mask.sum(dim=1)  # shape [d_out] - how many cols masked per row
    deg_col = mask.sum(dim=0)  # shape [d_in] - how many rows masked per col

    # Row statistics
    row_max = deg_row.max().item()
    row_mean = deg_row.float().mean().item()
    row_std = deg_row.float().std().item()
    row_nonzero = (deg_row > 0).sum().item()

    # Column statistics
    col_max = deg_col.max().item()
    col_mean = deg_col.float().mean().item()
    col_std = deg_col.float().std().item()
    col_nonzero = (deg_col > 0).sum().item()

    # Concentration statistics
    top1_row_share = topk_share(deg_row, 0.01)
    top5_row_share = topk_share(deg_row, 0.05)
    top1_col_share = topk_share(deg_col, 0.01)
    top5_col_share = topk_share(deg_col, 0.05)

    # Gini coefficients
    gini_row = gini(deg_row)
    gini_col = gini(deg_col)

    # Parse name for layer/module info
    parsed = parse_param_name(param_name)

    return {
        'param': param_name,
        'layer': parsed['layer'],
        'module': parsed['module'],
        'module_type': parsed['module_type'],
        'shape': [d_out, d_in],
        'd_out': d_out,
        'd_in': d_in,
        'masked_count': masked_count,
        'total_count': total_count,
        'masked_frac': masked_frac,
        # Row stats
        'row_max': row_max,
        'row_mean': row_mean,
        'row_std': row_std,
        'row_nonzero': row_nonzero,
        'top1_row_share': top1_row_share,
        'top5_row_share': top5_row_share,
        'gini_row': gini_row,
        # Column stats
        'col_max': col_max,
        'col_mean': col_mean,
        'col_std': col_std,
        'col_nonzero': col_nonzero,
        'top1_col_share': top1_col_share,
        'top5_col_share': top5_col_share,
        'gini_col': gini_col,
    }


def main():
    import argparse
    from transformers import AutoModelForCausalLM

    parser = argparse.ArgumentParser(description="Analyze degree distribution of SCI masks")
    parser.add_argument(
        "--mask_indices",
        type=str,
        required=True,
        help="Path to mask_indices.json from SCI computation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (to get tensor shapes)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/degree_analysis",
        help="Directory to save analysis results"
    )

    args = parser.parse_args()

    mask_indices_path = Path(args.mask_indices)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DEGREE ANALYSIS FOR SCI MASKED PARAMETERS")
    print("=" * 80)

    # Load mask indices
    print(f"\nLoading mask indices from {mask_indices_path}...")
    mask_indices = load_mask_indices(mask_indices_path)
    print(f"  Found {len(mask_indices)} masked tensors")

    # Load model to get shapes
    print(f"\nLoading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    print("  Model loaded")

    # Get shapes for all parameters
    param_shapes = {}
    for name, param in model.named_parameters():
        param_shapes[name] = tuple(param.shape)

    # Reconstruct boolean masks and compute stats
    print("\nComputing degree statistics...")
    all_stats = []
    mask_dict = {}

    for param_name, flat_indices in mask_indices.items():
        if param_name not in param_shapes:
            print(f"  Warning: {param_name} not found in model")
            continue

        shape = param_shapes[param_name]
        if len(shape) != 2:
            print(f"  Skipping {param_name} (not 2D: {shape})")
            continue

        # Reconstruct boolean mask
        mask = reconstruct_boolean_mask(flat_indices, shape)
        mask_dict[param_name] = mask

        # Compute stats
        stats = compute_tensor_stats(param_name, mask)
        if stats:
            all_stats.append(stats)
            print(f"  {param_name}: {stats['masked_count']:,} masked, "
                  f"top1_row={stats['top1_row_share']:.2%}, "
                  f"gini_row={stats['gini_row']:.3f}")

    # Save boolean mask dict
    mask_dict_path = output_dir / "mask_dict.pt"
    torch.save(mask_dict, mask_dict_path)
    print(f"\nBoolean masks saved to {mask_dict_path}")

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    # By layer
    layer_stats = {}
    for s in all_stats:
        layer = s['layer']
        if layer not in layer_stats:
            layer_stats[layer] = {'masked_count': 0, 'total_count': 0}
        layer_stats[layer]['masked_count'] += s['masked_count']
        layer_stats[layer]['total_count'] += s['total_count']

    print("\nMasked parameters by LAYER:")
    for layer in sorted(layer_stats.keys()):
        ls = layer_stats[layer]
        pct = ls['masked_count'] / ls['total_count'] * 100 if ls['total_count'] > 0 else 0
        print(f"  Layer {layer}: {ls['masked_count']:,} / {ls['total_count']:,} ({pct:.3f}%)")

    # By module type
    module_type_stats = {}
    for s in all_stats:
        mt = s['module_type']
        if mt not in module_type_stats:
            module_type_stats[mt] = {'masked_count': 0, 'total_count': 0}
        module_type_stats[mt]['masked_count'] += s['masked_count']
        module_type_stats[mt]['total_count'] += s['total_count']

    print("\nMasked parameters by MODULE TYPE:")
    for mt, mts in module_type_stats.items():
        pct = mts['masked_count'] / mts['total_count'] * 100 if mts['total_count'] > 0 else 0
        print(f"  {mt}: {mts['masked_count']:,} / {mts['total_count']:,} ({pct:.3f}%)")

    # By specific module
    module_stats = {}
    for s in all_stats:
        m = s['module']
        if m not in module_stats:
            module_stats[m] = {'masked_count': 0, 'total_count': 0}
        module_stats[m]['masked_count'] += s['masked_count']
        module_stats[m]['total_count'] += s['total_count']

    print("\nMasked parameters by MODULE:")
    for m in sorted(module_stats.keys()):
        ms = module_stats[m]
        pct = ms['masked_count'] / ms['total_count'] * 100 if ms['total_count'] > 0 else 0
        print(f"  {m}: {ms['masked_count']:,} / {ms['total_count']:,} ({pct:.3f}%)")

    # Top concentrated tensors
    print("\n" + "=" * 80)
    print("TOP 10 MOST CONCENTRATED TENSORS (by top1_row_share)")
    print("=" * 80)
    sorted_by_concentration = sorted(all_stats, key=lambda x: x['top1_row_share'], reverse=True)
    for i, s in enumerate(sorted_by_concentration[:10]):
        print(f"  {i+1}. {s['param']}")
        print(f"      top1_row_share={s['top1_row_share']:.2%}, "
              f"top5_row_share={s['top5_row_share']:.2%}, "
              f"gini_row={s['gini_row']:.3f}")

    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST ROW DEGREE (row_max)")
    print("=" * 80)
    sorted_by_row_max = sorted(all_stats, key=lambda x: x['row_max'], reverse=True)
    for i, s in enumerate(sorted_by_row_max[:10]):
        print(f"  {i+1}. {s['param']}")
        print(f"      row_max={s['row_max']}, row_mean={s['row_mean']:.2f}, "
              f"masked={s['masked_count']:,}")

    print("\n" + "=" * 80)
    print("TOP 10 HIGHEST COLUMN DEGREE (col_max)")
    print("=" * 80)
    sorted_by_col_max = sorted(all_stats, key=lambda x: x['col_max'], reverse=True)
    for i, s in enumerate(sorted_by_col_max[:10]):
        print(f"  {i+1}. {s['param']}")
        print(f"      col_max={s['col_max']}, col_mean={s['col_mean']:.2f}, "
              f"masked={s['masked_count']:,}")

    # Interpretation summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    avg_top1_row = sum(s['top1_row_share'] for s in all_stats) / len(all_stats)
    avg_gini_row = sum(s['gini_row'] for s in all_stats) / len(all_stats)

    print(f"\nAverage top1_row_share: {avg_top1_row:.2%}")
    print(f"Average gini_row: {avg_gini_row:.3f}")

    if avg_top1_row > 0.10:
        print("\n⚠️  HIGH CONCENTRATION: Top 1% of rows contain >{:.0f}% of masked params".format(avg_top1_row*100))
        print("   This suggests hub-like masking that may damage specific neurons.")
    else:
        print("\n✓  LOW CONCENTRATION: Mask is relatively spread out across rows.")

    if avg_gini_row > 0.5:
        print("\n⚠️  HIGH GINI: Degree distribution is unequal (Gini={:.3f})".format(avg_gini_row))
    else:
        print("\n✓  LOW GINI: Degree distribution is relatively uniform (Gini={:.3f})".format(avg_gini_row))

    # Check attention vs MLP concentration
    attn_stats = [s for s in all_stats if s['module_type'] == 'attention']
    mlp_stats = [s for s in all_stats if s['module_type'] == 'mlp']

    if attn_stats:
        avg_attn_top1 = sum(s['top1_row_share'] for s in attn_stats) / len(attn_stats)
        print(f"\nAttention avg top1_row_share: {avg_attn_top1:.2%}")
        if avg_attn_top1 > 0.15:
            print("   ⚠️  High concentration in attention may cause generation loops!")

    if mlp_stats:
        avg_mlp_top1 = sum(s['top1_row_share'] for s in mlp_stats) / len(mlp_stats)
        print(f"MLP avg top1_row_share: {avg_mlp_top1:.2%}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save JSON
    json_path = output_dir / "mask_degree_summary.json"
    with open(json_path, 'w') as f:
        json.dump({
            'per_tensor': all_stats,
            'by_layer': {str(k): v for k, v in layer_stats.items()},
            'by_module_type': module_type_stats,
            'by_module': module_stats,
            'summary': {
                'avg_top1_row_share': avg_top1_row,
                'avg_gini_row': avg_gini_row,
                'total_tensors': len(all_stats),
                'total_masked': sum(s['masked_count'] for s in all_stats),
            }
        }, f, indent=2)
    print(f"  JSON saved to {json_path}")

    # Save CSV
    csv_path = output_dir / "mask_degree_summary.csv"
    if all_stats:
        fieldnames = list(all_stats[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in all_stats:
                # Convert shape list to string for CSV
                row = s.copy()
                row['shape'] = str(row['shape'])
                writer.writerow(row)
    print(f"  CSV saved to {csv_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
