"""
Visualization utilities for attention weights in 3D medical image slice aggregation.

This module provides:
- plot_attention_profile: Bar chart of attention weights per slice
- plot_multiview_attention_comparison: Side-by-side comparison across view planes
- compute_attention_metrics: Entropy, concentration, GT comparison metrics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union
import logging

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def plot_attention_profile(
    attention_weights: np.ndarray,
    sample_id: Union[str, int],
    view_plane: str,
    output_dir: str,
    ground_truth_range: Optional[Tuple[int, int]] = None,
    class_label: Optional[str] = None,
    fig_size: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot attention weights as a bar chart along the slice axis.
    
    Args:
        attention_weights: 1D array of attention weights [num_slices]
        sample_id: Sample identifier
        view_plane: View plane name (axial/sagittal/coronal)
        output_dir: Directory to save the plot
        ground_truth_range: Optional tuple (start, end) of slice range with pathology
        class_label: Optional class label (e.g., "ACL tear")
        fig_size: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_slices = len(attention_weights)
    slice_indices = np.arange(num_slices)
    
    # Normalize attention weights if not already
    if not np.isclose(attention_weights.sum(), 1.0, atol=1e-3):
        attention_weights = attention_weights / attention_weights.sum()
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Color bars by attention magnitude
    colors = plt.cm.Reds(attention_weights / attention_weights.max())
    bars = ax.bar(slice_indices, attention_weights, color=colors, edgecolor='darkred', linewidth=0.5)
    
    # Add ground truth annotations if provided
    if ground_truth_range is not None:
        start, end = ground_truth_range
        ax.axvspan(start, end, alpha=0.2, color='green', label='GT region')
    
    # Styling
    ax.set_xlabel('Slice Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
    title = f'Attention Profile - {view_plane.capitalize()}'
    if class_label:
        title += f' ({class_label})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, num_slices - 0.5)
    ax.set_ylim(0, attention_weights.max() * 1.1)
    
    # Add legend if ground truth is provided
    if ground_truth_range is not None:
        ax.legend(loc='upper right')
    
    fig.tight_layout()
    
    filename = f'slices_attention_profile_{sample_id}_{view_plane}.png'
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved slices attention profile to {filepath}")


def plot_multiview_attention_comparison(
    attention_weights_dict: Dict[str, np.ndarray],
    sample_id: Union[str, int],
    output_dir: str,
    class_label: Optional[str] = None,
    fig_size: Tuple[int, int] = (14, 4),
) -> None:
    """
    Compare attention profiles across multiple view planes (axial, sagittal, coronal).
    
    Args:
        attention_weights_dict: Dict mapping view_plane -> attention_weights array
        sample_id: Sample identifier  
        output_dir: Directory to save the plot
        class_label: Optional class label
        fig_size: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_views = len(attention_weights_dict)
    fig, axes = plt.subplots(1, num_views, figsize=fig_size, sharey=True)
    
    if num_views == 1:
        axes = [axes]
    
    colors_map = {'axial': 'Blues', 'sagittal': 'Oranges', 'coronal': 'Greens'}
    
    for ax, (view_plane, attention_weights) in zip(axes, attention_weights_dict.items()):
        num_slices = len(attention_weights)
        slice_indices = np.arange(num_slices)
        
        # Normalize
        if not np.isclose(attention_weights.sum(), 1.0, atol=1e-3):
            attention_weights = attention_weights / attention_weights.sum()
        
        cmap = plt.cm.get_cmap(colors_map.get(view_plane, 'Reds'))
        colors = cmap(attention_weights / attention_weights.max())
        
        ax.bar(slice_indices, attention_weights, color=colors, edgecolor='gray', linewidth=0.3)
        ax.set_xlabel('Slice Index', fontsize=10)
        ax.set_title(f'{view_plane.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, num_slices - 0.5)
    
    axes[0].set_ylabel('Attention Weight', fontsize=10, fontweight='bold')
    
    title = f'Multi-View Attention Comparison (Sample: {sample_id})'
    if class_label:
        title += f' - {class_label}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    filename = f'multiview_attention_{sample_id}.png'
    filepath = os.path.join(output_dir, filename) 
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved multi-view attention comparison to {filepath}")


def plot_per_head_attention_profile(
    attention_weights: np.ndarray,
    sample_id: Union[str, int],
    view_plane: str,
    output_dir: str,
    class_label: Optional[str] = None,
    fig_size: Tuple[int, int] = (14, 10),
) -> None:
    """
    Plot per-head ABMIL attention profiles.
    
    Shows aggregated attention on top, followed by each head's individual profile.
    
    Args:
        attention_weights: 2D array [num_heads, num_slices]
        sample_id: Sample identifier
        view_plane: View plane name
        output_dir: Directory to save the plot
        class_label: Optional class label
        fig_size: Figure size
    """
    os.makedirs(output_dir, exist_ok=True)

    num_heads, num_slices = attention_weights.shape
    slice_indices = np.arange(num_slices)

    fig, axes = plt.subplots(num_heads + 1, 1, figsize=fig_size, sharex=True,
                             gridspec_kw={'hspace': 0.3})

    # Aggregated attention (average over heads)
    avg_attn = attention_weights.mean(axis=0)
    avg_attn = avg_attn / avg_attn.sum()
    colors = plt.cm.Reds(avg_attn / avg_attn.max())
    axes[0].bar(slice_indices, avg_attn, color=colors, edgecolor='darkred', linewidth=0.5)
    axes[0].set_ylabel('Weight', fontsize=9)
    axes[0].set_title('Aggregated (average over heads)', fontsize=11, fontweight='bold')
    axes[0].set_xlim(-0.5, num_slices - 0.5)

    head_cmaps = ['Blues', 'Oranges', 'Greens', 'Purples', 'YlOrBr', 'PiYG', 'BrBG', 'RdYlGn']
    for i in range(num_heads):
        attn = attention_weights[i]
        cmap = plt.cm.get_cmap(head_cmaps[i % len(head_cmaps)])
        bar_colors = cmap(attn / attn.max())
        axes[i + 1].bar(slice_indices, attn, color=bar_colors, edgecolor='gray', linewidth=0.3)
        axes[i + 1].set_ylabel('Weight', fontsize=9)
        axes[i + 1].set_title(f'Head {i + 1}', fontsize=10)
        axes[i + 1].set_xlim(-0.5, num_slices - 0.5)

    axes[-1].set_xlabel('Slice Index', fontsize=12, fontweight='bold')

    title = f'Per-Head Attention - {view_plane.capitalize()}'
    if class_label:
        title += f' ({class_label})'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()

    filename = f'per_head_attention_{sample_id}_{view_plane}.png'
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved per-head attention profile to {filepath}")


def compute_attention_metrics(
    attention_weights: np.ndarray,
    ground_truth_range: Optional[Tuple[int, int]] = None,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for attention weights.
    
    Args:
        attention_weights: 1D array of attention weights [num_slices]
        ground_truth_range: Optional tuple (start, end) defining pathology range
        top_k: Number of top attention slices for GT comparison
        
    Returns:
        Dictionary with metrics:
        - num_slices: Total number of slices
        - peak_position: Index of highest attention slice
        - peak_attention: Value of highest attention
        
        If ground_truth_range provided:
        - peak_in_gt_range: Whether peak is in GT region (0 or 1)
        - iou_topk_gt_range: IoU between top-k slices and GT region
        - attention_in_gt: Sum of attention in GT region
    """
    metrics = {}
    num_slices = len(attention_weights)
    
    # Normalize if needed
    if not np.isclose(attention_weights.sum(), 1.0, atol=1e-3):
        attention_weights = attention_weights / attention_weights.sum()
    
    # Basic statistics
    metrics['num_slices'] = num_slices
    metrics['peak_position'] = int(np.argmax(attention_weights))
    metrics['peak_attention'] = float(np.max(attention_weights))

    # Entropy (higher = more uniform distribution)
    eps = 1e-12
    entropy = -np.sum(attention_weights * np.log(attention_weights + eps))
    metrics['entropy'] = float(entropy)
    metrics['max_entropy'] = float(np.log(num_slices))
    metrics['normalized_entropy'] = float(entropy / np.log(num_slices)) if num_slices > 1 else 1.0

    # Gini coefficient (higher = more concentrated on few slices)
    sorted_weights = np.sort(attention_weights)
    n = len(sorted_weights)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_weights) - (n + 1) * np.sum(sorted_weights)) / (n * np.sum(sorted_weights) + eps)
    metrics['gini_coefficient'] = float(gini)

    # Ground truth comparison
    if ground_truth_range is not None:
        start, end = ground_truth_range
        peak_idx = metrics['peak_position']
        
        # Peak in GT range
        metrics['peak_in_gt_range'] = float(start <= peak_idx <= end)
        
        # Attention mass in GT region
        gt_attention = attention_weights[start:end + 1].sum()
        metrics['attention_in_gt'] = float(gt_attention)
        
        # IoU between top-k and GT range
        top_k_indices = set(np.argsort(attention_weights)[-top_k:][::-1])
        gt_range_indices = set(range(start, end + 1))
        intersection = len(top_k_indices & gt_range_indices)
        union = len(top_k_indices | gt_range_indices)
        metrics['iou_topk_gt_range'] = float(intersection / union) if union > 0 else 0.0
    
    return metrics

