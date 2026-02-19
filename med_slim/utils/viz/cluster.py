"""
UMAP embedding visualization for volumetric-image features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
import logging
from typing import Optional, List, Tuple, Dict

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


def plot_embedding_clustering(
    embeddings: np.ndarray,
    output_dir: str,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = '',
    fig_size: Tuple[int, int] = (10, 8),
    filename: str = 'embedding_umap.png',
    umap_kwargs: Optional[Dict] = None,
    continuous_color: bool = False,
    method: str = 'umap',
    tsne_kwargs: Optional[Dict] = None,
) -> str:
    """
    Visualize embeddings using UMAP or t-SNE dimensionality reduction.
    
    Args:
        embeddings: 2D array of embeddings [num_samples, embed_dim]
        output_dir: Directory to save the plot
        labels: Optional labels for coloring. Can be:
                - 1D array of class indices [num_samples]
                - 2D array of multi-label binary indicators [num_samples, num_labels]
                - 1D array of continuous values (when continuous_color=True)
        label_names: List of class/label names for legend (or single name for continuous colorbar)
        title: Plot title
        fig_size: Figure size
        filename: Output filename
        umap_kwargs: Additional arguments for UMAP
        continuous_color: If True, treat labels as continuous values and use colormap
        method: Dimensionality reduction method ('umap' or 'tsne')
        tsne_kwargs: Additional arguments for t-SNE
    
    Returns:
        Path to saved figure
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if method == 'tsne':
        default_tsne_kwargs = {
            'perplexity': min(30, len(embeddings) - 1),
            'metric': 'cosine',
            'random_state': 42,
            'n_iter': 1000,
        }
        if tsne_kwargs:
            default_tsne_kwargs.update(tsne_kwargs)
        logger.info(f"Running t-SNE on {len(embeddings)} samples...")
        reducer = TSNE(n_components=2, **default_tsne_kwargs)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        default_umap_kwargs = {
            'n_neighbors': 15, 
            'min_dist': 0.1, 
            'metric': 'cosine', 
            'random_state': 42
        }
        if umap_kwargs:
            default_umap_kwargs.update(umap_kwargs)
        logger.info(f"Running UMAP on {len(embeddings)} samples...")
        reducer = umap.UMAP(n_components=2, **default_umap_kwargs)
        embedding_2d = reducer.fit_transform(embeddings)
    
    # Class labels
    if labels is not None and labels.ndim == 2:
        # Multi-label
        combined_labels = []
        for row in labels:
            active = [label_names[i] for i, val in enumerate(row) if val == 1] if label_names else [str(i) for i, val in enumerate(row) if val == 1]
            combined_labels.append(" + ".join(active) if active else "Normal")
        df = pd.DataFrame({
            'UMAP1': embedding_2d[:, 0],
            'UMAP2': embedding_2d[:, 1],
            'Label': combined_labels,
        })
    elif labels is not None:
        # Binary label
        if label_names:
            label_strs = [label_names[int(l)] for l in labels]
        else:
            label_strs = [str(int(l)) for l in labels]
        df = pd.DataFrame({
            'UMAP1': embedding_2d[:, 0],
            'UMAP2': embedding_2d[:, 1],
            'Label': label_strs,
        })
    else:
        df = pd.DataFrame({
            'UMAP1': embedding_2d[:, 0],
            'UMAP2': embedding_2d[:, 1],
        })
    
    # Set seaborn style
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=fig_size)
    
    if labels is not None and continuous_color:
        # Continuous colormap (e.g., for slice position or volume ID)
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap='gist_rainbow',
            alpha=0.7,
            s=30,
            edgecolor='none',
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar_label = label_names[0] if label_names else ''
        cbar.set_label(cbar_label, fontsize=11)
    elif labels is not None:
        unique_labels = df['Label'].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        
        sns.scatterplot(
            data=df,
            x='UMAP1',
            y='UMAP2',
            hue='Label',
            palette=palette,
            alpha=0.7,
            s=50,
            edgecolor='none',
            ax=ax,
        )
        
        ax.legend(
            title='',
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(unique_labels), 5),
            frameon=False,
            fontsize=11,
            markerscale=1.2,
        )
    else:
        sns.scatterplot(
            data=df,
            x='UMAP1',
            y='UMAP2',
            alpha=0.7,
            s=50,
            edgecolor='none',
            ax=ax,
        )
    
    # Clean styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    fig.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info(f"Saved UMAP plot to {filepath}")
    return filepath
