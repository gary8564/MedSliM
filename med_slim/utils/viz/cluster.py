"""
UMAP / t-SNE embedding visualization for volumetric-image features.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import logging
from typing import Optional, List, Tuple, Dict

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


def _prepare_supervised_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels to a 1-D array suitable for supervised UMAP."""
    if labels.ndim == 2:
        return np.array(["_".join(str(int(v)) for v in row) for row in labels])
    if labels.dtype.kind in ("U", "S", "O"):
        return labels
    return labels.astype(int)


def plot_embedding_clustering(
    embeddings: np.ndarray,
    output_dir: str,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    title: str = '',
    fig_size: Tuple[int, int] = (10, 8),
    filename: str = 'embedding_umap.png',
    umap_kwargs: Optional[Dict] = None,
    method: str = 'umap',
    tsne_kwargs: Optional[Dict] = None,
    supervised: bool = False,
    pca_dim: Optional[int] = 50,
) -> str:
    """
    Visualize embeddings using UMAP or t-SNE dimensionality reduction.

    Preprocessing:
        - L2 normalization: contrastive SSL embeddings live on the unit
          hypersphere, so cosine distances are meaningful only after
          normalization.
        - PCA to `pca_dim` components: removes low-variance noise
          dimensions that distort the nearest-neighbor graph. Set to
          `None` to skip.

    Args:
        embeddings: 2D array of embeddings [num_samples, embed_dim]
        output_dir: Directory to save the plot
        labels: Optional labels for coloring. Can be:
                - 1D array of class indices [num_samples]
                - 1D array of strings [num_samples] (e.g. plane names)
                - 2D array of multi-label binary indicators [num_samples, num_labels]
        label_names: List of class/label names for legend
        title: Plot title
        fig_size: Figure size
        filename: Output filename
        umap_kwargs: Additional arguments for UMAP
        method: Dimensionality reduction method ('umap' or 'tsne')
        tsne_kwargs: Additional arguments for t-SNE
        supervised: Only valid for method='umap'. If True, use labels to guide UMAP layout.
        pca_dim: Number of PCA components for denoising before
                 dimensionality reduction. Set to None to skip PCA.

    Returns:
        Path to saved figure
    """
    os.makedirs(output_dir, exist_ok=True)

    # Preprocessing
    embeddings = normalize(embeddings, norm='l2')
    logger.info(f"L2-normalized {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    if pca_dim is not None and embeddings.shape[1] > pca_dim:
        embeddings = PCA(n_components=pca_dim, random_state=42).fit_transform(embeddings)
        logger.info(f"PCA reduced to {pca_dim} dimensions")

    if method == 'tsne':
        default_tsne_kwargs = {
            'perplexity': min(30, len(embeddings) - 1),
            'metric': 'cosine',
            'random_state': 42,
            'max_iter': 1000,
        }
        if tsne_kwargs:
            default_tsne_kwargs.update(tsne_kwargs)
        if supervised:
            logger.warning("Supervised mode is not supported for t-SNE, falling back to unsupervised.")
        logger.info(f"Running t-SNE on {len(embeddings)} samples...")
        reducer = TSNE(n_components=2, **default_tsne_kwargs)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        default_umap_kwargs = {
            'n_neighbors': 30, 
            'min_dist': 0.0, 
            'metric': 'cosine', 
            'random_state': 42
        }
        if umap_kwargs:
            default_umap_kwargs.update(umap_kwargs)
        reducer = umap.UMAP(n_components=2, **default_umap_kwargs)

        if supervised and labels is not None:
            y = _prepare_supervised_labels(labels)
            logger.info(f"Running supervised UMAP on {len(embeddings)} samples...")
            embedding_2d = reducer.fit_transform(embeddings, y=y)
        else:
            if supervised:
                logger.warning("Supervised UMAP requires labels; falling back to unsupervised.")
            logger.info(f"Running UMAP on {len(embeddings)} samples...")
            embedding_2d = reducer.fit_transform(embeddings)
    
    # Build DataFrame with label strings
    if labels is not None and labels.ndim == 2:
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
        if labels.dtype.kind in ("U", "S", "O"):
            label_strs = labels.tolist()
        elif label_names:
            label_strs = [label_names[int(v)] for v in labels]
        else:
            label_strs = [str(int(v)) for v in labels]
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
    
    if labels is not None:
        unique_labels = df['Label'].unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        
        sns.scatterplot(
            data=df,
            x='UMAP1',
            y='UMAP2',
            hue='Label',
            palette=palette,
            alpha=0.45,
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
