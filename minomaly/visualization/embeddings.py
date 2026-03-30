"""Scatter-plot utility for graph embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402


def scatter_embs(
    embs_np: np.ndarray,
    labels: np.ndarray | None = None,
    legend: dict[int, str] | str | None = None,
    reduce_dim: str | None = None,
    out_path: str | None = None,
) -> None:
    """Scatter plot of embeddings with optional PCA/TSNE reduction.

    Parameters
    ----------
    embs_np : np.ndarray
        2-D array of shape ``(N, D)`` containing embedding vectors.
    labels : np.ndarray | None
        Integer label per point (used to colour and group the scatter).
    legend : dict | str | None
        Mapping from label value to human-readable name, **or** a single
        string label applied to all points.
    reduce_dim : str | None
        ``"PCA"`` or ``"TSNE"`` (case-insensitive) to reduce to 2-D
        before plotting.  *None* skips reduction (expects 2-D input).
    out_path : str | None
        If provided, save the figure to this path and close it.
    """
    method = (reduce_dim or "").upper()
    if method == "PCA" and embs_np.shape[1] > 2:
        embs_np = PCA(n_components=2).fit_transform(embs_np)
    elif method == "TSNE" and embs_np.shape[1] > 2:
        embs_np = TSNE(n_components=2).fit_transform(embs_np)

    plt.figure()

    if labels is not None and legend is not None and isinstance(legend, dict):
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            lbl = legend.get(int(label), str(label))
            plt.scatter(
                embs_np[idx, 0],
                embs_np[idx, 1],
                label=lbl,
                cmap=cm.tab20,
                alpha=0.6,
                s=12,
            )
    else:
        lbl: Any = legend if isinstance(legend, str) else None
        plt.scatter(
            embs_np[:, 0],
            embs_np[:, 1],
            label=lbl,
            cmap=cm.tab20,
            alpha=0.6,
            s=12,
        )

    if out_path is not None:
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
