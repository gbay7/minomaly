"""Visualization callback that generates plots during search and training."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from minomaly.callbacks.base import Callback
from minomaly.registry import CALLBACKS


@CALLBACKS.register("visualization")
class VisualizationCallback(Callback):
    """Generates plots at key moments during search and training.

    Parameters
    ----------
    output_dir : str
        Root directory for generated plots.
    reduction_method : str | None
        Dimensionality reduction to apply before scatter plots.
        ``"PCA"`` or ``"TSNE"`` (case-insensitive), or *None* to skip.
    """

    def __init__(
        self,
        output_dir: str = "plots",
        reduction_method: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.reduction_method = reduction_method
        self._search_embs: list[np.ndarray] = []
        self._search_labels: list[int] = []
        self._training_losses: list[float] = []

    # ── Search hooks ────────────────────────────────────────────────────

    def on_search_step(
        self,
        step: int,
        beam_sets: list,
        verified: list,
        new_verified_count: int,
    ) -> None:
        """Collect beam embeddings for later visualization."""
        for bs in beam_sets:
            for beam in bs:
                emb = getattr(beam, "emb", None)
                if emb is not None:
                    emb_np = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
                    if emb_np.ndim == 1:
                        self._search_embs.append(emb_np)
                        is_verified = int(beam in verified) if verified else 0
                        self._search_labels.append(is_verified)

    def on_search_end(
        self,
        all_verified: set,
        patterns: list,
        stats: dict,
    ) -> None:
        """Generate scatter plots and export pattern visualizations."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Scatter plot of search embeddings
        if self._search_embs:
            embs_np = np.stack(self._search_embs)
            labels = np.array(self._search_labels)
            legend = {0: "non-verified", 1: "verified"}
            out_path = os.path.join(self.output_dir, "analyze_search.png")
            self._scatter_embs(
                embs_np,
                labels=labels,
                legend=legend,
                out_path=out_path,
            )

        # Export detected patterns
        if patterns:
            cluster_dir = os.path.join(self.output_dir, "cluster")
            self._export_patterns(patterns, out_dir=cluster_dir)

    # ── Training hooks ──────────────────────────────────────────────────

    def on_training_batch_end(
        self, batch_idx: int, loss: float, metrics: dict
    ) -> None:
        self._training_losses.append(loss)

    def on_training_end(self, model: Any, final_metrics: dict) -> None:
        """Plot training loss curve."""
        if not self._training_losses:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "training_loss.png")
        plt.figure()
        plt.plot(self._training_losses, linewidth=0.8)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    # ── Internal helpers ────────────────────────────────────────────────

    def _scatter_embs(
        self,
        embs_np: np.ndarray,
        labels: np.ndarray | None = None,
        legend: dict | str | None = None,
        out_path: str | None = None,
    ) -> None:
        """Scatter plot with optional dimensionality reduction."""
        method = (self.reduction_method or "").upper()
        if method == "PCA" and embs_np.shape[1] > 2:
            embs_np = PCA(n_components=2).fit_transform(embs_np)
        elif method == "TSNE" and embs_np.shape[1] > 2:
            embs_np = TSNE(n_components=2).fit_transform(embs_np)

        plt.figure()
        if labels is not None and legend is not None and isinstance(legend, dict):
            unique_labels = np.unique(labels)
            for label in unique_labels:
                idx = np.where(labels == label)[0]
                lbl = legend.get(label, str(label))
                plt.scatter(
                    embs_np[idx, 0],
                    embs_np[idx, 1],
                    label=lbl,
                    cmap=cm.tab20,
                    alpha=0.6,
                    s=12,
                )
        else:
            lbl = legend if isinstance(legend, str) else None
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

    def _export_patterns(
        self,
        patterns: list,
        out_dir: str = "plots/cluster",
        node_anchored: bool = True,
    ) -> None:
        """Draw and save pattern graphs as PDFs using NetworkX drawing."""
        import networkx as nx
        from collections import defaultdict

        os.makedirs(out_dir, exist_ok=True)
        count_by_size: dict[int, int] = defaultdict(int)

        for item in patterns:
            if isinstance(item, tuple) and len(item) == 2:
                graph, anchor = item
            else:
                graph = item
                anchor = None

            # Convert to nx.Graph if possible
            if hasattr(graph, "to_nx"):
                graph = graph.to_nx()
            if not isinstance(graph, nx.Graph):
                continue

            node_colors = [
                "red"
                if (node_anchored and anchor is not None and node == anchor)
                else "blue"
                for node in graph.nodes
            ]
            plt.figure()
            if node_anchored and anchor is not None:
                nx.draw(graph, with_labels=True, node_color=node_colors)
            else:
                nx.draw(graph, node_color=node_colors)

            size = len(graph)
            path = os.path.join(
                out_dir, f"{size}-{count_by_size[size]}.pdf"
            )
            plt.savefig(path)
            plt.close()
            count_by_size[size] += 1
