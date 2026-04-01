"""Context Clustering — discover context groups from embeddings.

Clusters context embeddings into C groups. Each group represents a
"context" — neighborhoods sharing similar structure + attribute patterns.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans, DBSCAN


class ContextClustering:
    """Cluster context embeddings into context groups.

    Parameters
    ----------
    n_clusters:
        Number of context clusters (for KMeans).
    method:
        Clustering method: "kmeans" or "dbscan".
    """

    def __init__(self, n_clusters: int = 10, method: str = "kmeans") -> None:
        self.n_clusters = n_clusters
        self.method = method
        self.labels_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self._model = None

    def fit(self, embeddings: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Cluster embeddings.

        Parameters
        ----------
        embeddings:
            (K, D) context embeddings.

        Returns
        -------
        tuple:
            (labels (K,), centers (C, D))
        """
        X = embeddings.cpu().numpy()

        if self.method == "kmeans":
            model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            centers = model.cluster_centers_
        elif self.method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(X)
            # Compute centers per cluster
            unique = set(labels) - {-1}
            centers = np.zeros((len(unique), X.shape[1]))
            for i, c in enumerate(sorted(unique)):
                centers[i] = X[labels == c].mean(axis=0)
            self.n_clusters = len(unique)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self._model = model
        self.labels_ = labels
        self.centers_ = centers

        # Report cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        for c, cnt in zip(unique, counts):
            print(f"  [context] cluster {c}: {cnt} neighborhoods")

        return labels, centers

    def predict(self, embedding: torch.Tensor) -> int:
        """Assign a single embedding to the nearest cluster.

        Parameters
        ----------
        embedding:
            (D,) or (1, D) context embedding.

        Returns
        -------
        int:
            Cluster ID.
        """
        x = embedding.cpu().numpy().reshape(1, -1)
        dists = np.linalg.norm(self.centers_ - x, axis=1)
        return int(np.argmin(dists))

    def predict_batch(self, embeddings: torch.Tensor) -> np.ndarray:
        """Assign multiple embeddings to nearest clusters.

        Parameters
        ----------
        embeddings:
            (N, D) context embeddings.

        Returns
        -------
        ndarray:
            (N,) cluster IDs.
        """
        X = embeddings.cpu().numpy()
        # Pairwise distances to centers
        # (N, 1, D) - (1, C, D) → (N, C)
        dists = np.linalg.norm(
            X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :],
            axis=2,
        )
        return np.argmin(dists, axis=1)
