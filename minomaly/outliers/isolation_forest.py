"""IsolationForest-based outlier detector.

Port of ``get_outlier_neighs`` from ``code-original/minomaly_struct/utils.py``.
Uses scikit-learn's IsolationForest on flattened CPU numpy embeddings, filtered
to neighborhoods below a maximum size.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from minomaly.outliers.base import OutlierDetector
from minomaly.registry import OUTLIERS

logger = logging.getLogger(__name__)


@OUTLIERS.register("isolation_forest")
class IsolationForestDetector(OutlierDetector):
    """Detect outlier neighborhoods using sklearn IsolationForest.

    Parameters
    ----------
    max_neigh_len:
        Maximum neighborhood size.  Neighborhoods larger than this are
        excluded from IsolationForest fitting and prediction.
    contamination:
        The proportion of outliers in the data set.  Passed directly to
        :class:`~sklearn.ensemble.IsolationForest`.  Use ``"auto"`` to let
        sklearn decide.
    random_state:
        Random seed for the IsolationForest.
    """

    def __init__(
        self,
        max_neigh_len: int,
        contamination: Union[float, str] = 0.2,
        random_state: int = 42,
    ) -> None:
        self.max_neigh_len = max_neigh_len
        self.contamination = contamination
        self.random_state = random_state

    def detect(
        self,
        embs: list[torch.Tensor],
        model: object,
        real_anchors: list[int],
        neighborhoods: list,
        **kwargs: object,
    ) -> tuple[set[int], np.ndarray]:
        """Detect outlier neighborhoods via IsolationForest.

        Parameters
        ----------
        embs:
            List of embedding batch tensors.
        model:
            The order-embedding model (unused by this detector but required
            by the interface).
        real_anchors:
            Per-neighborhood original anchor node IDs.
        neighborhoods:
            Per-neighborhood graph objects (NetworkX graphs).
        **kwargs:
            Unused.

        Returns
        -------
        tuple[set[int], np.ndarray]
            ``(starting_nodes, outlier_embeddings)``
        """
        # Use pre-computed embs_np if passed, otherwise flatten
        if "embs_np" in kwargs and kwargs["embs_np"] is not None:
            embs_np = kwargs["embs_np"]
        else:
            embs_np = torch.cat(embs, dim=0).cpu().numpy()

        real_anchors_np = np.array(real_anchors, dtype=int)

        # Filter by neighborhood size (number of nodes)
        def _num_nodes(n):
            if hasattr(n, "num_nodes") and n.num_nodes is not None:
                return n.num_nodes  # PyG Data
            if hasattr(n, "__len__"):
                return len(n)
            return 0

        neigh_len_cond = np.array(
            [_num_nodes(n) <= self.max_neigh_len for n in neighborhoods]
        )

        embs_filtered = embs_np[neigh_len_cond]
        print(f"IsolationForest: {embs_filtered.shape[0]} neighborhoods after filtering (max_len={self.max_neigh_len})")

        if len(embs_filtered) == 0:
            logger.warning(
                "IsolationForest: no neighborhoods within max_neigh_len=%d",
                self.max_neigh_len,
            )
            return set(), np.empty((0,))

        # Fit and predict
        print("IsolationForest: fitting...")
        clf = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        y_pred = clf.fit_predict(embs_filtered)
        print(f"IsolationForest: done, {np.sum(y_pred == -1)} outliers")

        # Collect outlier embeddings
        outlier_embs = embs_filtered[y_pred == -1]

        # Map back to anchor node IDs
        anchors_filtered = real_anchors_np[neigh_len_cond]
        starting_nodes: set[int] = set(
            int(node) for node in anchors_filtered[y_pred == -1]
        )

        logger.info(
            "IsolationForest detector: %d starting nodes from %d outlier neighborhoods",
            len(starting_nodes),
            int(np.sum(y_pred == -1)),
        )

        return starting_nodes, outlier_embs
