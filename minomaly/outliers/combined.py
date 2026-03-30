"""Combined outlier detector that merges results from multiple detectors."""

from __future__ import annotations

import logging

import numpy as np
import torch

from minomaly.outliers.base import OutlierDetector

logger = logging.getLogger(__name__)


class CombinedDetector(OutlierDetector):
    """Combine multiple outlier detectors via union or intersection.

    Parameters
    ----------
    detectors:
        List of :class:`OutlierDetector` instances to run.
    combine:
        How to merge the starting-node sets:

        * ``"union"`` -- a node is a starting node if **any** detector flags it.
        * ``"intersection"`` -- a node is a starting node only if **all**
          detectors flag it.
    """

    def __init__(
        self,
        detectors: list[OutlierDetector],
        combine: str = "union",
    ) -> None:
        if combine not in ("union", "intersection"):
            raise ValueError(
                f"combine must be 'union' or 'intersection', got {combine!r}"
            )
        self.detectors = detectors
        self.combine = combine

    def detect(
        self,
        embs: list[torch.Tensor],
        model: object,
        real_anchors: list[int],
        neighborhoods: list,
        **kwargs: object,
    ) -> tuple[set[int], np.ndarray]:
        """Run all sub-detectors and merge their results.

        Parameters
        ----------
        embs:
            List of embedding batch tensors.
        model:
            The order-embedding model.
        real_anchors:
            Per-neighborhood original anchor node IDs.
        neighborhoods:
            Per-neighborhood graph objects.
        **kwargs:
            Forwarded to each sub-detector's ``detect`` method.

        Returns
        -------
        tuple[set[int], np.ndarray]
            ``(combined_starting_nodes, combined_outlier_embeddings)``
        """
        if not self.detectors:
            return set(), np.empty((0,))

        all_node_sets: list[set[int]] = []
        all_emb_arrays: list[np.ndarray] = []

        for detector in self.detectors:
            nodes, det_embs = detector.detect(
                embs, model, real_anchors, neighborhoods, **kwargs
            )
            all_node_sets.append(nodes)
            if det_embs.ndim >= 2 and det_embs.shape[0] > 0:
                all_emb_arrays.append(det_embs)

        # Combine node sets
        if self.combine == "union":
            combined_nodes: set[int] = set()
            for node_set in all_node_sets:
                combined_nodes |= node_set
        else:  # intersection
            combined_nodes = all_node_sets[0].copy()
            for node_set in all_node_sets[1:]:
                combined_nodes &= node_set

        # Combine outlier embeddings (concatenate all unique outlier embeddings)
        if all_emb_arrays:
            combined_embs = np.concatenate(all_emb_arrays, axis=0)
            # Remove duplicates by rounding and using unique rows
            _, unique_idx = np.unique(
                np.round(combined_embs, decimals=6), axis=0, return_index=True
            )
            combined_embs = combined_embs[np.sort(unique_idx)]
        else:
            combined_embs = np.empty((0,))

        logger.info(
            "CombinedDetector (%s): %d starting nodes from %d detectors",
            self.combine,
            len(combined_nodes),
            len(self.detectors),
        )

        return combined_nodes, combined_embs
