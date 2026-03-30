"""Abstract base class for outlier detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class OutlierDetector(ABC):
    """Base class for outlier / starting-node detectors.

    Subclasses implement :meth:`detect` to identify anomalous starting nodes
    from the pre-computed neighborhood embeddings.
    """

    @abstractmethod
    def detect(
        self,
        embs: list[torch.Tensor],
        model: object,
        real_anchors: list[int],
        neighborhoods: list,
        **kwargs: object,
    ) -> tuple[set[int], np.ndarray]:
        """Detect outlier neighborhoods and return starting nodes.

        Parameters
        ----------
        embs:
            List of embedding tensors (one per batch), as produced by
            :func:`embed_neighs`.  Each tensor has shape
            ``(batch_size, hidden_dim)``.
        model:
            The :class:`~minomaly.models.order_embedder.OrderEmbedder` instance,
            providing ``predict()`` and ``clf_model()`` methods.
        real_anchors:
            List of original node IDs corresponding to each neighborhood
            (in the same order as the flattened embeddings).
        neighborhoods:
            List of neighborhood graphs (NetworkX or SubgraphView), in the
            same order as the flattened embeddings.
        **kwargs:
            Additional detector-specific parameters.

        Returns
        -------
        tuple[set[int], np.ndarray]
            ``(starting_node_set, outlier_embeddings_np)`` where
            ``starting_node_set`` is the set of node IDs identified as
            anomalous starting points and ``outlier_embeddings_np`` is a
            numpy array of embeddings for the detected outlier neighborhoods.
        """
        ...
