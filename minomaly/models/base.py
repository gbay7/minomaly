"""Abstract base classes for graph embedding models.

:class:`GraphEncoder`
    ABC for GNN backbones that produce graph-level embeddings.

:class:`EmbeddingModel`
    ABC for order-embedding models that wrap an encoder and a classifier
    to assess subgraph containment in the embedding space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


class GraphEncoder(ABC, nn.Module):
    """ABC for GNN backbones producing graph-level embeddings."""

    @abstractmethod
    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Encode a (batched) graph and return graph-level embeddings.

        Parameters
        ----------
        data:
            A single :class:`Data` or a :class:`Batch` of graphs.

        Returns
        -------
        torch.Tensor
            Graph-level embedding tensor of shape ``(batch_size, hidden_dim)``.
        """
        ...


class EmbeddingModel(ABC, nn.Module):
    """ABC for order-embedding models.

    Subclasses must define :attr:`emb_model` (a :class:`GraphEncoder`)
    and implement :meth:`predict` and :meth:`criterion`.
    """

    emb_model: GraphEncoder

    @abstractmethod
    def predict(
        self, pred: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute order-embedding violation scores for a pair of embeddings.

        Parameters
        ----------
        pred:
            A tuple ``(emb_as, emb_bs)`` of graph-level embeddings.

        Returns
        -------
        torch.Tensor
            Per-pair violation scores.
        """
        ...

    @abstractmethod
    def criterion(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        intersect_embs: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the order-embedding loss.

        Parameters
        ----------
        pred:
            A tuple ``(emb_as, emb_bs)`` of graph-level embeddings.
        intersect_embs:
            Intersection embeddings (unused in the standard order-embedding
            formulation but kept for API compatibility).
        labels:
            Binary labels indicating positive (1 = true subgraph) or
            negative (0) pairs.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        ...

    def batch_predict(
        self,
        emb_as: torch.Tensor,
        emb_bs: torch.Tensor,
    ) -> torch.Tensor:
        """Batch pairwise violation/distance scores.

        Parameters
        ----------
        emb_as:
            Shape ``(B, D)`` — reference embeddings (e.g. neighborhoods).
        emb_bs:
            Shape ``(N, D)`` — query embeddings (e.g. beams).

        Returns
        -------
        torch.Tensor
            Shape ``(N, B)`` — pairwise scores.  Small values indicate
            that the query is likely a subgraph of the reference.

        The default implementation loops over pairs using :meth:`predict`.
        Subclasses should override for efficiency.
        """
        # Default: expand and call predict
        # emb_bs (N, D) vs emb_as (B, D) → (N, B)
        N = emb_bs.shape[0]
        B = emb_as.shape[0]
        emb_bs_exp = emb_bs.unsqueeze(1).expand(N, B, -1).reshape(N * B, -1)
        emb_as_exp = emb_as.unsqueeze(0).expand(N, B, -1).reshape(N * B, -1)
        scores = self.predict((emb_as_exp, emb_bs_exp))
        return scores.reshape(N, B)
