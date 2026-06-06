"""Order-embedding model for subgraph containment prediction.

Port of ``OrderEmbedder`` from ``code-original/common/models.py``.  The model
wraps a graph encoder (looked up from the :data:`ENCODERS` registry) and a
lightweight classifier to predict whether one graph is a subgraph of another
based on order-embedding violations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minomaly.models.base import EmbeddingModel
from minomaly.registry import EMBEDDERS, ENCODERS
from minomaly.utils.device import get_device


@EMBEDDERS.register("order")
class OrderEmbedder(EmbeddingModel):
    """Order-embedding model for subgraph containment.

    Contains:

    * ``emb_model``: A :class:`~minomaly.models.base.GraphEncoder` looked up
      from the :data:`ENCODERS` registry (default ``"skip_last_gnn"``).
    * ``clf_model``: ``nn.Sequential(Linear(1, 2), LogSoftmax(dim=-1))``
      classifying the scalar order-embedding distance into two classes.

    Prediction
    ----------
    Given embeddings ``emb_as`` and ``emb_bs``:

    .. math::

        e = \\sum \\max(0,\\; \\text{emb\\_bs} - \\text{emb\\_as})^2

    A small ``e`` means *b* is likely a subgraph of *a*.

    Loss
    ----
    For **positive** pairs (true subgraph relationship), minimises ``e``.
    For **negative** pairs, pushes ``e`` above ``margin``.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        margin: float = 0.1,
        encoder_name: str = "skip_last_gnn",
        **encoder_kwargs,
    ) -> None:
        super().__init__()

        # Build the graph encoder from the registry.
        encoder_cls = ENCODERS[encoder_name]
        self.emb_model = encoder_cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **encoder_kwargs,
        )

        self.margin = margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(
            nn.Linear(1, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(
        self,
        emb_as: torch.Tensor,
        emb_bs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identity forward -- returns the pair of embeddings unchanged.

        This is consistent with the original implementation where the
        ``forward`` method simply passes embeddings through so that
        ``predict`` and ``criterion`` can operate on them.
        """
        return emb_as, emb_bs

    def predict(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute order-embedding violation scores.

        Parameters
        ----------
        pred:
            Tuple ``(emb_as, emb_bs)`` of graph-level embeddings.

        Returns
        -------
        torch.Tensor
            Per-pair violation ``e``.  Small values indicate that *b* is
            likely a subgraph of *a*.
        """
        emb_as, emb_bs = pred
        e = torch.sum(
            torch.max(
                torch.zeros_like(emb_as, device=emb_as.device),
                emb_bs - emb_as,
            )
            ** 2,
            dim=1,
        )
        return e

    def batch_predict(
        self,
        emb_as: torch.Tensor,
        emb_bs: torch.Tensor,
    ) -> torch.Tensor:
        """Batch pairwise order-embedding violations.

        Parameters
        ----------
        emb_as:
            Shape ``(B, D)`` — reference (neighborhood) embeddings.
        emb_bs:
            Shape ``(N, D)`` — query (beam) embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``(N, B)`` — violations.
        """
        N, D = emb_bs.shape
        B = emb_as.shape[0]
        # Memory budget: keep the diff tensor below ~256 MB so that the
        # full (N, B, D) intermediate never materialises on dense graphs
        # with large reference sets (k = 20k–150k).
        # 256 MB / (4 bytes * D) elements per slice = chunk along B.
        budget_bytes = 256 * 1024 * 1024
        chunk = max(1, budget_bytes // max(1, 4 * N * D))
        if chunk >= B:
            diff = emb_bs.unsqueeze(1) - emb_as.unsqueeze(0)
            return torch.sum(torch.clamp(diff, min=0) ** 2, dim=2)
        out = torch.empty(N, B, device=emb_bs.device, dtype=emb_bs.dtype)
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            slice_diff = emb_bs.unsqueeze(1) - emb_as[start:end].unsqueeze(0)
            out[:, start:end] = torch.sum(
                torch.clamp(slice_diff, min=0) ** 2, dim=2
            )
            del slice_diff
        return out

    def criterion(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        intersect_embs: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Order-embedding loss with margin.

        For positive pairs (``labels == 1``), the violation ``e`` is
        minimised directly.  For negative pairs (``labels == 0``), the
        loss is ``max(0, margin - e)`` so that ``e`` is pushed above the
        margin.

        Parameters
        ----------
        pred:
            Tuple ``(emb_as, emb_bs)`` of embeddings.
        intersect_embs:
            Unused (kept for API compatibility).
        labels:
            Binary labels (``1`` = positive / true subgraph pair,
            ``0`` = negative).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        emb_as, emb_bs = pred

        e = torch.sum(
            torch.clamp(emb_bs - emb_as, min=0) ** 2,
            dim=1,
        )

        pos_mask = labels == 1
        neg_mask = labels == 0

        # Positive: minimize violation (mean-normalized)
        pos_loss = e[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=emb_as.device)

        # Negative: softplus(margin - e) for continuous gradients
        neg_loss = F.softplus(self.margin - e[neg_mask]).mean() if neg_mask.any() else torch.tensor(0.0, device=emb_as.device)

        # Train clf_model jointly (NLL on violation → class prediction)
        clf_pred = self.clf_model(e.detach().unsqueeze(1))
        clf_loss = F.nll_loss(clf_pred, labels.long())

        return pos_loss + neg_loss + clf_loss
