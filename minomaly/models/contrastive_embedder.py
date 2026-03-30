"""Contrastive cross-attention embedder for subgraph containment.

Replaces SPMiner's order-embedding approach with a learned similarity
function trained via InfoNCE contrastive loss.  A GATv2 backbone provides
dynamic attention over graph structure.

Novel contributions vs SPMiner:
1. Learned similarity MLP (|a-b|, a*b features) instead of fixed order-violation
2. InfoNCE loss uses full-batch negatives instead of margin-based hinge
3. GATv2 attention reveals which edges matter for containment
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minomaly.models.base import EmbeddingModel
from minomaly.registry import EMBEDDERS, ENCODERS
from minomaly.training.contrastive_loss import info_nce_loss


@EMBEDDERS.register("contrastive")
class ContrastiveEmbedder(EmbeddingModel):
    """Contrastive embedding model for subgraph containment.

    Uses a learned similarity function on graph-level embeddings instead
    of the rigid coordinate-wise order-violation formula.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (1 for anchor indicator).
    hidden_dim : int
        Graph embedding dimension.
    temperature : float
        InfoNCE temperature (smaller = sharper contrast).
    margin : float
        Margin for the auxiliary hinge loss component.
    encoder_name : str
        Registered encoder name (default ``"gatv2"``).
    **encoder_kwargs
        Forwarded to the encoder constructor.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        temperature: float = 0.1,
        margin: float = 0.1,
        encoder_name: str = "gatv2",
        **encoder_kwargs,
    ) -> None:
        super().__init__()

        encoder_cls = ENCODERS[encoder_name]
        self.emb_model = encoder_cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **encoder_kwargs,
        )

        self.temperature = temperature
        self.margin = margin
        self.hidden_dim = hidden_dim

        # Learned similarity MLP: takes element-wise |diff| and product
        # of two embeddings → scalar similarity score.
        self.sim_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Classifier head (same interface as OrderEmbedder)
        self.clf_model = nn.Sequential(
            nn.Linear(1, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(
        self, emb_as: torch.Tensor, emb_bs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return emb_as, emb_bs

    def _asymmetric_distance(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric containment distance: order-violation + learned residual.

        Combines the proven order-embedding violation (directional, captures
        containment) with a learned correction from the sim_mlp.
        Output is bounded via sigmoid to [0, 1].  0 = b is subgraph of a.
        """
        # Order-embedding violation (asymmetric, proven to work)
        order_viol = torch.sum(torch.clamp(emb_b - emb_a, min=0) ** 2, dim=-1)
        # Learned residual from interaction features
        diff = torch.abs(emb_a - emb_b)
        prod = emb_a * emb_b
        features = torch.cat([diff, prod], dim=-1)
        residual = self.sim_mlp(features).squeeze(-1)
        # Combine and bound to [0, 1]
        return torch.sigmoid(order_viol + residual)

    def predict(
        self, pred: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Asymmetric containment distance.

        Parameters
        ----------
        pred : tuple[Tensor, Tensor]
            ``(emb_as, emb_bs)`` of shape ``(N, D)`` each.

        Returns
        -------
        Tensor
            Shape ``(N,)`` — distance in [0, 1].  Small = likely subgraph.
        """
        emb_as, emb_bs = pred
        return self._asymmetric_distance(emb_as, emb_bs)

    def batch_predict(
        self, emb_as: torch.Tensor, emb_bs: torch.Tensor,
    ) -> torch.Tensor:
        """Batch pairwise asymmetric distances.

        Parameters
        ----------
        emb_as : Tensor
            Shape ``(B, D)`` — reference (supergraph) embeddings.
        emb_bs : Tensor
            Shape ``(N, D)`` — query (subgraph) embeddings.

        Returns
        -------
        Tensor
            Shape ``(N, B)`` — distances in [0, 1] (small = likely subgraph).
        """
        a_exp = emb_as.unsqueeze(0).expand(emb_bs.shape[0], -1, -1)
        b_exp = emb_bs.unsqueeze(1).expand(-1, emb_as.shape[0], -1)
        return self._asymmetric_distance(a_exp, b_exp)

    def criterion(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        intersect_embs: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Combined InfoNCE + similarity + classification loss.

        1. **InfoNCE**: contrastive loss pulling positive pairs close in
           embedding space using the full batch as negatives.
        2. **Similarity hinge**: minimize sim_mlp output for positives,
           push above margin for negatives.
        3. **Classification**: train clf_model to predict containment
           from the similarity score.
        """
        emb_as, emb_bs = pred

        # 1. InfoNCE contrastive loss on raw embeddings
        contrastive = info_nce_loss(
            emb_as, emb_bs, labels, self.temperature,
        )

        # 2. Asymmetric distance loss (BCE since output is in [0,1])
        # Target: 0 for positives (subgraph), 1 for negatives
        dist = self.predict(pred)  # (N,) in [0, 1]
        target = 1.0 - labels  # 0→1 (neg=far), 1→0 (pos=close)
        dist_loss = F.binary_cross_entropy(dist, target)

        # 3. Classification loss (trains clf_model jointly)
        clf_out = self.clf_model(dist.detach().unsqueeze(1))
        clf_loss = F.nll_loss(clf_out, labels.long())

        N = dist.shape[0]
        return contrastive + dist_loss * N + clf_loss * N
