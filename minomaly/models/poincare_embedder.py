"""Poincaré embedding model for subgraph containment in hyperbolic space.

Subgraph containment maps naturally to hyperbolic geometry: smaller subgraphs
embed closer to the origin, larger supergraphs sit near the boundary of the
Poincaré ball.  Distance in the ball encodes the containment relation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from minomaly.models.base import EmbeddingModel
from minomaly.models.hyperbolic_math import (
    exp_map_zero,
    log_map_zero,
    poincare_distance,
    poincare_distance_batch,
    project,
)
from minomaly.registry import EMBEDDERS, ENCODERS


@EMBEDDERS.register("poincare")
class PoincareEmbedder(EmbeddingModel):
    """Poincaré-ball embedding model for subgraph containment.

    Uses hyperbolic distance instead of Euclidean order-embedding violation.
    Small distance between embeddings indicates a subgraph relation.

    The encoder can be:
    - ``"hyperbolic_gnn"``: native hyperbolic encoder (outputs in the ball).
    - ``"skip_last_gnn"``: Euclidean encoder whose output is projected into
      the ball via ``exp_map_zero``.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (default 1 for anchor indicator).
    hidden_dim : int
        Embedding dimension.
    margin : float
        Margin for the hinge loss on negative pairs.
    curvature : float
        Initial curvature of the Poincaré ball (> 0).
    learnable_curvature : bool
        Whether curvature is a learnable parameter.
    encoder_name : str
        Registered encoder to use as backbone.
    **encoder_kwargs
        Forwarded to the encoder constructor.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        margin: float = 0.5,
        curvature: float = 1.0,
        learnable_curvature: bool = True,
        encoder_name: str = "hyperbolic_gnn",
        **encoder_kwargs,
    ) -> None:
        super().__init__()

        self.encoder_name = encoder_name
        self._is_hyperbolic_encoder = encoder_name == "hyperbolic_gnn"

        # Build encoder from registry
        encoder_cls = ENCODERS[encoder_name]
        if self._is_hyperbolic_encoder:
            self.emb_model = encoder_cls(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                curvature_init=curvature,
                learnable_curvature=learnable_curvature,
                **encoder_kwargs,
            )
        else:
            self.emb_model = encoder_cls(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                **encoder_kwargs,
            )

        # Curvature (shared with encoder if hyperbolic, separate otherwise)
        if self._is_hyperbolic_encoder:
            # Share curvature with the encoder
            self.c = self.emb_model.c
        elif learnable_curvature:
            self.c = nn.Parameter(torch.tensor([curvature]))
        else:
            self.register_buffer("c", torch.tensor([curvature]))

        self.margin = margin

        # Learned Euclidean→tangent projection: maps arbitrary-magnitude
        # encoder output to small-norm vectors suitable for exp_map_zero.
        if not self._is_hyperbolic_encoder:
            self._euclidean_to_tangent = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),  # bounds output to [-1, 1] per dim
            )
        else:
            self._euclidean_to_tangent = nn.Identity()


        # Classifier head: takes scaled scalar distance → 2-class logits
        self.clf_model = nn.Sequential(
            nn.Linear(1, 2),
            nn.LogSoftmax(dim=-1),
        )

    def _get_c(self, device: torch.device | None = None) -> torch.Tensor:
        """Return curvature, clamped to [0.1, 10.0] to avoid degenerate geometry."""
        c = torch.clamp(torch.abs(self.c), min=0.1, max=10.0)
        if device is not None and c.device != device:
            c = c.to(device)
        return c

    def _to_ball(self, emb: torch.Tensor) -> torch.Tensor:
        """Project Euclidean encoder output into the Poincaré ball.

        For Euclidean encoders: apply the learned projection MLP then
        exp_map into the ball.  The MLP learns to scale features into the
        right magnitude range for exp_map_zero (avoiding tanh saturation).
        """
        if self._is_hyperbolic_encoder:
            return emb  # already in the ball
        c = self._get_c(emb.device)
        emb = self._euclidean_to_tangent(emb)  # learned projection to small norms
        return project(exp_map_zero(emb, c), c)

    def forward(
        self, emb_as: torch.Tensor, emb_bs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Identity forward (API compat)."""
        return emb_as, emb_bs

    def predict(
        self, pred: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Poincaré distance between embedding pairs.

        Parameters
        ----------
        pred : tuple[Tensor, Tensor]
            ``(emb_as, emb_bs)`` both of shape ``(N, D)``, in the ball.

        Returns
        -------
        Tensor
            Shape ``(N,)`` — geodesic distances.  Small distance means
            *b* is likely a subgraph of *a*.
        """
        emb_as, emb_bs = pred
        c = self._get_c(emb_as.device)
        emb_as = self._to_ball(emb_as)
        emb_bs = self._to_ball(emb_bs)
        return poincare_distance(emb_as, emb_bs, c)

    def batch_predict(
        self, emb_as: torch.Tensor, emb_bs: torch.Tensor,
    ) -> torch.Tensor:
        """Batch pairwise Poincaré distances.

        Parameters
        ----------
        emb_as : Tensor
            Shape ``(B, D)`` — reference embeddings (neighborhoods).
        emb_bs : Tensor
            Shape ``(N, D)`` — beam embeddings.

        Returns
        -------
        Tensor
            Shape ``(N, B)`` — pairwise scaled distances.
        """
        c = self._get_c(emb_as.device)
        emb_as = self._to_ball(emb_as)
        emb_bs = self._to_ball(emb_bs)
        return poincare_distance_batch(emb_bs, emb_as, c)

    def criterion(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        intersect_embs: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Hyperbolic distance loss + clf_model classification loss.

        1. Distance hinge loss on scaled distances.
        2. Classification loss to train clf_model jointly.
        """
        d = self.predict(pred)  # scaled distances

        # 1. Distance hinge loss
        hinge = d.clone()
        neg_mask = labels == 0
        hinge[neg_mask] = torch.clamp(self.margin - d[neg_mask], min=0)
        distance_loss = hinge.sum()

        # 2. Classification loss (NLLLoss since clf_model outputs LogSoftmax)
        clf_out = self.clf_model(d.detach().unsqueeze(1))  # (N, 2)
        clf_loss = torch.nn.functional.nll_loss(clf_out, labels.long())

        return distance_loss + clf_loss * d.shape[0]
