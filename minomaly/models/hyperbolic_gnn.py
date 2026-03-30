"""Hyperbolic Graph Neural Network (HGNN) encoder.

Uses the tangent-space aggregation approach from HGCN:

1. Map node embeddings from Poincare ball to tangent space at origin
   (:func:`log_map_zero`).
2. Aggregate neighbours in tangent space (standard message passing).
3. Map back to Poincare ball (:func:`exp_map_zero`).
4. Pool to graph-level embedding.

Reference
---------
Chami et al., "Hyperbolic Graph Convolutional Neural Networks" (NeurIPS 2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from minomaly.models.base import GraphEncoder
from minomaly.models.hyperbolic_math import (
    BALL_EPS,
    MIN_NORM,
    exp_map_zero,
    lambda_x,
    log_map_zero,
    poincare_distance,
    project,
)
from minomaly.registry import ENCODERS


class HyperbolicSAGEConv(nn.Module):
    """A single hyperbolic SAGE-style convolution layer.

    Steps:

    1. Log-map all node embeddings to the tangent space at the origin.
    2. Standard SAGE aggregation in tangent space (linear transform on
       neighbour features + scatter-add aggregation).
    3. Concatenate with the self-transform and apply a linear update + ReLU.
    4. Exp-map the result back to the Poincare ball.

    Parameters
    ----------
    in_channels:
        Input feature dimension (in tangent-space / ball coordinates).
    out_channels:
        Output feature dimension.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_update = nn.Linear(2 * out_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Run one hyperbolic message-passing step.

        Parameters
        ----------
        x:
            Node embeddings in the Poincare ball, shape ``(num_nodes, D)``.
        edge_index:
            Edge connectivity in COO format ``(2, num_edges)``.
        c:
            Positive curvature scalar tensor.

        Returns
        -------
        torch.Tensor
            Updated node embeddings in the Poincare ball.
        """
        # Map to tangent space at the origin
        x_tangent = log_map_zero(x, c)

        # Neighbour aggregation in tangent space
        row, col = edge_index
        # Remove self-loops for aggregation
        mask = row != col
        row, col = row[mask], col[mask]

        neigh_msg = self.lin_neigh(x_tangent[col])
        # Scatter-add aggregation
        aggr = torch.zeros_like(x_tangent)
        aggr.scatter_add_(0, row.unsqueeze(-1).expand_as(neigh_msg), neigh_msg)

        # Self transform
        self_msg = self.lin_self(x_tangent)

        # Concat + update
        combined = torch.cat([aggr, self_msg], dim=-1)
        updated = self.lin_update(combined)
        updated = F.relu(updated)

        # Map back to the ball
        return project(exp_map_zero(updated, c), c)


@ENCODERS.register("hyperbolic_gnn")
class HyperbolicGNN(GraphEncoder):
    """GNN operating in the Poincare ball via tangent-space aggregation.

    Architecture:

    * ``pre_mp``: ``Linear(input_dim, hidden_dim)`` in tangent space,
      followed by :func:`exp_map_zero` to project into the ball.
    * *n_layers* :class:`HyperbolicSAGEConv` layers (log -> aggregate -> exp).
    * Pooling: :func:`log_map_zero` all nodes, ``global_mean_pool`` in
      tangent space, :func:`exp_map_zero` back to ball.
    * ``post_mp``: :func:`log_map_zero` -> MLP -> :func:`exp_map_zero`
      (final embedding lives on the ball).

    Parameters
    ----------
    input_dim:
        Node feature dimension (default 1 for anchor indicator).
    hidden_dim:
        Hidden dimension throughout the network.
    output_dim:
        Output embedding dimension.
    n_layers:
        Number of hyperbolic convolution layers (default 8).
    dropout:
        Dropout rate applied between convolution layers.
    curvature_init:
        Initial value for the curvature parameter *c* (default 1.0).
    learnable_curvature:
        If ``True`` the curvature is a learnable ``nn.Parameter``;
        otherwise it is registered as a fixed buffer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 8,
        dropout: float = 0.0,
        curvature_init: float = 1.0,
        learnable_curvature: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Learnable (or fixed) curvature
        if learnable_curvature:
            self.c = nn.Parameter(torch.tensor([curvature_init]))
        else:
            self.register_buffer("c", torch.tensor([curvature_init]))

        self.n_layers = n_layers
        self.dropout = dropout

        # Pre-message-passing: project input features into the ball
        self.pre_mp = nn.Linear(input_dim, hidden_dim)

        # Hyperbolic convolution layers
        self.convs = nn.ModuleList(
            [HyperbolicSAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )

        # Post-message-passing: MLP in tangent space
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_curvature(self) -> torch.Tensor:
        """Return the curvature ensuring it is strictly positive."""
        return torch.clamp(torch.abs(self.c), min=0.1, max=10.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Encode a (batched) graph and return graph-level embeddings.

        Parameters
        ----------
        data:
            A PyG ``Data`` or ``Batch`` object with ``x``, ``edge_index``,
            and ``batch`` attributes.

        Returns
        -------
        torch.Tensor
            Graph-level embedding tensor of shape ``(batch_size, output_dim)``
            lying on the Poincare ball (``c * ||emb||^2 < 1``).
        """
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch: torch.Tensor = data.batch

        c = self._get_curvature().to(x.device)

        # Pre-MP in tangent space, then project into the ball
        x = self.pre_mp(x)
        x = exp_map_zero(x, c)

        # Message-passing layers
        for conv in self.convs:
            x = conv(x, edge_index, c)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool: log to tangent -> mean -> exp back to ball
        x_tangent = log_map_zero(x, c)
        pooled = global_mean_pool(x_tangent, batch)
        pooled = exp_map_zero(pooled, c)

        # Post-MP: tangent MLP -> back to ball
        pooled_tangent = log_map_zero(pooled, c)
        out = self.post_mp(pooled_tangent)
        out = exp_map_zero(out, c)

        return out
