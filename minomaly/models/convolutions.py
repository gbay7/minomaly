"""Custom graph convolution layers.

Ports of the ``SAGEConv`` and ``GINConv`` classes from
``code-original/common/models.py``, cleaned up with type hints and without
any behavioural changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils


class SAGEConv(pyg_nn.MessagePassing):
    """Custom GraphSAGE convolution.

    Aggregates neighbour embeddings via a ``Linear`` transform, concatenates
    the aggregation with the centre-node embedding, and projects through a
    second ``Linear`` layer.

    This matches the behaviour of the original ``SAGEConv`` in
    ``code-original/common/models.py``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        size: Optional[tuple[int, int]] = None,
        res_n_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one message-passing step.

        Parameters
        ----------
        x:
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index:
            Edge connectivity in COO format ``(2, num_edges)``.
        edge_weight:
            Optional per-edge scalar weights.
        size:
            Optional size hint for bipartite message passing.
        res_n_id:
            Optional residual node indices (unused in practice but kept
            for API compatibility with the original code).
        """
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        return self.propagate(
            edge_index,
            size=size,
            x=x,
            edge_weight=edge_weight,
            res_n_id=res_n_id,
        )

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.lin(x_j)

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        res_n_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        aggr_out = torch.cat([aggr_out, x], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


class GINConv(pyg_nn.MessagePassing):
    """Custom GIN convolution with optional weighted-edge support.

    This matches the behaviour of the original ``GINConv`` in
    ``code-original/common/models.py``.
    """

    def __init__(
        self,
        nn_module: nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:
        super().__init__(aggr="add")
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one GIN message-passing step."""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(
            edge_index, edge_weight
        )
        out = self.nn(
            (1 + self.eps) * x
            + self.propagate(edge_index, x=x, edge_weight=edge_weight)
        )
        return out

    def message(
        self,
        x_j: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"
