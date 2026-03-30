"""GATv2 graph encoder with skip connections.

Uses PyG's ``GATv2Conv`` (dynamic attention) as the message-passing layer.
Skip connections concatenate all intermediate layer outputs (like SkipLastGNN)
before a final MLP produces the graph-level embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Batch, Data

from minomaly.models.base import GraphEncoder
from minomaly.registry import ENCODERS


@ENCODERS.register("gatv2")
class GATv2Encoder(GraphEncoder):
    """GATv2-based graph encoder with skip connections.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (1 for anchor indicator).
    hidden_dim : int
        Hidden dimension per layer.
    output_dim : int
        Final graph-level embedding dimension.
    n_layers : int
        Number of GATv2Conv layers.
    n_heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout rate applied after each layer.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # Project input features to hidden_dim
        self.pre_mp = nn.Linear(input_dim, hidden_dim)

        # GATv2 layers — each outputs hidden_dim (heads * out_channels)
        head_dim = hidden_dim // n_heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                pyg_nn.GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=n_heads,
                    concat=True,       # output = n_heads * head_dim = hidden_dim
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Post-MLP: concat all layer outputs → project to output_dim
        # Input: hidden_dim * (n_layers + 1) — initial + each layer
        concat_dim = hidden_dim * (n_layers + 1)
        self.post_mp = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: Data | Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pre-message-passing
        x = self.pre_mp(x)
        all_embs = [x]

        # GATv2 layers with skip connections
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual connection
            all_embs.append(x)

        # Concatenate all layer outputs
        emb = torch.cat(all_embs, dim=-1)  # (num_nodes, hidden_dim * (n_layers+1))

        # Pool to graph-level
        emb = pyg_nn.global_add_pool(emb, batch)

        # Post-MLP
        emb = self.post_mp(emb)
        return emb
