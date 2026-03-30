"""SkipLastGNN graph encoder.

Port of ``SkipLastGNN`` from ``code-original/common/models.py`` with the
following changes:

* Uses ``data.x`` instead of ``data.node_feature`` (standard PyG convention).
* Registered in the :data:`ENCODERS` registry under ``"skip_last_gnn"``.
* Imports custom convolutions from :mod:`minomaly.models.convolutions`.
* No dependency on DeepSNAP or ``feature_preprocess``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from minomaly.models.base import GraphEncoder
from minomaly.models.convolutions import GINConv, SAGEConv
from minomaly.registry import ENCODERS


@ENCODERS.register("skip_last_gnn")
class SkipLastGNN(GraphEncoder):
    """Graph-level encoder with learnable skip connections.

    Architecture:

    1. ``pre_mp``: Linear projection of input features.
    2. ``n_layers`` convolution layers with skip connections (concatenation
       of all previous layer outputs, gated by learnable sigmoid weights).
    3. ``global_add_pool`` over all node embeddings.
    4. ``post_mp``: MLP producing the final graph-level embedding.

    Supports convolution types: ``"SAGE"``, ``"GIN"``, ``"GCN"``, ``"GAT"``,
    ``"graph"``, ``"gated"``, ``"PNA"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 8,
        conv_type: str = "SAGE",
        skip: str = "learnable",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.skip = skip
        self.conv_type = conv_type

        pre_mp_out = 3 * hidden_dim if conv_type == "PNA" else hidden_dim
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, pre_mp_out))

        conv_model = self.build_conv_model(conv_type)

        if conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()

        if skip == "learnable":
            self.learnable_skip = nn.Parameter(
                torch.ones(self.n_layers, self.n_layers)
            )

        for layer_idx in range(n_layers):
            if skip in ("all", "learnable"):
                hidden_input_dim = hidden_dim * (layer_idx + 1)
            else:
                hidden_input_dim = hidden_dim

            if conv_type == "PNA":
                self.convs_sum.append(conv_model(3 * hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(3 * hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(3 * hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (n_layers + 1)
        if conv_type == "PNA":
            post_input_dim *= 3

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

    @staticmethod
    def build_conv_model(model_type: str) -> type:
        """Return a convolution constructor for the given *model_type*."""
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            return lambda i, h: GINConv(
                nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
            )
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, 1)
        elif model_type == "PNA":
            return SAGEConv
        else:
            raise ValueError(f"Unrecognised conv_type: {model_type!r}")

    def forward(self, data) -> torch.Tensor:
        """Encode a (batched) graph and return graph-level embeddings.

        Parameters
        ----------
        data:
            A PyG ``Data`` or ``Batch`` object with ``x``, ``edge_index``,
            and ``batch`` attributes.

        Returns
        -------
        torch.Tensor
            Graph-level embedding tensor of shape ``(batch_size, hidden_dim)``.
        """
        x: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        batch: torch.Tensor = data.batch

        x = self.pre_mp(x)

        all_emb = x.unsqueeze(1)
        emb = x

        num_conv = (
            len(self.convs_sum) if self.conv_type == "PNA" else len(self.convs)
        )

        for i in range(num_conv):
            if self.skip == "learnable":
                skip_vals = self.learnable_skip[i, : i + 1].unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)
                if self.conv_type == "PNA":
                    x = torch.cat(
                        (
                            self.convs_sum[i](curr_emb, edge_index),
                            self.convs_mean[i](curr_emb, edge_index),
                            self.convs_max[i](curr_emb, edge_index),
                        ),
                        dim=-1,
                    )
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == "all":
                if self.conv_type == "PNA":
                    x = torch.cat(
                        (
                            self.convs_sum[i](emb, edge_index),
                            self.convs_mean[i](emb, edge_index),
                            self.convs_max[i](emb, edge_index),
                        ),
                        dim=-1,
                    )
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            if self.skip == "learnable":
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb
