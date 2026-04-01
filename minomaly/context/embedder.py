"""Context Embedder — attribute-aware neighborhood embedding.

Re-embeds sampled neighborhoods using original node features (not GLASS
labels) through an attribute-aware GNN. Produces graph-level embeddings
where neighborhoods with similar structure + similar attribute patterns
are close in embedding space.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm

from minomaly.data.convert import batch_pyg_data
from minomaly.registry import ENCODERS


class ContextEmbedder(nn.Module):
    """Embed neighborhoods using original node attributes.

    Takes sampled neighborhoods (with GLASS features) and re-creates
    them with the original node features from the dataset. Then encodes
    through a GNN to produce context embeddings for clustering.

    Parameters
    ----------
    input_dim:
        Original feature dimension (e.g., 1433 for Cora).
    hidden_dim:
        Embedding dimension.
    encoder_name:
        Registered encoder to use (e.g., "skip_last_gnn").
    **encoder_kwargs:
        Forwarded to the encoder constructor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        encoder_name: str = "skip_last_gnn",
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = ENCODERS.build(
            encoder_name,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            **encoder_kwargs,
        )
        self.hidden_dim = hidden_dim

    def _rebuild_with_features(
        self,
        neighborhood: Data,
        node_list: list[int],
        original_features: torch.Tensor,
    ) -> Data:
        """Replace GLASS labels with original node features.

        Parameters
        ----------
        neighborhood:
            PyG Data with GLASS features (2D) and relabeled edges.
        node_list:
            Original node IDs [anchor, n1, n2, ...] in the sampled neighborhood.
            Index 0 = anchor (mapped to local node 0).
        original_features:
            Full feature matrix (N, F) from the dataset.

        Returns
        -------
        Data:
            Same structure but with original features as x.
        """
        anchor = node_list[0]
        others = set(node_list) - {anchor}
        # Reconstruct the same mapping used by the sampler:
        # anchor → 0, rest → 1..n-1 (sorted by the set iteration order from sampling)
        mapping = {anchor: 0}
        mapping.update({n: i + 1 for i, n in enumerate(sorted(others))})

        num_nodes = len(mapping)
        # Build feature matrix with original features
        x = torch.zeros(num_nodes, original_features.shape[1])
        for orig_id, local_id in mapping.items():
            x[local_id] = original_features[orig_id]

        return Data(
            x=x,
            edge_index=neighborhood.edge_index.clone(),
            num_nodes=num_nodes,
        )

    def embed_neighborhoods(
        self,
        neighborhoods: list[Data],
        node_lists: list[list[int]],
        original_features: torch.Tensor,
        batch_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Embed all neighborhoods with original features.

        Parameters
        ----------
        neighborhoods:
            Sampled neighborhoods (PyG Data with GLASS features).
        node_lists:
            Per-neighborhood original node ID lists.
        original_features:
            Dataset feature matrix (N, F).
        batch_size:
            Embedding batch size.
        device:
            Target device.

        Returns
        -------
        Tensor:
            (K, hidden_dim) context embeddings.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_embs = []
        n = len(neighborhoods)

        for i in tqdm(range(0, n, batch_size), desc="Context embedding"):
            top = min(n, i + batch_size)
            batch_data = []
            for j in range(i, top):
                data = self._rebuild_with_features(
                    neighborhoods[j], node_lists[j], original_features,
                )
                batch_data.append(data)

            batch = batch_pyg_data(batch_data, device=device)
            with torch.no_grad():
                emb = self.encoder(batch)  # (B, hidden_dim)
            all_embs.append(emb)

        return torch.cat(all_embs, dim=0)  # (K, hidden_dim)
