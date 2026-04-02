"""Context Embedder Trainer — self-supervised contrastive learning.

Trains the context GNN to produce embeddings where neighborhoods with
similar structure + similar attribute patterns are close, and
dissimilar ones are far apart. No labels needed.

Uses graph-level contrastive learning:
1. For each neighborhood, create two augmented views (node/edge dropout)
2. Pull augmented views together (positive pairs)
3. Push different neighborhoods apart (negative pairs)
4. Loss: InfoNCE (NT-Xent)
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from minomaly.context.embedder import ContextEmbedder
from minomaly.data.convert import batch_pyg_data


def _augment_graph(data: Data, drop_node_rate: float = 0.1, drop_edge_rate: float = 0.2) -> Data:
    """Create an augmented view by randomly dropping nodes and edges."""
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes

    # Node feature masking (zero out random nodes' features)
    if drop_node_rate > 0 and num_nodes > 1:
        mask = torch.rand(num_nodes) > drop_node_rate
        mask[0] = True  # keep anchor
        x[~mask] = 0.0

    # Edge dropout
    if drop_edge_rate > 0 and edge_index.shape[1] > 0:
        n_edges = edge_index.shape[1]
        keep = torch.rand(n_edges) > drop_edge_rate
        if keep.any():
            edge_index = edge_index[:, keep]

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """InfoNCE (NT-Xent) contrastive loss.

    z1, z2: (B, D) — embeddings of two augmented views.
    Positive pairs: (z1[i], z2[i]). Negatives: all other pairs.
    """
    B = z1.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=z1.device)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Similarity matrix (2B x 2B)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim[mask].view(2 * B, -1)  # (2B, 2B-1)

    # Positive pairs: z1[i] ↔ z2[i]
    # In the 2B ordering: pos for i is at index B+i-1 (adjusted for mask)
    pos_sim = (z1 * z2).sum(dim=1) / temperature  # (B,)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  # (2B,)

    # Loss: -log(exp(pos) / sum(exp(all)))
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()


class ContextTrainer:
    """Self-supervised contrastive trainer for the context embedder.

    Creates augmented views of sampled neighborhoods and trains
    the encoder to produce similar embeddings for augmentations
    of the same neighborhood.

    Parameters
    ----------
    embedder:
        The ContextEmbedder to train.
    lr:
        Learning rate.
    temperature:
        InfoNCE temperature.
    drop_node_rate:
        Probability of masking each node's features.
    drop_edge_rate:
        Probability of dropping each edge.
    n_epochs:
        Number of training epochs over the neighborhoods.
    batch_size:
        Training batch size.
    """

    def __init__(
        self,
        embedder: ContextEmbedder,
        lr: float = 1e-3,
        temperature: float = 0.5,
        drop_node_rate: float = 0.1,
        drop_edge_rate: float = 0.2,
        n_epochs: int = 50,
        batch_size: int = 128,
    ) -> None:
        self.embedder = embedder
        self.lr = lr
        self.temperature = temperature
        self.drop_node_rate = drop_node_rate
        self.drop_edge_rate = drop_edge_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def train(
        self,
        context_data: list[Data],
        device: torch.device,
    ) -> list[float]:
        """Train the context embedder contrastively.

        Parameters
        ----------
        context_data:
            List of PyG Data objects with original features.
        device:
            Training device.

        Returns
        -------
        list[float]:
            Loss per epoch.
        """
        self.embedder.train()
        self.embedder.to(device)
        optimizer = torch.optim.Adam(self.embedder.parameters(), lr=self.lr)

        losses = []
        n = len(context_data)
        indices = list(range(n))

        for epoch in range(self.n_epochs):
            random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                if len(batch_idx) < 2:
                    continue

                # Create two augmented views
                view1_list = [_augment_graph(context_data[j], self.drop_node_rate, self.drop_edge_rate)
                              for j in batch_idx]
                view2_list = [_augment_graph(context_data[j], self.drop_node_rate, self.drop_edge_rate)
                              for j in batch_idx]

                batch1 = batch_pyg_data(view1_list, device=device)
                batch2 = batch_pyg_data(view2_list, device=device)

                z1 = self.embedder.encoder(batch1)  # (B, D)
                z2 = self.embedder.encoder(batch2)  # (B, D)

                loss = _info_nce_loss(z1, z2, self.temperature)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.embedder.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  [context train] epoch {epoch+1}/{self.n_epochs}, loss={avg_loss:.4f}")

        self.embedder.eval()
        return losses
