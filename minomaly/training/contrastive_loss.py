"""InfoNCE contrastive loss for subgraph containment learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss for subgraph containment.

    For positive pairs (label=1), pull embeddings close.
    Each positive uses all negatives in the batch as contrastive examples.

    Parameters
    ----------
    emb_a : Tensor
        Shape ``(N, D)`` — target graph embeddings.
    emb_b : Tensor
        Shape ``(N, D)`` — query graph embeddings.
    labels : Tensor
        Shape ``(N,)`` — 1 for positive (true subgraph), 0 for negative.
    temperature : float
        Temperature for the softmax.  Smaller → sharper distribution.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    # Normalise to unit sphere for cosine similarity
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)

    # Similarity matrix: (N, N) — each query vs every target
    sim_matrix = torch.mm(emb_b, emb_a.t()) / temperature  # (N, N)

    # Positive pairs are on the diagonal where labels == 1
    pos_mask = labels == 1
    n_pos = pos_mask.sum().item()

    if n_pos == 0:
        return torch.tensor(0.0, device=emb_a.device, requires_grad=True)

    # For each positive query i, the positive target is i (diagonal)
    # and all other targets are negatives.
    # Standard InfoNCE: loss_i = -log(exp(sim_ii / τ) / Σ_j exp(sim_ij / τ))
    # = -sim_ii / τ + log(Σ_j exp(sim_ij / τ))

    # Only compute loss for positive pairs
    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    pos_sims = sim_matrix[pos_indices]  # (n_pos, N)

    # Targets: diagonal elements for positive pairs
    targets = torch.arange(len(pos_indices), device=emb_a.device)

    # Map to indices in the full sim matrix
    # pos_sims[k, pos_indices[k]] should be the positive
    # Use cross-entropy: target for row k is column pos_indices[k]
    loss = F.cross_entropy(pos_sims, pos_indices)

    return loss
