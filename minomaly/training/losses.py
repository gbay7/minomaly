"""Order embedding loss function."""

import torch


def order_embedding_loss(
    emb_as: torch.Tensor,
    emb_bs: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Compute the order embedding hinge loss.

    For positive pairs (label=1, b IS subgraph of a):
        minimise  e = Σ max(0, emb_b - emb_a)²

    For negative pairs (label=0):
        push  e ≥ margin  →  loss = max(0, margin − e)

    Args:
        emb_as: (B, D) embeddings of "target" graphs (potential supergraphs).
        emb_bs: (B, D) embeddings of "query" graphs (potential subgraphs).
        labels: (B,) 1 for positive, 0 for negative.
        margin: margin for negative pairs.

    Returns:
        Scalar loss.
    """
    e = torch.sum(
        torch.clamp(emb_bs - emb_as, min=0) ** 2, dim=1
    )
    neg_mask = labels == 0
    e[neg_mask] = torch.clamp(margin - e[neg_mask], min=0)
    return e.sum()
