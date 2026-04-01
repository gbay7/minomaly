"""Contextual Scorer — conditional frequency computation.

Computes Freq_G(G' | c) = frequency within context cluster c.
Combines structural and contextual frequency via beta weighting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class ContextualScorer:
    """Score beams using context-conditional frequency.

    Partitions the structural reference embeddings by context label,
    then computes frequency within each partition.

    Parameters
    ----------
    structural_embs:
        (K, D) structural embeddings (same as used in search).
    context_labels:
        (K,) cluster labels from ContextClustering.
    n_clusters:
        Number of context clusters.
    beta:
        Combination weight: beta * structural + (1-beta) * contextual.
    threshold:
        Precomputed clf threshold for supergraph classification.
    """

    def __init__(
        self,
        structural_embs: torch.Tensor,
        context_labels: np.ndarray,
        n_clusters: int,
        beta: float = 0.5,
        threshold: float = 0.1,
    ) -> None:
        self.structural_embs = structural_embs
        self.context_labels = context_labels
        self.n_clusters = n_clusters
        self.beta = beta
        self.threshold = threshold
        self.device = structural_embs.device

        # Pre-partition: indices per cluster
        self.partitions: dict[int, torch.Tensor] = {}
        for c in range(n_clusters):
            mask = context_labels == c
            indices = torch.tensor(
                np.where(mask)[0], dtype=torch.long, device=self.device,
            )
            self.partitions[c] = indices

        # Report partition sizes
        for c, idx in self.partitions.items():
            print(f"  [context scorer] partition {c}: {len(idx)} refs")

    def conditional_frequency(
        self,
        beam_embs: torch.Tensor,
        context_ids: np.ndarray,
        model,
    ) -> torch.Tensor:
        """Compute Freq_G(G' | c) for each beam within its context.

        Parameters
        ----------
        beam_embs:
            (N, D) beam embeddings.
        context_ids:
            (N,) context cluster ID per beam.
        model:
            Embedding model with batch_predict().

        Returns
        -------
        Tensor:
            (N,) conditional frequencies.
        """
        N = beam_embs.shape[0]
        ctx_freqs = torch.zeros(N, device=self.device)

        # Group beams by context for efficient batch processing
        for c in range(self.n_clusters):
            mask = context_ids == c
            if not mask.any():
                continue

            beam_idx = np.where(mask)[0]
            ref_idx = self.partitions.get(c)
            if ref_idx is None or len(ref_idx) == 0:
                continue

            refs = self.structural_embs[ref_idx]  # (R_c, D)
            beams_c = beam_embs[beam_idx]          # (N_c, D)

            with torch.no_grad():
                violations = model.batch_predict(refs, beams_c)  # (N_c, R_c)
                is_super = violations < self.threshold
                counts = is_super.float().sum(dim=1)  # (N_c,)

            ctx_freqs[beam_idx] = counts / len(ref_idx)

        return ctx_freqs

    def combined_frequency(
        self,
        structural_freq: torch.Tensor,
        contextual_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Combine structural and contextual frequency.

        score = beta * structural + (1-beta) * contextual

        Parameters
        ----------
        structural_freq:
            (N,) global structural frequency.
        contextual_freq:
            (N,) context-conditional frequency.

        Returns
        -------
        Tensor:
            (N,) combined frequency.
        """
        return self.beta * structural_freq + (1 - self.beta) * contextual_freq
