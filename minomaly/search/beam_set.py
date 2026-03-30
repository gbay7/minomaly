"""BeamSet — batch operations over a collection of Beams."""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Batch

from minomaly.data.convert import batch_pyg_data
from minomaly.scoring.base import ScoringFunction
from minomaly.search.beam import Beam


class BeamSet:
    """Collection of beams with GPU-accelerated batch operations.

    Unlike the original (which subclassed ``list``), this wraps a list
    for a cleaner API.
    """

    def __init__(self, beams: list[Beam] | None = None) -> None:
        self.beams: list[Beam] = beams if beams is not None else []

    def __len__(self) -> int:
        return len(self.beams)

    def __iter__(self):
        return iter(self.beams)

    def __getitem__(self, idx):
        return self.beams[idx]

    def __iadd__(self, other: BeamSet | list[Beam]) -> BeamSet:
        if isinstance(other, BeamSet):
            self.beams.extend(other.beams)
        else:
            self.beams.extend(other)
        return self

    # ── Batch embedding (GPU) ─────────────────────────────────────────

    def embed_all(
        self, emb_model_or_full, node_anchored: bool = True,
    ) -> BeamSet:
        """Batch-embed all beams in a single forward pass.

        If the model has ``embed_and_project``, uses it to embed AND
        project in one call (avoids recomputing projection at query time).
        """
        if not self.beams:
            return self
        # Detect if this is a full model with embed_and_project
        if hasattr(emb_model_or_full, "embed_and_project"):
            emb_fn = emb_model_or_full.embed_and_project
            model_device = next(emb_model_or_full.parameters()).device
        else:
            emb_fn = emb_model_or_full
            model_device = next(emb_model_or_full.parameters()).device
        data_list = [b.get_pyg_data() for b in self.beams]
        batch = batch_pyg_data(data_list, device=model_device)
        with torch.no_grad():
            embs = emb_fn(batch)
        for beam, emb in zip(self.beams, embs):
            beam.emb = emb.detach()
        return self

    # ── Batch scoring (GPU-accelerated) ───────────────────────────────

    def compute_all_scores(
        self,
        embs: list[torch.Tensor],
        model: torch.nn.Module,
        scorer: ScoringFunction,
        alpha: float = 0.5,
        unchange_direction: bool = False,
    ) -> BeamSet:
        """Compute strength scores for all beams via batched GPU ops.

        Instead of N sequential loops over embedding batches (original),
        this stacks all beam embeddings and computes pairwise violations
        in one tensor operation per embedding batch.
        """
        if not self.beams:
            return self

        n_beams = len(self.beams)
        beam_embs = torch.stack([b.emb for b in self.beams])  # (N, D)
        device = beam_embs.device

        freq_counts = torch.zeros(n_beams, device=device)
        total_n_embs = 0

        for emb_batch in embs:
            batch_size = len(emb_batch)
            total_n_embs += batch_size
            emb_batch = emb_batch.to(device)

            # Pairwise violations/distances: (N, B)
            # Delegates to model so Euclidean, Poincaré, etc. all work
            violations = model.batch_predict(emb_batch, beam_embs)

            # Classify each violation → subgraph prediction
            preds = model.clf_model(violations.unsqueeze(-1))  # (N, B, 2)
            supergraphs = torch.argmax(preds, dim=-1)  # (N, B)
            freq_counts += supergraphs.sum(dim=1).float()

        # Assign scores to beams
        for i, beam in enumerate(self.beams):
            beam.freq = (freq_counts[i].item() / total_n_embs) if total_n_embs > 0 else 0.0
            beam.score = scorer(beam.freq, beam.weight, alpha, beam.last_score)

            if unchange_direction:
                beam.unchange = beam.unchange + 1 if beam.score <= beam.last_score else 0
            else:
                beam.unchange = beam.unchange + 1 if beam.score >= beam.last_score else 0

        return self

    # ── Pruning ───────────────────────────────────────────────────────

    def prune(
        self, min_strength: float, max_strength: float, max_unchanged: int,
    ) -> BeamSet:
        self.beams = [
            b for b in self.beams
            if not b.is_prunable(min_strength, max_strength, max_unchanged)
        ]
        return self

    # ── Sorting / top-k ───────────────────────────────────────────────

    def sort_and_keep(
        self, top_k: int = 1, node_votes: Optional[dict[int, int]] = None,
    ) -> BeamSet:
        """Sort ascending by score, keep top_k (lowest = most anomalous).

        Ties are broken by node vote counts if provided.
        """
        if not self.beams:
            return self
        self.beams.sort(key=lambda b: b.score if b.score is not None else float("inf"))

        if node_votes is not None:
            min_score = self.beams[0].score
            tied = [b for b in self.beams if b.score == min_score]
            for b in tied:
                node_votes[b.node] = node_votes.get(b.node, 0) + 1
            tied.sort(key=lambda b: node_votes.get(b.node, 0), reverse=True)
            self.beams = tied[:top_k]
        else:
            self.beams = self.beams[:top_k]
        return self

    # ── Verification ──────────────────────────────────────────────────

    def extract_verified(
        self, min_strength: float, max_strength: float,
    ) -> BeamSet:
        """Extract and remove verified beams."""
        verified = [b for b in self.beams if b.is_verified(min_strength, max_strength)]
        self.beams = [b for b in self.beams if not b.is_verified(min_strength, max_strength)]
        return BeamSet(verified)

    def get_verified_neighbor_copies(
        self, min_strength: float, max_strength: float,
    ) -> BeamSet:
        """Create copies of verified beams' non-anchor nodes as new beams."""
        copies = []
        for beam in self.beams:
            if beam.is_verified(min_strength, max_strength):
                for node in beam.neigh:
                    if node != beam.anchor():
                        copies.append(beam.copy(node))
        return BeamSet(copies)
