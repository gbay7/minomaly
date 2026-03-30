"""Beam — a candidate subgraph pattern during search."""

from __future__ import annotations

from typing import Optional

import torch

from minomaly.data.graph import GraphData
from minomaly.scoring.base import ScoringFunction


class Beam:
    """A single candidate subgraph pattern grown during beam search.

    The subgraph is represented as a list of node ids in the parent
    :class:`GraphData`. The anchor is always ``neigh[0]``.

    Bug fixes vs. original:
      1. ``copy()`` creates independent score state (no shared refs).
      2. Self-loop handling is consistent via ``add_self_loop`` parameter.
      3. Scoring is delegated to an injected :class:`ScoringFunction`.
    """

    __slots__ = (
        "node", "graph", "neigh", "frontier", "visited", "emb",
        "score", "freq", "weight", "unchange", "last_score", "original",
        "_add_self_loop", "_node_anchored",
    )

    def __init__(
        self,
        node: int,
        graph: GraphData,
        neigh: Optional[list[int]] = None,
        frontier: Optional[list[int]] = None,
        visited: Optional[set[int]] = None,
        total_weight: Optional[int] = None,
        unchange: int = 0,
        last_score: float = float("inf"),
        add_self_loop: bool = True,
        node_anchored: bool = True,
    ) -> None:
        self.node = node
        self.graph = graph
        self._add_self_loop = add_self_loop
        self._node_anchored = node_anchored

        if neigh is not None:
            self.neigh = neigh + [node]
        else:
            self.neigh = [node]

        if frontier is not None and visited is not None:
            new_neighbors = set(graph.neighbors(node).tolist())
            self.frontier = list((set(frontier) | new_neighbors) - (visited | {node}))
            self.visited = visited | {node}
        elif frontier is None and visited is None:
            # Initial single-node beam: discover neighbors
            self.frontier = list(set(graph.neighbors(node).tolist()))
            self.visited = {node}
        else:
            self.frontier = frontier if frontier is not None else []
            self.visited = visited if visited is not None else {node}

        self.emb: Optional[torch.Tensor] = None
        self.score: Optional[float] = None
        self.freq: Optional[float] = None

        tw = total_weight if total_weight is not None else graph.num_nodes
        self.weight: float = len(self.neigh) / tw
        self.unchange = unchange
        self.last_score = last_score
        self.original = True

    def anchor(self) -> int:
        """The anchor node (first node in the neighborhood list)."""
        return self.neigh[0]

    # ── Candidate generation ──────────────────────────────────────────

    def gen_candidates(
        self,
        total_weight: Optional[int] = None,
        max_cands: Optional[int] = None,
        sample_ratio: Optional[float] = None,
    ) -> list[Beam]:
        """Expand the frontier: return one new Beam per candidate node."""
        import random

        frontier = list(self.frontier)
        if sample_ratio is not None:
            k = max(1, round(sample_ratio * len(frontier)))
            frontier = random.sample(frontier, min(k, len(frontier)))
        if max_cands is not None and len(frontier) > max_cands:
            frontier = random.sample(frontier, max_cands)

        return [
            Beam(
                node=cand,
                graph=self.graph,
                neigh=list(self.neigh),
                frontier=list(self.frontier),
                visited=set(self.visited),
                total_weight=total_weight,
                unchange=self.unchange,
                last_score=self.score if self.score is not None else self.last_score,
                add_self_loop=self._add_self_loop,
                node_anchored=self._node_anchored,
            )
            for cand in frontier
        ]

    # ── Embedding ─────────────────────────────────────────────────────

    def get_pyg_data(self) -> "torch_geometric.data.Data":
        """Build a PyG Data for this beam's subgraph."""
        nodes_t = torch.tensor(self.neigh, dtype=torch.long, device=self.graph.device)
        view = self.graph.subgraph(nodes_t)
        return view.to_pyg(
            anchor_global=self.anchor(),
            node_anchored=self._node_anchored,
            add_self_loop=self._add_self_loop,
        )

    def embed(self, emb_model: torch.nn.Module) -> torch.Tensor:
        """Embed this beam's subgraph. Caches the result."""
        if self.emb is not None:
            return self.emb
        from minomaly.data.convert import batch_pyg_data
        model_device = next(emb_model.parameters()).device
        data = self.get_pyg_data()
        batch = batch_pyg_data([data], device=model_device)
        self.emb = emb_model(batch).squeeze(0).detach()
        return self.emb

    # ── Scoring ───────────────────────────────────────────────────────

    def compute_strength(
        self,
        embs: list[torch.Tensor],
        model: torch.nn.Module,
        scorer: ScoringFunction,
        alpha: float = 0.5,
        unchange_direction: bool = False,
    ) -> float:
        """Compute frequency and score for this beam.

        Iterates through all embedding batches to count how many
        neighborhoods this beam is predicted to be a subgraph of.
        """
        freq_count, n_embs = 0, 0
        for emb_batch in embs:
            n_embs += len(emb_batch)
            # model.predict returns violation score
            violations = model.predict((emb_batch, self.emb))
            # model.clf_model classifies violation → subgraph probability
            preds = model.clf_model(violations.unsqueeze(1))
            supergraphs = torch.argmax(preds, dim=1)
            freq_count += supergraphs.sum().item()

        self.freq = freq_count / n_embs if n_embs > 0 else 0.0
        self.score = scorer(self.freq, self.weight, alpha, self.last_score)

        # Track convergence
        if unchange_direction:
            self.unchange = self.unchange + 1 if self.score <= self.last_score else 0
        else:
            self.unchange = self.unchange + 1 if self.score >= self.last_score else 0

        return self.score

    # ── Pruning & verification ────────────────────────────────────────

    def is_prunable(
        self, min_strength: float, max_strength: float, max_unchanged: int,
    ) -> bool:
        if self.score is None:
            return False
        if self.score > max_strength and self.unchange > max_unchanged:
            return True
        if self.score < min_strength:
            return True
        return False

    def is_verified(self, min_strength: float, max_strength: float) -> bool:
        if self.score is None:
            return False
        return min_strength <= self.score <= max_strength

    # ── Copy (for add_verified_neighs) ────────────────────────────────

    def copy(self, new_anchor: int) -> Beam:
        """Create a copy with *new_anchor* as the anchor.

        Bug fix: creates fresh score state instead of sharing the original's.
        """
        beam = Beam.__new__(Beam)
        beam.node = new_anchor
        beam.graph = self.graph
        beam.neigh = [new_anchor] + [n for n in self.neigh if n != new_anchor]
        beam.frontier = list(self.frontier)
        beam.visited = set(self.visited)
        beam.emb = None  # must be re-embedded
        beam.score = None
        beam.freq = None
        beam.weight = self.weight
        beam.unchange = 0
        beam.last_score = float("inf")
        beam.original = False
        beam._add_self_loop = self._add_self_loop
        beam._node_anchored = self._node_anchored
        return beam

    # ── Identity ──────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Beam):
            return False
        return self.anchor() == other.anchor()

    def __hash__(self) -> int:
        return hash(self.anchor())

    def __lt__(self, other: Beam) -> bool:
        return (self.score or 0) < (other.score or 0)

    def to_dict(self) -> dict:
        return {
            "anchor": self.anchor(),
            "added_node": self.node,
            "neigh": self.neigh,
            "neigh_len": len(self.neigh),
            "frontier_len": len(self.frontier),
            "weight": self.weight,
            "freq": self.freq,
            "score": self.score,
            "last_score": self.last_score,
            "unchange": self.unchange,
            "original": self.original,
        }

    def __repr__(self) -> str:
        return f"Beam(anchor={self.anchor()}, size={len(self.neigh)}, score={self.score})"
