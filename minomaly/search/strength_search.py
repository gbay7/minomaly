"""StrengthSearchAgent — beam search for anomalous subgraph patterns."""

from __future__ import annotations

import random
import time
from typing import Optional

from tqdm import tqdm

from minomaly.callbacks.base import Callback
from minomaly.callbacks.composite import CallbackList
from minomaly.data.graph import GraphData
from minomaly.registry import SEARCH
from minomaly.scoring.base import ScoringFunction
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.search.pattern_store import PatternStore


@SEARCH.register("strength")
class StrengthSearchAgent:
    """Beam search agent that grows patterns by frequency-based strength.

    At every step the agent:
      1. Generates candidate beams by expanding each beam's frontier.
      2. Batch-embeds all candidates (GPU).
      3. Batch-computes strength scores (GPU).
      4. Prunes hopeless beams.
      5. Keeps top-k (lowest score = most anomalous).
      6. Extracts verified beams in ``[min_strength, max_strength]``.
      7. Fires ``on_search_step`` callback.

    Changes from original:
      - Scoring function is injected (not hardcoded to freq).
      - GPU-batched frequency computation via :class:`BeamSet`.
      - No ThreadPoolExecutor (GIL provided no benefit).
      - Callbacks for observability.
    """

    def __init__(
        self,
        model,
        graphs: list[GraphData],
        embs: list,
        scorer: ScoringFunction,
        *,
        node_anchored: bool = True,
        add_self_loop: bool = True,
        n_beams: int = 1,
        min_strength: float = 0.0,
        max_strength: float = 0.01,
        alpha: float = 0.33,
        max_unchanged: int = 5,
        unchange_direction: bool = False,
        min_steps: int = 1,
        max_steps: int = 7,
        max_cands: Optional[int] = None,
        sample_random_cands: Optional[float] = None,
        add_verified_neighs: bool = False,
        min_neigh_repeat: int = 2,
        input_dim: int = 2,
        freq_cache: Optional[torch.Tensor] = None,
        min_subgraph_size: int = 1,
        **kwargs,
    ) -> None:
        self.input_dim = input_dim
        self.freq_cache = freq_cache  # (K,) per-neighborhood frequencies from Phase 2
        self.model = model
        self.graphs = graphs
        self.embs = embs
        self.scorer = scorer
        self.node_anchored = node_anchored
        self.add_self_loop = add_self_loop
        self.n_beams = n_beams
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.alpha = alpha
        self.max_unchanged = max_unchanged
        self.unchange_direction = unchange_direction
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.max_cands = max_cands
        self.sample_random_cands = sample_random_cands
        self.add_verified_neighs = add_verified_neighs
        self.min_neigh_repeat = min_neigh_repeat

        self.min_subgraph_size = min_subgraph_size

        self.num_nodes = sum(g.num_nodes for g in graphs)
        self.node_votes: dict[int, int] = {}

        # Populated during search
        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=node_anchored)

    @staticmethod
    def _grow_beam_to_size(beam: Beam, target_size: int) -> Optional[Beam]:
        """Grow a beam randomly to *target_size* without embedding/scoring."""
        while len(beam.neigh) < target_size:
            if not beam.frontier:
                return beam
            node = random.choice(beam.frontier)
            beam = Beam(
                node=node,
                graph=beam.graph,
                neigh=list(beam.neigh),
                frontier=list(beam.frontier),
                visited=set(beam.visited),
                total_weight=beam.graph.num_nodes,
                add_self_loop=beam._add_self_loop,
                node_anchored=beam._node_anchored,
                input_dim=beam._input_dim,
            )
        return beam

    def run(
        self,
        starting_nodes: list[int],
        graph_idx: int = 0,
        callbacks: Optional[list[Callback]] = None,
    ) -> BeamSet:
        """Execute the full search loop.

        Args:
            starting_nodes: initial nodes to start pattern growth from.
            graph_idx: index of the graph in ``self.graphs`` to search.
            callbacks: optional list of Callback objects.

        Returns:
            BeamSet of verified anomalous beams.
        """
        cb = CallbackList(callbacks or [])
        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=self.node_anchored)

        graph = self.graphs[graph_idx]

        # Initialize beam sets: one per starting node
        beam_sets: list[BeamSet] = []
        for node in starting_nodes:
            beam = Beam(
                node=int(node),
                graph=graph,
                total_weight=self.num_nodes,
                add_self_loop=self.add_self_loop,
                node_anchored=self.node_anchored,
                input_dim=self.input_dim,
            )
            if self.min_subgraph_size > 1:
                beam = self._grow_beam_to_size(beam, self.min_subgraph_size)
            if beam is not None:
                beam_sets.append(BeamSet([beam]))

        steps = max(1, self.min_subgraph_size - 1)
        while beam_sets and steps < self.max_steps:
            steps += 1
            beam_sets = self._step(beam_sets, steps, cb)

        cb.on_search_end(
            all_verified=set(self.verified),
            patterns=self.pattern_store.get_unique_patterns(),
            stats={"total_verified": len(self.verified),
                   "unique_patterns": self.pattern_store.unique_count},
        )
        return self.verified

    def _step(
        self,
        beam_sets: list[BeamSet],
        step: int,
        cb: CallbackList,
    ) -> list[BeamSet]:
        """Run one expansion step — batched across ALL beam sets.

        Instead of embedding/scoring each beam_set individually (100
        small GPU calls), this merges all candidates into one batch
        for a single embed + score pass, then splits back.
        """
        # 1. Generate ALL candidates across all beam sets
        all_cands: list[Beam] = []
        set_boundaries: list[int] = []  # track which cands belong to which set

        for beam_set in beam_sets:
            start = len(all_cands)
            for beam in beam_set:
                if not beam.frontier:
                    continue
                cands = beam.gen_candidates(
                    total_weight=self.num_nodes,
                    max_cands=self.max_cands,
                    sample_ratio=self.sample_random_cands,
                )
                all_cands.extend(cands)
            set_boundaries.append((start, len(all_cands)))

        if not all_cands:
            cb.on_search_step(step=step, beam_sets=[], verified=list(self.verified), new_verified_count=0)
            return []

        # 2-3. ONE batch embed + score for ALL candidates
        mega_beam_set = BeamSet(all_cands)
        mega_beam_set.embed_all(self.model, self.node_anchored)
        mega_beam_set.compute_all_scores(
            self.embs, self.model, self.scorer,
            self.alpha, self.unchange_direction,
            freq_cache=self.freq_cache,
        )

        # 4-6. Split back and process per beam_set
        new_beam_sets: list[BeamSet] = []
        new_verified_count = 0

        for start, end in set_boundaries:
            if start == end:
                continue
            new_beams = BeamSet(all_cands[start:end])

            new_beams.prune(self.min_strength, self.max_strength, self.max_unchanged)
            new_beams.sort_and_keep(self.n_beams, self.node_votes)

            verified = new_beams.extract_verified(self.min_strength, self.max_strength)
            if step < self.min_steps:
                verified = BeamSet()

            for beam in verified:
                self.pattern_store.add(beam)
            self.verified += verified
            new_verified_count += len(verified)

            if self.add_verified_neighs and verified:
                neighbor_copies = verified.get_verified_neighbor_copies(
                    self.min_strength, self.max_strength,
                )
                self.copied_verified += neighbor_copies

            for beam in new_beams:
                if beam.score is not None:
                    self.node_votes[beam.node] = self.node_votes.get(beam.node, 0) + 1

            if new_beams:
                new_beam_sets.append(new_beams)

        cb.on_search_step(
            step=step,
            beam_sets=new_beam_sets,
            verified=list(self.verified),
            new_verified_count=new_verified_count,
        )
        return new_beam_sets
