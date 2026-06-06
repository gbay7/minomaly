"""Sampled Search — constant-time frequency estimation for dense graphs.

Never scores against ALL K references. Uses a fixed random sample (m)
at every step, including verification. Frequency estimate error is
O(1/√m), which is acceptable for anomaly detection.

For a graph with 10k refs and m=500: each step costs 500 comparisons
instead of 10000 — 20x faster. With 100 beams × 8 steps × 55 batches:
55 × 8 × 0.5s ≈ 4 minutes instead of 55 minutes.
"""

from __future__ import annotations

import random as _rnd
import time
from typing import Optional

import torch

from minomaly.callbacks.base import Callback
from minomaly.callbacks.composite import CallbackList
from minomaly.data.convert import batch_pyg_data
from minomaly.data.graph import GraphData
from minomaly.registry import SEARCH
from minomaly.scoring.base import ScoringFunction
from minomaly.search._freq import chunked_freq_score
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.search.pattern_store import PatternStore
from minomaly.search.strength_search import StrengthSearchAgent


from minomaly.search._batch import candidates_to_batch


def _precompute_threshold(clf_model, device) -> float:
    params = list(clf_model.parameters())
    w = params[0].detach().cpu()
    b = params[1].detach().cpu()
    denom = (w[1, 0] - w[0, 0]).item()
    if abs(denom) < 1e-10:
        return 0.1
    return (b[0] - b[1]).item() / denom


@SEARCH.register("sampled")
class SampledSearchAgent:
    """Constant-time search using sampled frequency estimation.

    Every step (including verification) uses a random sample of m
    reference embeddings. No full-K scoring ever. Fast for dense graphs.
    """

    def __init__(
        self,
        model, graphs, embs, scorer, *,
        node_anchored=True, add_self_loop=True,
        n_beams=1, min_strength=0.0, max_strength=0.01,
        alpha=0.33, max_unchanged=5, unchange_direction=False,
        min_steps=1, max_steps=7,
        max_cands=None, sample_random_cands=None,
        add_verified_neighs=False, min_neigh_repeat=2,
        input_dim=2, freq_cache=None,
        sample_size=500, min_subgraph_size=1, **kwargs,
    ):
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
        self.max_cands = int(max_cands) if max_cands is not None else None
        self.sample_random_cands = sample_random_cands
        self.add_verified_neighs = add_verified_neighs
        self.min_neigh_repeat = min_neigh_repeat
        self.input_dim = input_dim
        self.sample_size = sample_size
        self.min_subgraph_size = min_subgraph_size

        self.num_nodes = sum(g.num_nodes for g in graphs)
        self.all_embs = torch.cat(embs, dim=0)
        self.K = self.all_embs.shape[0]
        self.device = self.all_embs.device

        # Phase 2 frequency cache: (K,) per-ref cached frequencies
        self.freq_cache = freq_cache.to(self.device) if freq_cache is not None else None

        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=node_anchored)
        self.node_votes = {}

    def run(self, starting_nodes, graph_idx=0, callbacks=None):
        cb = CallbackList(callbacks or [])
        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=self.node_anchored)

        graph = self.graphs[graph_idx]

        # Init beams
        beam_sets = []
        for node in starting_nodes:
            beam = Beam(
                node=int(node), graph=graph, total_weight=self.num_nodes,
                add_self_loop=self.add_self_loop, node_anchored=self.node_anchored,
                input_dim=self.input_dim,
            )
            if self.min_subgraph_size > 1:
                beam = StrengthSearchAgent._grow_beam_to_size(beam, self.min_subgraph_size)
            if beam is not None:
                beam_sets.append(BeamSet([beam]))

        # Fixed random sample of reference embeddings
        m = min(self.sample_size, self.K)
        sample_idx = torch.randperm(self.K, device=self.device)[:m]
        ref_sample = self.all_embs[sample_idx]

        # Sampled freq cache: Phase 2 frequencies for the sampled refs
        sampled_fc = None
        if self.freq_cache is not None:
            sampled_fc = self.freq_cache[sample_idx]  # (m,)

        t0 = time.time()
        steps = max(1, self.min_subgraph_size - 1)
        while beam_sets and steps < self.max_steps:
            steps += 1

            # Generate ALL candidates
            all_cands = []
            set_boundaries = []
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
                break

            # ── Mega-batch embed (vectorized batch construction) ───────
            batch = candidates_to_batch(all_cands, self.device)
            with torch.no_grad():
                if hasattr(self.model, "embed_and_project"):
                    cand_embs = self.model.embed_and_project(batch)
                else:
                    cand_embs = self.model.emb_model(batch)
            for c, emb in zip(all_cands, cand_embs):
                c.emb = emb.detach()

            # ── Chunked score against sampled refs (memory-safe) ────────
            all_emb_stack = torch.stack([c.emb for c in all_cands])
            freq_counts = chunked_freq_score(
                self.model, ref_sample, all_emb_stack,
                sampled_fc=sampled_fc, max_strength=self.max_strength,
            )

            # Assign scores (normalize by m for comparable freq)
            freqs_this_step = []
            for i, c in enumerate(all_cands):
                c.freq = freq_counts[i].item() / m
                freqs_this_step.append(c.freq)
                c.freq_history.append((len(c.neigh), c.freq))
                c.score = self.scorer(c.freq, c.weight, self.alpha, c.last_score)
                if self.unchange_direction:
                    c.unchange = c.unchange + 1 if c.score <= c.last_score else 0
                else:
                    c.unchange = c.unchange + 1 if c.score >= c.last_score else 0

            # Diagnostic: log freq stats per step
            if freqs_this_step:
                import statistics
                _mn = min(freqs_this_step)
                _mx = max(freqs_this_step)
                _md = statistics.median(freqs_this_step)
                print(f"    step {steps}: {len(all_cands)} cands, "
                      f"freq min={_mn:.4f} med={_md:.4f} max={_mx:.4f}, "
                      f"refs={m}, "
                      f"max_strength={self.max_strength:.6f}")

            # ── Per-beam: prune, sort, verify ─────────────────────────
            new_beam_sets = []
            new_verified_count = 0
            for s, e in set_boundaries:
                if s == e:
                    continue
                new_beams = BeamSet(all_cands[s:e])
                new_beams.prune(self.min_strength, self.max_strength, self.max_unchanged)
                new_beams.sort_and_keep(self.n_beams, self.node_votes)

                if steps >= self.min_steps:
                    verified = new_beams.extract_verified(self.min_strength, self.max_strength)
                else:
                    verified = BeamSet()

                for beam in verified:
                    self.pattern_store.add(beam)
                self.verified += verified
                new_verified_count += len(verified)

                if self.add_verified_neighs and verified:
                    for beam in verified:
                        for n in beam.neigh:
                            if n != beam.anchor():
                                self.copied_verified.beams.append(beam.copy(n))

                for beam in new_beams:
                    if beam.score is not None:
                        self.node_votes[beam.node] = self.node_votes.get(beam.node, 0) + 1

                if new_beams:
                    new_beam_sets.append(new_beams)

            beam_sets = new_beam_sets
            cb.on_search_step(
                step=steps, beam_sets=beam_sets,
                verified=list(self.verified), new_verified_count=new_verified_count,
            )

        cb.on_search_end(
            all_verified=set(self.verified),
            patterns=self.pattern_store.get_unique_patterns(),
            stats={"total_verified": len(self.verified),
                   "unique_patterns": self.pattern_store.unique_count},
        )
        return self.verified
