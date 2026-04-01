"""Adaptive Search — cheap exploration, precise verification.

Combines the best of strength and sampled search:
- Steps < min_steps (exploration): score against m sampled refs (fast)
- Steps >= min_steps (verification): score against ALL K refs (exact)

Additional optimizations:
- Precomputed clf threshold (replaces clf_model forward pass)
- Insight 1: freq_cache lower bounds for early non-anomalous detection
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
from minomaly.search.beam import Beam
from minomaly.search.beam_set import BeamSet
from minomaly.search.pattern_store import PatternStore


def _precompute_threshold(clf_model, device) -> float:
    """Extract decision boundary from Linear(1,2)+LogSoftmax classifier."""
    params = list(clf_model.parameters())
    w = params[0].detach().cpu()  # (2, 1)
    b = params[1].detach().cpu()  # (2,)
    denom = (w[1, 0] - w[0, 0]).item()
    if abs(denom) < 1e-10:
        return 0.1
    return (b[0] - b[1]).item() / denom


@SEARCH.register("adaptive")
class AdaptiveSearchAgent:
    """Adaptive search: cheap exploration + precise verification.

    During exploration (steps < min_steps): scores against a fixed
    random sample of m reference embeddings. Fast, sufficient for
    candidate ranking when freq >> threshold.

    During verification (steps >= min_steps): scores against ALL K
    reference embeddings. Exact frequency for verification decisions.

    Uses precomputed clf threshold instead of clf_model forward pass.
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
        sample_size=500, **kwargs,
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

        self.num_nodes = sum(g.num_nodes for g in graphs)
        self.all_embs = torch.cat(embs, dim=0)  # (K, D)
        self.K = self.all_embs.shape[0]
        self.device = self.all_embs.device

        # Precomputed clf threshold: violations < τ ↔ clf_model predicts supergraph
        self.threshold = _precompute_threshold(model.clf_model, self.device)

        # Phase 2 freq cache
        self.freq_cache = freq_cache.to(self.device) if freq_cache is not None else None

        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=node_anchored)
        self.node_votes = {}
        self.sample_size = sample_size

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
            beam_sets.append(BeamSet([beam]))

        # Fixed random sample for exploration steps
        m = min(self.sample_size, self.K)
        sample_idx = torch.randperm(self.K, device=self.device)[:m]
        ref_sample = self.all_embs[sample_idx]  # (m, D)
        sampled_fc = self.freq_cache[sample_idx] if self.freq_cache is not None else None

        t0 = time.time()
        steps = 1
        while beam_sets and steps < self.max_steps:
            steps += 1

            # ── 1. Generate ALL candidates ────────────────────────
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

            # ── 2. Mega-batch embed (one GPU call) ────────────────
            data_list = [c.get_pyg_data() for c in all_cands]
            batch = batch_pyg_data(data_list, device=self.device)
            with torch.no_grad():
                if hasattr(self.model, "embed_and_project"):
                    cand_embs = self.model.embed_and_project(batch)
                else:
                    cand_embs = self.model.emb_model(batch)
            for c, emb in zip(all_cands, cand_embs):
                c.emb = emb.detach()

            # ── 3. Adaptive scoring ───────────────────────────────
            is_verification = steps >= self.min_steps

            if is_verification:
                # VERIFICATION: score against ALL K refs (exact)
                refs = self.all_embs  # (K, D)
                fc = self.freq_cache  # (K,) or None
            else:
                # EXPLORATION: score against m sampled refs (fast)
                refs = ref_sample     # (m, D)
                fc = sampled_fc       # (m,) or None

            R = refs.shape[0]
            all_emb_stack = torch.stack([c.emb for c in all_cands])
            N = all_emb_stack.shape[0]

            # Score in chunks to avoid OOM when R is large
            freq_counts = torch.zeros(N, device=self.device)
            lower_bounds = torch.zeros(N, device=self.device) if fc is not None else None
            chunk_size = 1000  # process refs in chunks

            with torch.no_grad():
                for r_start in range(0, R, chunk_size):
                    r_end = min(r_start + chunk_size, R)
                    ref_chunk = refs[r_start:r_end]
                    violations = self.model.batch_predict(ref_chunk, all_emb_stack)  # (N, chunk)
                    is_super = violations < self.threshold
                    freq_counts += is_super.float().sum(dim=1)

                    # Insight 1: update lower bounds from this chunk
                    if fc is not None:
                        fc_chunk = fc[r_start:r_end]
                        masked_fc = is_super.float() * fc_chunk.unsqueeze(0)  # (N, chunk)
                        chunk_bounds = masked_fc.max(dim=1).values
                        lower_bounds = torch.max(lower_bounds, chunk_bounds)

            # Insight 1: mark non-anomalous candidates
            if lower_bounds is not None:
                non_anom = lower_bounds > self.max_strength
                for i in non_anom.nonzero(as_tuple=True)[0].tolist():
                    freq_counts[i] = lower_bounds[i].item() * R

            # Assign scores
            freqs_this_step = []
            for i, c in enumerate(all_cands):
                c.freq = freq_counts[i].item() / R
                freqs_this_step.append(c.freq)
                c.freq_history.append((len(c.neigh), c.freq))
                c.score = self.scorer(c.freq, c.weight, self.alpha, c.last_score)
                if self.unchange_direction:
                    c.unchange = c.unchange + 1 if c.score <= c.last_score else 0
                else:
                    c.unchange = c.unchange + 1 if c.score >= c.last_score else 0

            # Diagnostic
            if freqs_this_step:
                import statistics
                _mn = min(freqs_this_step)
                _mx = max(freqs_this_step)
                _md = statistics.median(freqs_this_step)
                mode = "VERIF" if is_verification else "EXPL"
                print(f"    step {steps} [{mode}]: {len(all_cands)} cands, "
                      f"freq min={_mn:.4f} med={_md:.4f} max={_mx:.4f}, "
                      f"refs={R}/{self.K}, "
                      f"max_strength={self.max_strength:.6f}")

            # ── 4. Per-beam: prune, sort, verify ──────────────────
            new_beam_sets = []
            new_verified_count = 0
            for s, e in set_boundaries:
                if s == e:
                    continue
                new_beams = BeamSet(all_cands[s:e])
                new_beams.prune(self.min_strength, self.max_strength, self.max_unchanged)
                new_beams.sort_and_keep(self.n_beams, self.node_votes)

                if is_verification:
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
