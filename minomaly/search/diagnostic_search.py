"""Diagnostic Search Agent — detailed tracking for debugging.

Wraps the incremental search with per-node, per-step logging to understand
exactly why precision differs from the original code.

Saves per-batch JSON with:
- Every verified beam's anchor, neigh, score, freq
- Every copied beam's anchor, source anchor, score
- Starting nodes that were true anomalies vs false
- Per-step beam count and supergraph set sizes
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
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


@SEARCH.register("diagnostic")
class DiagnosticSearchAgent:
    """Incremental search with full diagnostic output."""

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
        output_dir="plots/diagnostic",
        anomalous_nodes=None,
    ) -> None:
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
        self.all_embs = torch.cat(embs, dim=0)
        self.K = self.all_embs.shape[0]
        self.device = self.all_embs.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.anomalous_nodes = set(anomalous_nodes or [])

        # Precompute clf threshold for diagnostics
        params = list(model.clf_model.parameters())
        w = params[0].detach().cpu()
        b = params[1].detach().cpu()
        denom = (w[1, 0] - w[0, 0]).item()
        self.threshold = (b[0] - b[1]).item() / denom if abs(denom) > 1e-10 else 0.1
        print(f"[diag] clf threshold: {self.threshold:.6f}", flush=True)

        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=node_anchored)

    def run(self, starting_nodes, graph_idx=0, callbacks=None):
        cb = CallbackList(callbacks or [])
        self.verified = BeamSet()
        self.copied_verified = BeamSet()
        self.pattern_store = PatternStore(node_anchored=self.node_anchored)

        graph = self.graphs[graph_idx]
        diag_data = {
            "starting_nodes": starting_nodes,
            "n_true_anomalies_in_starting": len(set(starting_nodes) & self.anomalous_nodes),
            "beams": [],
        }

        # Per-node search with full tracking
        for node_idx, node in enumerate(starting_nodes):
            beam = Beam(
                node=int(node), graph=graph, total_weight=self.num_nodes,
                add_self_loop=self.add_self_loop, node_anchored=self.node_anchored,
                input_dim=self.input_dim,
            )
            super_mask = torch.ones(self.K, dtype=torch.bool, device=self.device)
            is_true_anomaly = node in self.anomalous_nodes

            beam_log = {
                "anchor": int(node),
                "is_true_anomaly": is_true_anomaly,
                "steps": [],
                "verified": False,
                "final_score": None,
                "final_freq": None,
                "final_neigh": None,
            }

            for step in range(2, self.max_steps + 2):
                if not beam.frontier:
                    break

                cands = beam.gen_candidates(
                    total_weight=self.num_nodes,
                    max_cands=self.max_cands,
                    sample_ratio=self.sample_random_cands,
                )
                if not cands:
                    break

                # Embed
                data_list = [c.get_pyg_data() for c in cands]
                batch = batch_pyg_data(data_list, device=self.device)
                with torch.no_grad():
                    if hasattr(self.model, "embed_and_project"):
                        cand_embs = self.model.embed_and_project(batch)
                    else:
                        cand_embs = self.model.emb_model(batch)
                for c, emb in zip(cands, cand_embs):
                    c.emb = emb.detach()

                # Score against supergraph set
                n_super = super_mask.sum().item()
                if n_super == 0:
                    break

                ref_embs = self.all_embs[super_mask]
                cand_stack = torch.stack([c.emb for c in cands])

                with torch.no_grad():
                    violations = self.model.batch_predict(ref_embs, cand_stack)
                    is_super = (violations < self.threshold)
                    freq_counts = is_super.float().sum(dim=1)

                for ci, c in enumerate(cands):
                    c.freq = freq_counts[ci].item() / self.K
                    c.score = self.scorer(c.freq, c.weight, self.alpha, c.last_score)
                    if self.unchange_direction:
                        c.unchange = c.unchange + 1 if c.score <= c.last_score else 0
                    else:
                        c.unchange = c.unchange + 1 if c.score >= c.last_score else 0

                cands.sort(key=lambda b: b.score if b.score is not None else float("inf"))
                best = cands[0]

                step_log = {
                    "step": step,
                    "n_candidates": len(cands),
                    "n_supergraphs": int(n_super),
                    "best_freq": best.freq,
                    "best_score": best.score,
                    "best_node_added": int(best.node),
                    "best_neigh_size": len(best.neigh),
                    "pruned": best.is_prunable(self.min_strength, self.max_strength, self.max_unchanged),
                    "verified": step >= self.min_steps and best.is_verified(self.min_strength, self.max_strength),
                }
                beam_log["steps"].append(step_log)

                if best.is_prunable(self.min_strength, self.max_strength, self.max_unchanged):
                    break

                # Update supergraph set
                best_idx = cands.index(best)
                ref_indices = super_mask.nonzero(as_tuple=True)[0]
                new_mask = torch.zeros_like(super_mask)
                new_mask[ref_indices[is_super[best_idx]]] = True
                super_mask = new_mask
                beam = best

                if step >= self.min_steps and best.is_verified(self.min_strength, self.max_strength):
                    beam_log["verified"] = True
                    beam_log["final_score"] = best.score
                    beam_log["final_freq"] = best.freq
                    beam_log["final_neigh"] = [int(n) for n in best.neigh]
                    beam_log["final_neigh_true_anomalies"] = len(set(best.neigh) & self.anomalous_nodes)
                    beam_log["final_neigh_normal"] = len(best.neigh) - beam_log["final_neigh_true_anomalies"]

                    self.pattern_store.add(best)
                    self.verified.beams.append(best)
                    if self.add_verified_neighs:
                        for n in best.neigh:
                            if n != best.anchor():
                                self.copied_verified.beams.append(best.copy(n))
                    break

            diag_data["beams"].append(beam_log)

            if (node_idx + 1) % 50 == 0:
                n_ver = len(self.verified)
                print(f"  [{node_idx+1}/{len(starting_nodes)}] {n_ver} verified", flush=True)

        # Save diagnostics
        diag_data["total_verified"] = len(self.verified)
        diag_data["total_copied"] = len(self.copied_verified)

        # Analyze
        verified_anchors = [b.anchor() for b in self.verified]
        tp_anchors = [a for a in verified_anchors if a in self.anomalous_nodes]
        fp_anchors = [a for a in verified_anchors if a not in self.anomalous_nodes]

        diag_data["analysis"] = {
            "tp_anchors": tp_anchors,
            "fp_anchors": fp_anchors,
            "tp_count": len(tp_anchors),
            "fp_count": len(fp_anchors),
            "precision_before_copies": len(tp_anchors) / max(len(verified_anchors), 1),
        }

        # Analyze false positives: why were they verified?
        fp_details = []
        for bm in self.verified:
            if bm.anchor() not in self.anomalous_nodes:
                neigh_set = set(bm.neigh)
                n_anom_in_neigh = len(neigh_set & self.anomalous_nodes)
                fp_details.append({
                    "anchor": int(bm.anchor()),
                    "neigh": [int(n) for n in bm.neigh],
                    "freq": bm.freq,
                    "score": bm.score,
                    "anomalous_in_neigh": n_anom_in_neigh,
                    "near_anomaly": n_anom_in_neigh > 0,
                })
        diag_data["fp_details"] = fp_details

        out_path = self.output_dir / "diagnostic.json"
        with open(out_path, "w") as f:
            json.dump(diag_data, f, indent=2, default=str)
        print(f"[diag] Saved to {out_path}", flush=True)
        print(f"[diag] TP={len(tp_anchors)}, FP={len(fp_anchors)}, "
              f"Precision(before copies)={diag_data['analysis']['precision_before_copies']:.1%}", flush=True)
        print(f"[diag] FP near anomaly: {sum(1 for d in fp_details if d['near_anomaly'])}/{len(fp_details)}", flush=True)

        cb.on_search_end(
            all_verified=set(self.verified),
            patterns=self.pattern_store.get_unique_patterns(),
            stats={"total_verified": len(self.verified), "unique_patterns": self.pattern_store.unique_count},
        )
        return self.verified
