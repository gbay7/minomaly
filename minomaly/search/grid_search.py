"""Grid search for optimal max_freq and max_steps hyperparameters.

Runs the detection pipeline with different (max_freq, max_steps) combinations
and reports metrics for each. Results are saved as a JSON file and printed
as a table.

Usage:
    from minomaly.search.grid_search import run_grid_search
    results = run_grid_search(model, graphs, embs, anomalous_nodes, all_nodes, ...)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from minomaly.evaluation.metrics import get_stat_results
from minomaly.registry import SCORING, SEARCH
from minomaly.data.graph import GraphData


def run_grid_search(
    model,
    graphs: list[GraphData],
    embs: list[torch.Tensor],
    anomalous_nodes: list[int],
    all_nodes: list[int],
    starting_nodes: set[int],
    *,
    # Grid ranges
    max_freq_range: list[float],
    max_steps_range: list[int],
    # Fixed search params
    search_strategy: str = "fast",
    min_strength: float = 0.0,
    max_unchanged: int = 5,
    n_beams: int = 1,
    alpha: float = 0.33,
    max_cands: Optional[int] = None,
    add_verified_neighs: bool = True,
    min_neigh_repeat: int = 2,
    node_anchored: bool = True,
    add_self_loop: bool = True,
    input_dim: int = 1,
    nodes_batch_size: int = 100,
    num_nodes: Optional[int] = None,
    freq_cache: Optional[torch.Tensor] = None,
    output_path: Optional[str] = None,
) -> list[dict]:
    """Run grid search over (max_freq, max_steps).

    Parameters
    ----------
    model : EmbeddingModel
    graphs : list of GraphData
    embs : list of embedding tensors
    anomalous_nodes : ground truth anomalous node IDs
    all_nodes : all node IDs
    starting_nodes : set of starting node IDs from outlier detection
    max_freq_range : list of max_freq values to test
    max_steps_range : list of max_steps values to test
    output_path : optional path to save results JSON

    Returns
    -------
    list[dict] : results for each (max_freq, max_steps) combination
    """
    if num_nodes is None:
        num_nodes = sum(g.num_nodes for g in graphs)

    scorer = SCORING.build("freq")
    results = []

    total = len(max_freq_range) * len(max_steps_range)
    print(f"\nGrid search: {len(max_freq_range)} freq × {len(max_steps_range)} steps = {total} configs", flush=True)
    print(f"{'max_freq':>10} {'max_steps':>10} {'AUROC':>8} {'F1':>8} {'P':>8} {'R':>8} {'AP':>8} {'V':>5} {'Time':>6}", flush=True)
    print("-" * 80, flush=True)

    best_auroc = 0.0
    best_config = None

    for max_freq in max_freq_range:
        for max_steps in max_steps_range:
            freq_thresh = max_freq / num_nodes
            max_strength = scorer(freq_thresh, max_steps / num_nodes, alpha)

            # Run search on all starting nodes
            t0 = time.time()
            verified_all = set()
            starting_list = list(starting_nodes)

            for batch_start in range(0, len(starting_list), nodes_batch_size):
                batch = starting_list[batch_start:batch_start + nodes_batch_size]
                batch = [n for n in batch if n not in {b.anchor() for b in verified_all}]
                if not batch:
                    continue

                agent = SEARCH.build(
                    search_strategy,
                    model=model,
                    graphs=graphs,
                    embs=embs,
                    scorer=scorer,
                    node_anchored=node_anchored,
                    add_self_loop=add_self_loop,
                    n_beams=n_beams,
                    min_strength=min_strength,
                    max_strength=max_strength,
                    alpha=alpha,
                    max_unchanged=max_unchanged,
                    unchange_direction=False,
                    min_steps=max_steps,
                    max_steps=max_steps,
                    max_cands=max_cands,
                    add_verified_neighs=add_verified_neighs,
                    min_neigh_repeat=min_neigh_repeat,
                    input_dim=input_dim,
                    freq_cache=freq_cache,
                )
                v = agent.run(batch, graph_idx=0)
                verified_all.update(set(v))

            elapsed = time.time() - t0
            stats = get_stat_results(anomalous_nodes, verified_all, all_nodes)

            row = {
                "max_freq": max_freq,
                "max_steps": max_steps,
                "auroc": stats["auroc"],
                "f1": stats["f1"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "ap": stats["ap"],
                "verified": len(verified_all),
                "time": elapsed,
            }
            results.append(row)

            print(
                f"{max_freq:10.1f} {max_steps:10d} "
                f"{stats['auroc']:8.4f} {stats['f1']:8.4f} "
                f"{stats['precision']:8.4f} {stats['recall']:8.4f} "
                f"{stats['ap']:8.4f} {len(verified_all):5d} "
                f"{elapsed:5.0f}s",
                flush=True,
            )

            if stats["auroc"] > best_auroc:
                best_auroc = stats["auroc"]
                best_config = row

    print("-" * 80, flush=True)
    if best_config:
        print(
            f"Best: max_freq={best_config['max_freq']:.1f}, "
            f"max_steps={best_config['max_steps']}, "
            f"AUROC={best_config['auroc']:.4f}, "
            f"F1={best_config['f1']:.4f}, "
            f"P={best_config['precision']:.4f}, "
            f"R={best_config['recall']:.4f}",
            flush=True,
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"results": results, "best": best_config}, f, indent=2)
        print(f"Results saved to {output_path}", flush=True)

    return results
