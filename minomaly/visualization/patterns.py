"""Pattern graph export utilities."""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import networkx as nx


def export_patterns(
    out_graphs: list[tuple[Any, int]],
    graphs_out_path: str = "plots/cluster",
    results_out_path: str = "results/out-patterns.p",
    node_anchored: bool = True,
    count_by_anchor: bool = False,
) -> None:
    """Draw and save pattern graphs as PDFs, and pickle the raw data.

    Parameters
    ----------
    out_graphs : list of (graph, anchor) tuples
        Each *graph* is an ``nx.Graph`` or any object exposing a
        ``to_nx()`` method.  *anchor* is the integer node ID that
        anchors the pattern.
    graphs_out_path : str
        Directory for per-pattern PDF files.
    results_out_path : str
        Path for the pickled pattern list.
    node_anchored : bool
        If *True*, highlight the anchor node in red and show labels.
    count_by_anchor : bool
        If *True*, file names are ``{anchor}-{idx}.pdf``; otherwise
        ``{size}-{idx}.pdf``.
    """
    os.makedirs(graphs_out_path, exist_ok=True)

    counter: dict[Any, int] = defaultdict(int)

    print(f"Saving graph patterns to {graphs_out_path}")
    for pattern, anchor in out_graphs:
        # Convert to nx.Graph when needed
        if hasattr(pattern, "to_nx"):
            pattern = pattern.to_nx()
        if not isinstance(pattern, nx.Graph):
            continue

        node_colors = [
            "red" if (node_anchored and node == anchor) else "blue"
            for node in pattern.nodes
        ]

        plt.figure()
        if node_anchored:
            nx.draw(pattern, with_labels=True, node_color=node_colors)
        else:
            nx.draw(pattern, node_color=node_colors)

        if count_by_anchor:
            key = anchor
            path = os.path.join(
                graphs_out_path, f"{anchor}-{counter[key]}.pdf"
            )
        else:
            key = len(pattern)
            path = os.path.join(
                graphs_out_path, f"{key}-{counter[key]}.pdf"
            )

        plt.savefig(path)
        plt.close()
        counter[key] += 1

    # Pickle the raw pattern list
    print(f"Saving pattern results to {results_out_path}")
    results_dir = os.path.dirname(results_out_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_out_path, "wb") as f:
        pickle.dump(out_graphs, f)
