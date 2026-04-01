"""Anomalous structure interpretation and visualization.

Matches the paper's Table 7/8/9 style:
- Red nodes = true anomalous (ground truth)
- Green nodes = true normal (ground truth)
- Each pattern shows Count and A/N ratio

Generates:
- Pattern gallery (grid of all detected structures, TP/FP separated)
- Deduplicated pattern catalog (by WL hash, like Table 7)
- Context view: pattern within its local neighborhood
- Frequency trajectory comparison TP vs FP
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import torch

from minomaly.data.graph import GraphData
from minomaly.hashing import wl_hash
from minomaly.search.beam import Beam


# ── Colors matching paper ────────────────────────────────────────────
RED = "#e74c3c"       # true anomalous node
GREEN = "#2ecc71"     # true normal node
ANCHOR_EDGE = "#c0392b"  # highlight anchor with thick border
EDGE_COLOR = "#95a5a6"
CONTEXT_GRAY = "#ecf0f1"
CONTEXT_EDGE = "#bdc3c7"


def interpret_patterns(
    verified_beams: list[Beam],
    graph: GraphData,
    anomalous_nodes: set[int],
    output_dir: Path,
    max_patterns: int = 50,
    context_hops: int = 1,
) -> None:
    """Generate all interpretation artifacts for detected patterns."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tp = [b for b in verified_beams if b.anchor() in anomalous_nodes]
    fp = [b for b in verified_beams if b.anchor() not in anomalous_nodes]

    # 1. Pattern gallery (paper Table 7 style)
    _plot_pattern_gallery(tp, "True Positives", anomalous_nodes, output_dir / "gallery_tp.png", max_patterns)
    _plot_pattern_gallery(fp, "False Positives", anomalous_nodes, output_dir / "gallery_fp.png", max_patterns)

    # 2. Deduplicated pattern catalog (like Table 7)
    _plot_pattern_catalog(verified_beams, anomalous_nodes, output_dir / "catalog.png")

    # 3. Individual patterns with context
    indiv_dir = output_dir / "individual"
    indiv_dir.mkdir(exist_ok=True)
    for i, beam in enumerate(verified_beams[:max_patterns]):
        is_tp = beam.anchor() in anomalous_nodes
        label = "TP" if is_tp else "FP"
        _plot_pattern_with_context(
            beam, graph, anomalous_nodes, context_hops,
            indiv_dir / f"{label}_{beam.anchor()}_size{len(beam.neigh)}.png",
        )

    # 4. Frequency trajectory comparison
    _plot_freq_trajectories(tp, fp, output_dir / "freq_trajectories.png")

    # 5. Missed anomalies (false negatives)
    detected_anchors = {b.anchor() for b in verified_beams}
    _plot_missed_anomalies(graph, anomalous_nodes, detected_anchors, output_dir / "missed_anomalies.png")

    # 6. Summary
    _write_interpretation_summary(tp, fp, anomalous_nodes, output_dir / "interpretation.txt")

    print(f"[interpretation] {len(tp)} TP, {len(fp)} FP patterns → {output_dir}")


def _beam_to_nx(beam: Beam, anomalous_nodes: set[int]) -> tuple[nx.Graph, int, dict]:
    """Convert beam to networkx graph with ground-truth labels.

    Returns (graph, local_anchor_id, global_to_local_mapping).
    """
    neigh_set = set(beam.neigh)
    anchor = beam.anchor()

    # Mapping: anchor → 0, rest sorted
    mapping = {anchor: 0}
    rest = sorted(neigh_set - {anchor})
    mapping.update({n: i + 1 for i, n in enumerate(rest)})

    g = nx.Graph()
    g.add_nodes_from(range(len(mapping)))

    for node in beam.neigh:
        for nb in beam.graph.neighbors(node).tolist():
            if nb in neigh_set and nb != node:
                u, v = mapping[node], mapping[nb]
                g.add_edge(u, v)

    # Node attributes
    for node, local in mapping.items():
        g.nodes[local]["global_id"] = node
        g.nodes[local]["is_anomalous"] = node in anomalous_nodes
        g.nodes[local]["is_anchor"] = node == anchor

    return g, 0, mapping


def _draw_pattern(
    ax: plt.Axes,
    beam: Beam,
    anomalous_nodes: set[int],
    title: str = "",
    show_ids: bool = True,
) -> None:
    """Draw a single pattern on axes. Red=anomalous, Green=normal (paper style)."""
    g, anchor_local, mapping = _beam_to_nx(beam, anomalous_nodes)

    if len(g) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center")
        ax.set_title(title, fontsize=8)
        return

    pos = nx.spring_layout(g, seed=42, k=2.0 / max(1, len(g) ** 0.5))

    # Colors: red = anomalous, green = normal (matching paper)
    colors = [RED if g.nodes[n]["is_anomalous"] else GREEN for n in g.nodes]

    # Anchor gets thick black border
    edgecolors = ["black" if n == anchor_local else "white" for n in g.nodes]
    linewidths = [2.5 if n == anchor_local else 1.0 for n in g.nodes]

    # Labels: show global node IDs
    labels = {n: str(g.nodes[n]["global_id"]) for n in g.nodes} if show_ids else {}

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=EDGE_COLOR, width=1.5)
    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=colors, node_size=350,
        edgecolors=edgecolors, linewidths=linewidths,
    )
    if labels:
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6, font_color="white", font_weight="bold")

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _pattern_stats(beam: Beam, anomalous_nodes: set[int]) -> tuple[int, int]:
    """Return (n_anomalous, n_normal) in the pattern."""
    a = sum(1 for n in beam.neigh if n in anomalous_nodes)
    return a, len(beam.neigh) - a


def _plot_pattern_gallery(
    beams: list[Beam],
    title: str,
    anomalous_nodes: set[int],
    output_path: Path,
    max_patterns: int = 50,
) -> None:
    """Grid of pattern thumbnails with A/N stats."""
    beams = beams[:max_patterns]
    if not beams:
        return

    n = len(beams)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, beam in enumerate(beams):
        r, c = i // cols, i % cols
        a_count, n_count = _pattern_stats(beam, anomalous_nodes)
        _draw_pattern(
            axes[r, c], beam, anomalous_nodes,
            title=f"anchor={beam.anchor()}\nA/N: {a_count}/{n_count}  freq={beam.freq:.4f}",
        )

    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    fig.suptitle(f"{title} (n={len(beams)})", fontsize=14, y=1.01)

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=10, label="True anomalous node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=10, label="True normal node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10,
               markeredgecolor="black", markeredgewidth=2, label="Anchor node (black border)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pattern_catalog(
    beams: list[Beam],
    anomalous_nodes: set[int],
    output_path: Path,
) -> None:
    """Deduplicated catalog like paper Table 7: one per WL hash, with Count and A/N."""
    # Group by WL hash
    hash_groups: dict[tuple, list[Beam]] = defaultdict(list)
    for beam in beams:
        g, anchor_local, _ = _beam_to_nx(beam, anomalous_nodes)
        for n in g.nodes:
            g.nodes[n]["anchor"] = 1 if n == anchor_local else 0
        h = wl_hash(g, node_anchored=True)
        hash_groups[h].append(beam)

    sorted_groups = sorted(hash_groups.items(), key=lambda kv: -len(kv[1]))

    if not sorted_groups:
        return

    n = min(20, len(sorted_groups))

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, (h, group) in enumerate(sorted_groups[:n]):
        r, c = i // cols, i % cols
        beam = group[0]  # representative
        count = len(group)
        a_count, n_count = _pattern_stats(beam, anomalous_nodes)
        # Compute A/N across all instances
        total_a = sum(_pattern_stats(b, anomalous_nodes)[0] for b in group)
        total_n = sum(_pattern_stats(b, anomalous_nodes)[1] for b in group)
        avg_a = total_a / count
        avg_n = total_n / count

        _draw_pattern(
            axes[r, c], beam, anomalous_nodes,
            title=f"{len(beam.neigh)}-{i}\nCount: {count}   A/N: {a_count}/{n_count}",
        )

    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    fig.suptitle(f"Pattern Catalog ({len(sorted_groups)} unique patterns)", fontsize=14)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=10, label="True anomalous node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=10, label="True normal node"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pattern_with_context(
    beam: Beam,
    graph: GraphData,
    anomalous_nodes: set[int],
    context_hops: int,
    output_path: Path,
) -> None:
    """Draw pattern in local neighborhood. Red=anomalous, green=normal, gray=context."""
    pattern_nodes = set(beam.neigh)
    anchor = beam.anchor()

    # Expand to context via BFS
    context = set(pattern_nodes)
    frontier = set(pattern_nodes)
    for _ in range(context_hops):
        new_frontier = set()
        for node in frontier:
            for nb in graph.neighbors(node).tolist():
                if nb not in context:
                    new_frontier.add(nb)
        context |= new_frontier
        frontier = new_frontier

    context_list = sorted(context)
    mapping = {n: i for i, n in enumerate(context_list)}
    g = nx.Graph()
    g.add_nodes_from(range(len(context_list)))

    for node in context_list:
        for nb in graph.neighbors(node).tolist():
            if nb in context and nb != node:
                g.add_edge(mapping[node], mapping[nb])

    # Node appearance
    colors = []
    sizes = []
    edgecolors = []
    linewidths = []
    for n in g.nodes:
        gid = context_list[n]
        in_pattern = gid in pattern_nodes
        is_anom = gid in anomalous_nodes
        is_anchor = gid == anchor

        if in_pattern:
            colors.append(RED if is_anom else GREEN)
            sizes.append(400 if is_anchor else 300)
            edgecolors.append("black" if is_anchor else "white")
            linewidths.append(3.0 if is_anchor else 1.0)
        else:
            # Context nodes: faded version
            colors.append("#fadbd8" if is_anom else CONTEXT_GRAY)
            sizes.append(120)
            edgecolors.append("none")
            linewidths.append(0)

    # Edge appearance
    edge_colors = []
    edge_widths = []
    for u, v in g.edges:
        gu, gv = context_list[u], context_list[v]
        if gu in pattern_nodes and gv in pattern_nodes:
            edge_colors.append("#7f8c8d")
            edge_widths.append(2.0)
        else:
            edge_colors.append(CONTEXT_EDGE)
            edge_widths.append(0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(g, seed=42, k=1.5 / max(1, len(g) ** 0.5))

    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=colors, node_size=sizes,
        edgecolors=edgecolors, linewidths=linewidths,
    )

    # Label only pattern nodes
    pattern_labels = {mapping[n]: str(n) for n in pattern_nodes if n in mapping}
    nx.draw_networkx_labels(g, pos, labels=pattern_labels, ax=ax, font_size=7, font_color="white", font_weight="bold")

    is_tp = anchor in anomalous_nodes
    a_count, n_count = _pattern_stats(beam, anomalous_nodes)
    ax.set_title(
        f"{'TP' if is_tp else 'FP'} | anchor={anchor} | "
        f"size={len(beam.neigh)} | A/N: {a_count}/{n_count} | freq={beam.freq:.4f}\n"
        f"context: {len(context)} nodes",
        fontsize=10,
    )
    ax.axis("off")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=10, label="Anomalous node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=10, label="Normal node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CONTEXT_GRAY, markersize=8, label="Context (1-hop)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10,
               markeredgecolor="black", markeredgewidth=2, label="Anchor"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_freq_trajectories(
    tp_beams: list[Beam],
    fp_beams: list[Beam],
    output_path: Path,
) -> None:
    """Frequency trajectory comparison TP vs FP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for beam in tp_beams[:30]:
        fh = beam.freq_history
        if len(fh) >= 2:
            steps, freqs = zip(*fh)
            ax1.plot(steps, freqs, "o-", color=RED, alpha=0.4, markersize=3, linewidth=1)
    ax1.set_title(f"TP frequency trajectories (n={len(tp_beams)})", fontsize=12)
    ax1.set_xlabel("Pattern size")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim(bottom=-0.02)
    ax1.grid(True, alpha=0.3)

    for beam in fp_beams[:30]:
        fh = beam.freq_history
        if len(fh) >= 2:
            steps, freqs = zip(*fh)
            ax2.plot(steps, freqs, "o-", color=GREEN, alpha=0.4, markersize=3, linewidth=1)
    ax2.set_title(f"FP frequency trajectories (n={len(fp_beams)})", fontsize=12)
    ax2.set_xlabel("Pattern size")
    ax2.set_ylabel("Frequency")
    ax2.set_ylim(bottom=-0.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_interpretation_summary(
    tp_beams: list[Beam],
    fp_beams: list[Beam],
    anomalous_nodes: set[int],
    output_path: Path,
) -> None:
    """Write text summary of detected patterns."""
    lines = [
        "Interpretation Summary",
        "=" * 40,
        f"Total verified: {len(tp_beams) + len(fp_beams)}",
        f"True positives: {len(tp_beams)}",
        f"False positives: {len(fp_beams)}",
        f"Precision: {len(tp_beams) / max(1, len(tp_beams) + len(fp_beams)):.3f}",
        "",
    ]

    lines.append("True Positive Patterns:")
    lines.append("-" * 40)
    for b in sorted(tp_beams, key=lambda x: x.freq or 0):
        a, n = _pattern_stats(b, anomalous_nodes)
        lines.append(
            f"  anchor={b.anchor():5d} size={len(b.neigh)} "
            f"A/N={a}/{n} freq={b.freq:.4f} neigh={b.neigh}"
        )

    lines.append("")
    lines.append("False Positive Patterns:")
    lines.append("-" * 40)
    for b in sorted(fp_beams, key=lambda x: x.freq or 0):
        a, n = _pattern_stats(b, anomalous_nodes)
        lines.append(
            f"  anchor={b.anchor():5d} size={len(b.neigh)} "
            f"A/N={a}/{n} freq={b.freq:.4f} neigh={b.neigh}"
        )

    output_path.write_text("\n".join(lines))


def _plot_missed_anomalies(
    graph: GraphData,
    anomalous_nodes: set[int],
    detected_anchors: set[int],
    output_path: Path,
) -> None:
    """Visualize missed anomalies (FN) in their 1-hop neighborhood."""
    missed = sorted(anomalous_nodes - detected_anchors)
    if not missed:
        return

    GOLD = "#f1c40f"
    n = min(len(missed), 10)
    missed = missed[:n]

    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, node in enumerate(missed):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        neighs = set(graph.neighbors(node).tolist())
        context = {node} | neighs
        context_list = sorted(context)
        mapping = {n: i for i, n in enumerate(context_list)}

        g = nx.Graph()
        g.add_nodes_from(range(len(context_list)))
        for nd in context_list:
            for nb in graph.neighbors(nd).tolist():
                if nb in context and nb != nd:
                    g.add_edge(mapping[nd], mapping[nb])

        pos = nx.spring_layout(g, seed=42, k=1.8 / max(1, len(g) ** 0.5))

        colors, sizes, ec, lw = [], [], [], []
        for nd in g.nodes:
            gid = context_list[nd]
            if gid == node:
                colors.append(GOLD); sizes.append(500)
                ec.append("black"); lw.append(3.0)
            elif gid in anomalous_nodes and gid in detected_anchors:
                colors.append(RED); sizes.append(300)
                ec.append("white"); lw.append(1.0)
            elif gid in anomalous_nodes:
                colors.append("#e67e22"); sizes.append(300)
                ec.append("black"); lw.append(2.0)
            else:
                colors.append(GREEN); sizes.append(150)
                ec.append("white"); lw.append(0.5)

        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#bdc3c7", width=1.0)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes,
                               edgecolors=ec, linewidths=lw)
        labels = {n: str(context_list[n]) for n in g.nodes
                  if context_list[n] in anomalous_nodes or context_list[n] == node}
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6,
                                font_color="white", font_weight="bold")

        anom_n = sum(1 for n in neighs if n in anomalous_nodes)
        det_n = sum(1 for n in neighs if n in detected_anchors)
        ax.set_title(f"MISSED: {node}\ndeg={len(neighs)}, anom_nb={anom_n}, det_nb={det_n}", fontsize=9)
        ax.axis("off")

    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GOLD, markersize=12,
               markeredgecolor="black", markeredgewidth=2, label="Missed anomaly (FN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=10,
               label="Detected anomaly (TP)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=10,
               markeredgecolor="black", markeredgewidth=2, label="Undetected anomaly"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=8,
               label="Normal node"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Missed Anomalies — False Negatives (n={len(missed)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
