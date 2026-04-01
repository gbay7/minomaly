"""Post-detection experiment analysis.

Generates diagnostic plots and summary tables after each detection run.
All outputs saved to the experiment's output folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_analysis(results: dict, plots_dir: Path, anomalous_nodes: list[int]) -> None:
    """Generate all analysis artifacts for an experiment."""
    anomalous_set = set(anomalous_nodes)
    verified = results.get("all_verified_beams", [])

    _plot_summary_table(results, plots_dir)
    _plot_freq_curves(verified, anomalous_set, plots_dir)
    _plot_size_distribution(verified, anomalous_set, plots_dir)
    _plot_precision_by_size(verified, anomalous_set, plots_dir)

    print(f"[analysis] Saved plots to {plots_dir}")


def _plot_summary_table(results: dict, plots_dir: Path) -> None:
    """Render P/R/F1/AUC/time as a clean summary image."""
    sr = results.get("stat_results", {})
    rows = [
        ["Precision", f"{sr.get('precision', 0):.4f}"],
        ["Recall", f"{sr.get('recall', 0):.4f}"],
        ["F1", f"{sr.get('f1', 0):.4f}"],
        ["AUROC", f"{sr.get('auroc', 0):.4f}"],
        ["AP", f"{sr.get('ap', 0):.4f}"],
        ["TP / FP / FN", f"{sr.get('tp', 0)} / {sr.get('fp', 0)} / {sr.get('fn', 0)}"],
        ["Verified", str(results.get("verified_count", 0))],
        ["Starting nodes", str(results.get("starting_nodes_count", 0))],
        ["Time", str(results.get("total_time", "N/A"))],
        ["Search", results.get("search_params", {}).get("search_strategy", "?")],
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_freq_curves(verified: list[dict], anomalous_set: set, plots_dir: Path) -> None:
    """Plot frequency vs step for TP and FP verified patterns."""
    tp_curves, fp_curves = [], []
    for b in verified:
        fh = b.get("freq_history", [])
        if len(fh) < 2:
            continue
        steps = [s for s, _ in fh]
        freqs = [f for _, f in fh]
        if b.get("is_true", False):
            tp_curves.append((steps, freqs))
        else:
            fp_curves.append((steps, freqs))

    if not tp_curves and not fp_curves:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for steps, freqs in tp_curves[:50]:
        ax1.plot(steps, freqs, alpha=0.3, color="green", linewidth=0.8)
    ax1.set_title(f"TP freq curves (n={len(tp_curves)})")
    ax1.set_xlabel("Pattern size")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim(bottom=-0.01)

    for steps, freqs in fp_curves[:50]:
        ax2.plot(steps, freqs, alpha=0.3, color="red", linewidth=0.8)
    ax2.set_title(f"FP freq curves (n={len(fp_curves)})")
    ax2.set_xlabel("Pattern size")
    ax2.set_ylabel("Frequency")
    ax2.set_ylim(bottom=-0.01)

    fig.tight_layout()
    fig.savefig(plots_dir / "freq_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_size_distribution(verified: list[dict], anomalous_set: set, plots_dir: Path) -> None:
    """Histogram of verified pattern sizes, split TP/FP."""
    if not verified:
        return

    tp_sizes = [b["neigh_len"] for b in verified if b.get("is_true", False)]
    fp_sizes = [b["neigh_len"] for b in verified if not b.get("is_true", False)]

    if not tp_sizes and not fp_sizes:
        return

    all_sizes = tp_sizes + fp_sizes
    bins = range(min(all_sizes), max(all_sizes) + 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    if tp_sizes:
        ax.hist(tp_sizes, bins=bins, alpha=0.7, label=f"TP (n={len(tp_sizes)})", color="green")
    if fp_sizes:
        ax.hist(fp_sizes, bins=bins, alpha=0.7, label=f"FP (n={len(fp_sizes)})", color="red")
    ax.set_xlabel("Pattern size (nodes)")
    ax.set_ylabel("Count")
    ax.set_title("Verified pattern size distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_precision_by_size(verified: list[dict], anomalous_set: set, plots_dir: Path) -> None:
    """Precision at each pattern size."""
    if not verified:
        return

    from collections import defaultdict
    by_size: dict[int, list[bool]] = defaultdict(list)
    for b in verified:
        by_size[b["neigh_len"]].append(b.get("is_true", False))

    sizes = sorted(by_size.keys())
    precisions = [sum(by_size[s]) / len(by_size[s]) for s in sizes]
    counts = [len(by_size[s]) for s in sizes]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(sizes, precisions, alpha=0.7, color="steelblue", label="Precision")
    ax1.set_xlabel("Pattern size")
    ax1.set_ylabel("Precision", color="steelblue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(sizes, counts, "ro-", markersize=5, label="Count")
    ax2.set_ylabel("Count", color="red")

    ax1.set_title("Precision by pattern size")
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_by_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
