"""Evaluation callback that computes detection metrics at each batch end."""

from __future__ import annotations

import json
import os
from typing import Any

from minomaly.callbacks.base import Callback
from minomaly.evaluation.metrics import get_stat_results
from minomaly.registry import CALLBACKS


@CALLBACKS.register("evaluation")
class EvaluationCallback(Callback):
    """Computes detection metrics at each batch end and writes results.

    Parameters
    ----------
    anomalous_nodes : list[int]
        Ground-truth anomalous node IDs.
    all_nodes : list[int]
        All node IDs in the graph (used to compute metrics over the
        full population).
    output_path : str
        Path to write the final ``anomalies.json`` file.
    """

    def __init__(
        self,
        anomalous_nodes: list[int],
        all_nodes: list[int],
        output_path: str = "anomalies.json",
    ) -> None:
        self.anomalous_nodes: set[int] = set(anomalous_nodes)
        self.all_nodes: list[int] = list(all_nodes)
        self.output_path: str = output_path
        self._batch_metrics: list[dict] = []

    # ── Search hooks ────────────────────────────────────────────────────

    def on_search_batch_end(
        self,
        batch_number: int,
        batch_verified: list,
        cumulative_verified: set,
        elapsed_time: float,
    ) -> None:
        """Compute and store P/R/F1/AUC/AP after each batch."""
        if not cumulative_verified:
            return

        verified_list = list(cumulative_verified)
        metrics = get_stat_results(
            self.anomalous_nodes, verified_list, self.all_nodes
        )
        metrics["batch_number"] = batch_number
        metrics["elapsed_time"] = elapsed_time
        self._batch_metrics.append(metrics)

    def on_search_end(
        self,
        all_verified: set,
        patterns: list,
        stats: dict,
    ) -> None:
        """Compute final metrics and write anomalies.json."""
        final_metrics: dict[str, Any] = {}
        if all_verified:
            verified_list = list(all_verified)
            final_metrics = get_stat_results(
                self.anomalous_nodes, verified_list, self.all_nodes
            )

        output = {
            "stats": {
                **stats,
                **final_metrics,
            },
            "batch_metrics": self._batch_metrics,
        }

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(output, f, indent=4, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for numpy types."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
