"""Base callback with all hook points. Methods are no-ops by default."""

from __future__ import annotations

from abc import ABC
from typing import Any


class Callback(ABC):
    """Base callback with all hook points.

    Every method is a no-op by default so subclasses only need to override
    the hooks they care about.
    """

    # ── Search hooks ────────────────────────────────────────────────────

    def on_search_start(
        self, starting_nodes: list[int], config: Any
    ) -> None:
        pass

    def on_search_step(
        self,
        step: int,
        beam_sets: list,
        verified: list,
        new_verified_count: int,
    ) -> None:
        pass

    def on_search_batch_end(
        self,
        batch_number: int,
        batch_verified: list,
        cumulative_verified: set,
        elapsed_time: float,
    ) -> None:
        pass

    def on_search_end(
        self,
        all_verified: set,
        patterns: list,
        stats: dict,
    ) -> None:
        pass

    # ── Training hooks ──────────────────────────────────────────────────

    def on_training_start(self, model: Any, config: Any) -> None:
        pass

    def on_training_batch_end(
        self, batch_idx: int, loss: float, metrics: dict
    ) -> None:
        pass

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_metrics: dict
    ) -> None:
        pass

    def on_training_end(self, model: Any, final_metrics: dict) -> None:
        pass

    # ── Embedding hooks ─────────────────────────────────────────────────

    def on_embedding_batch(self, batch_idx: int, n_total: int) -> None:
        pass

    # ── Outlier detection hooks ─────────────────────────────────────────

    def on_outliers_detected(
        self, starting_nodes: set, method_counts: dict
    ) -> None:
        pass
