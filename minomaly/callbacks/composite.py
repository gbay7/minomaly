"""Composite callback that dispatches all hook calls to a list of callbacks."""

from __future__ import annotations

import logging
from typing import Any

from minomaly.callbacks.base import Callback

logger = logging.getLogger(__name__)


class CallbackList(Callback):
    """Dispatches all hook calls to a list of callbacks.

    If a single callback raises an exception the error is logged but
    does not prevent the remaining callbacks from executing.
    """

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks: list[Callback] = list(callbacks)

    # ── Internal dispatcher ─────────────────────────────────────────────

    def _dispatch(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, method_name)(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Callback %s.%s raised an exception",
                    type(cb).__name__,
                    method_name,
                )

    # ── Search hooks ────────────────────────────────────────────────────

    def on_search_start(
        self, starting_nodes: list[int], config: Any
    ) -> None:
        self._dispatch("on_search_start", starting_nodes, config)

    def on_search_step(
        self,
        step: int,
        beam_sets: list,
        verified: list,
        new_verified_count: int,
    ) -> None:
        self._dispatch(
            "on_search_step", step, beam_sets, verified, new_verified_count
        )

    def on_search_batch_end(
        self,
        batch_number: int,
        batch_verified: list,
        cumulative_verified: set,
        elapsed_time: float,
    ) -> None:
        self._dispatch(
            "on_search_batch_end",
            batch_number,
            batch_verified,
            cumulative_verified,
            elapsed_time,
        )

    def on_search_end(
        self,
        all_verified: set,
        patterns: list,
        stats: dict,
    ) -> None:
        self._dispatch("on_search_end", all_verified, patterns, stats)

    # ── Training hooks ──────────────────────────────────────────────────

    def on_training_start(self, model: Any, config: Any) -> None:
        self._dispatch("on_training_start", model, config)

    def on_training_batch_end(
        self, batch_idx: int, loss: float, metrics: dict
    ) -> None:
        self._dispatch("on_training_batch_end", batch_idx, loss, metrics)

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_metrics: dict
    ) -> None:
        self._dispatch("on_epoch_end", epoch, train_loss, val_metrics)

    def on_training_end(self, model: Any, final_metrics: dict) -> None:
        self._dispatch("on_training_end", model, final_metrics)

    # ── Embedding hooks ─────────────────────────────────────────────────

    def on_embedding_batch(self, batch_idx: int, n_total: int) -> None:
        self._dispatch("on_embedding_batch", batch_idx, n_total)

    # ── Outlier detection hooks ─────────────────────────────────────────

    def on_outliers_detected(
        self, starting_nodes: set, method_counts: dict
    ) -> None:
        self._dispatch("on_outliers_detected", starting_nodes, method_counts)
