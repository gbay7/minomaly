"""Logging callback that prints progress to the console with timestamps."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from minomaly.callbacks.base import Callback
from minomaly.registry import CALLBACKS

logger = logging.getLogger(__name__)


def _ts() -> str:
    """Return a compact UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


@CALLBACKS.register("logging")
class LoggingCallback(Callback):
    """Logs progress to console with timestamps.

    Parameters
    ----------
    log_interval : int
        During training, print loss every *log_interval* batches.
    """

    def __init__(self, log_interval: int = 50) -> None:
        self.log_interval = log_interval

    # ── Search hooks ────────────────────────────────────────────────────

    def on_search_start(
        self, starting_nodes: list[int], config: Any
    ) -> None:
        print(
            f"[{_ts()}] Search started with {len(starting_nodes)} "
            f"starting node(s)"
        )

    def on_search_step(
        self,
        step: int,
        beam_sets: list,
        verified: list,
        new_verified_count: int,
    ) -> None:
        n_beams = sum(len(bs) for bs in beam_sets) if beam_sets else 0
        bar = "█" * step + "░" * max(0, 8 - step)
        print(
            f"  [{bar}] Step {step}: "
            f"{n_beams} beams, "
            f"{len(verified)} verified "
            f"(+{new_verified_count})",
            flush=True,
        )

    def on_search_batch_end(
        self,
        batch_number: int,
        batch_verified: list,
        cumulative_verified: set,
        elapsed_time: float,
    ) -> None:
        print(
            f"[{_ts()}] Batch {batch_number} done in {elapsed_time:.1f}s | "
            f"batch verified: {len(batch_verified)}, "
            f"cumulative verified: {len(cumulative_verified)}"
        )

    def on_search_end(
        self,
        all_verified: set,
        patterns: list,
        stats: dict,
    ) -> None:
        print(
            f"[{_ts()}] Search finished | "
            f"total verified: {len(all_verified)}, "
            f"unique patterns: {len(patterns)}"
        )
        if stats:
            for key, value in stats.items():
                print(f"  {key}: {value}")

    # ── Training hooks ──────────────────────────────────────────────────

    def on_training_batch_end(
        self, batch_idx: int, loss: float, metrics: dict
    ) -> None:
        if batch_idx % self.log_interval == 0:
            extra = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            suffix = f" | {extra}" if extra else ""
            print(
                f"[{_ts()}] Batch {batch_idx}: loss={loss:.6f}{suffix}"
            )

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_metrics: dict
    ) -> None:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in val_metrics.items()
        )
        suffix = f" | {metrics_str}" if metrics_str else ""
        print(
            f"[{_ts()}] Epoch {epoch}: train_loss={train_loss:.6f}{suffix}"
        )

    def on_training_end(self, model: Any, final_metrics: dict) -> None:
        print(f"[{_ts()}] Training finished")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
