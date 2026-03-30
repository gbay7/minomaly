"""Checkpoint callback that saves model state during training."""

from __future__ import annotations

import os
from typing import Any

import torch

from minomaly.callbacks.base import Callback
from minomaly.registry import CALLBACKS


@CALLBACKS.register("checkpoint")
class CheckpointCallback(Callback):
    """Saves model checkpoints during training.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where checkpoint ``.pt`` files are written.
    save_interval : int
        Save a checkpoint every *save_interval* epochs.
    """

    def __init__(
        self,
        checkpoint_dir: str = "ckpt",
        save_interval: int = 1000,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval

    # ── Training hooks ──────────────────────────────────────────────────

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_metrics: dict
    ) -> None:
        """Save a model checkpoint at the configured interval."""
        if (epoch + 1) % self.save_interval != 0:
            return
        self._save(epoch, tag=f"epoch_{epoch}")

    def on_training_end(self, model: Any, final_metrics: dict) -> None:
        """Save the final model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "model_final.pt")
        state = (
            model.state_dict()
            if hasattr(model, "state_dict")
            else model
        )
        torch.save(state, path)
        print(f"Final checkpoint saved to {path}")

    # ── Internal ────────────────────────────────────────────────────────

    def _save(self, epoch: int, tag: str = "") -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"checkpoint_{tag}.pt" if tag else f"checkpoint_{epoch}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        # The model is not directly available from on_epoch_end,
        # so we store it from on_training_start if provided.
        if hasattr(self, "_model") and self._model is not None:
            state = (
                self._model.state_dict()
                if hasattr(self._model, "state_dict")
                else self._model
            )
            torch.save(state, path)
            print(f"Checkpoint saved to {path}")

    def on_training_start(self, model: Any, config: Any) -> None:
        """Stash a reference to the model for periodic checkpointing."""
        self._model = model
