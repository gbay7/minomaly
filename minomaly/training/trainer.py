"""Accelerated training loop for the order-embedding GNN.

Optimizations over the original:
1. **Mixed precision (AMP)** — FP16 forward/backward, FP32 optimizer.
2. **Single batched forward pass** — stack all 4 graphs into one Batch
   instead of 4 separate forward passes.
3. **Prefetch data** — generate next batch while GPU is busy.
4. **Skip per-batch diagnostics** — only compute e_pos/e_neg every N steps.
5. **torch.compile** — JIT compile the model for faster kernels (PyTorch 2+).
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch_geometric.data import Batch
from tqdm import tqdm

from minomaly.config.schema import TrainingConfig
from minomaly.training.data_gen import TrainingPairGenerator
from minomaly.training.losses import order_embedding_loss
from minomaly.utils.device import get_device

logger = logging.getLogger(__name__)


# --- Process-pool data generation -------------------------------------------
# Batch generation is GIL-bound networkx work that costs ~5x a GPU step, so a
# single prefetch thread cannot hide it.  Worker processes generate CPU batches
# in parallel; the main loop moves them to GPU.  Functions are module-level so
# they pickle for ProcessPoolExecutor.

_WORKER_DG: TrainingPairGenerator | None = None


def _data_worker_init(dg_bytes: bytes, base_seed: int) -> None:
    """Reconstruct the generator in each worker and seed it distinctly."""
    global _WORKER_DG
    import os
    import random

    import numpy as np

    _WORKER_DG = pickle.loads(dg_bytes)
    seed = (base_seed * 1_000_003 + os.getpid()) % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)


def _data_worker_generate(batch_size: int) -> tuple[Batch, int]:
    """Generate one combined batch on CPU (moved to GPU by the main loop)."""
    return _WORKER_DG.generate_combined_batch(batch_size, device=torch.device("cpu"))


class OrderEmbeddingTrainer:
    """Accelerated training loop for order-embedding GNN models."""

    def __init__(
        self,
        model,
        train_config: TrainingConfig,
        data_generator: TrainingPairGenerator,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.config = train_config
        self.data_generator = data_generator
        self.callbacks = callbacks or []
        self.device = get_device()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()

        # AMP: disable for now — PyG message passing doesn't fully support FP16
        self.use_amp = False
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Try torch.compile (PyTorch 2+)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except Exception:
            pass  # torch.compile not available or failed

    def _build_optimizer(self) -> optim.Optimizer:
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        name = self.config.optimizer.lower()
        if name == "adam":
            return optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif name == "adamw":
            return optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif name == "sgd":
            return optim.SGD(params, lr=self.config.lr, momentum=0.9, weight_decay=self.config.weight_decay)
        else:
            return optim.Adam(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _generate_batch(self):
        """Generate a training batch (called in background thread).

        Returns one combined Batch (all four groups concatenated) plus the
        per-group size, so the encoder runs a single large forward pass.
        """
        return self.data_generator.generate_combined_batch(
            self.config.batch_size, device=self.device,
        )

    def train(self) -> dict[str, Any]:
        """Accelerated training loop."""
        self.model.train()
        cfg = self.config
        n_batches = cfg.n_batches
        batch_size = cfg.batch_size
        margin = getattr(self.model, "margin", 0.1)
        eval_interval = cfg.eval_interval
        clip_grad = cfg.clip_grad
        diag_interval = 50  # compute diagnostics every N batches

        running_loss = 0.0
        best_val_acc = 0.0
        best_auc = 0.0
        last_metrics: dict[str, Any] = {}

        # Prefetch setup: process pool (parallel, hides data-gen cost) or a
        # single background thread (default).
        num_workers = getattr(cfg, "num_data_workers", 0) or 0
        use_pool = num_workers > 0
        if use_pool:
            import multiprocessing as mp
            pool = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context("spawn"),
                initializer=_data_worker_init,
                initargs=(pickle.dumps(self.data_generator), 42),
            )
            prefetch_depth = num_workers + 2
            futures = deque(
                pool.submit(_data_worker_generate, batch_size)
                for _ in range(prefetch_depth)
            )
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._generate_batch)

        pbar = tqdm(range(n_batches), desc="Training")
        for batch_idx in pbar:
            # Get current batch, prefetch next
            if use_pool:
                combined, gsz = futures.popleft().result()
                if batch_idx < n_batches - prefetch_depth:
                    futures.append(pool.submit(_data_worker_generate, batch_size))
                combined = combined.to(self.device, non_blocking=True)
            else:
                combined, gsz = future.result()
                if batch_idx < n_batches - 1:
                    future = executor.submit(self._generate_batch)

            # Single forward pass on the combined batch, then split into the
            # four groups [pos_t, neg_t, pos_q, neg_q] (each of size gsz).
            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                emb_all = self.model.emb_model(combined)
                emb_pt, emb_nt, emb_pq, emb_nq = torch.split(emb_all, gsz)

                n_pos = emb_pt.size(0)
                n_neg = emb_nt.size(0)

                emb_a = torch.cat([emb_pt, emb_nt], dim=0)
                emb_b = torch.cat([emb_pq, emb_nq], dim=0)
                labels = torch.cat([
                    torch.ones(n_pos, device=self.device),
                    torch.zeros(n_neg, device=self.device),
                ])

                loss = self.model.criterion((emb_a, emb_b), None, labels)

            # Backward with AMP
            self.scaler.scale(loss).backward()
            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            running_loss += loss_val

            # Lightweight diagnostics (every N steps, no extra forward pass)
            if batch_idx % diag_interval == 0:
                with torch.no_grad():
                    e_pos = emb_a[:n_pos].detach()  # reuse existing embeddings
                    e_neg = emb_a[n_pos:].detach()
                    # Quick violation estimate from already-computed embeddings
                    e_all = self.model.predict((emb_a.detach(), emb_b.detach()))
                    e_pos_mean = e_all[:n_pos].mean().item()
                    e_neg_mean = e_all[n_pos:].mean().item()
                pbar.set_postfix(
                    loss=f"{loss_val:.2f}",
                    e_pos=f"{e_pos_mean:.3f}",
                    e_neg=f"{e_neg_mean:.3f}",
                )

            # Callbacks
            batch_metrics = {"loss": loss_val}
            for cb in self.callbacks:
                if hasattr(cb, "on_training_batch_end"):
                    cb.on_training_batch_end(batch_idx, loss_val, batch_metrics)

            # Periodic validation
            if (batch_idx + 1) % eval_interval == 0:
                val_metrics = self._validate(cfg.val_size)
                avg_loss = running_loss / eval_interval
                running_loss = 0.0

                epoch = (batch_idx + 1) // eval_interval
                epoch_metrics = {"epoch": epoch, "batch_idx": batch_idx, "avg_loss": avg_loss, **val_metrics}
                last_metrics = epoch_metrics

                vm = val_metrics
                tp, fp, fn, tn = vm.get("tp", 0), vm.get("fp", 0), vm.get("fn", 0), vm.get("tn", 0)
                pbar.set_postfix_str(
                    f"loss={avg_loss:.2f} "
                    f"acc={vm.get('val_acc', 0):.1%} "
                    f"P={vm.get('precision', 0):.1%} "
                    f"R={vm.get('recall', 0):.1%} "
                    f"F1={vm.get('f1', 0):.1%} "
                    f"AUC={vm.get('auc', 0):.3f} "
                    f"CM=[{tp}/{fp}/{fn}/{tn}] "
                    f"e+={vm.get('mean_e_pos', 0):.3f} "
                    f"e-={vm.get('mean_e_neg', 0):.3f}"
                )

                # Save best by AUC
                auc = vm.get("auc", 0.0)
                if auc > best_auc:
                    best_auc = auc
                    ckpt_dir = Path(cfg.checkpoint_dir)
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(ckpt_dir / "best_model.pt")

                val_acc = vm.get("val_acc", 0.0)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                for cb in self.callbacks:
                    if hasattr(cb, "on_epoch_end"):
                        cb.on_epoch_end(epoch, avg_loss, epoch_metrics)

        if use_pool:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=False)

        final_metrics: dict[str, Any] = {
            "n_batches": n_batches,
            "best_val_acc": best_val_acc,
            "best_auc": best_auc,
            **last_metrics,
        }

        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_checkpoint(ckpt_dir / "final_model.pt")
        self.save_checkpoint(ckpt_dir / "model.pt")

        for cb in self.callbacks:
            if hasattr(cb, "on_training_end"):
                cb.on_training_end(self.model, final_metrics)

        return final_metrics

    @torch.no_grad()
    def _validate(self, val_size: int) -> dict[str, Any]:
        """Validation with full metrics."""
        self.model.eval()
        margin = getattr(self.model, "margin", 0.1)
        half = val_size // 2

        pos_t, pos_q, neg_t, neg_q = self.data_generator.generate_batch(half, device=self.device)

        with autocast("cuda", enabled=self.use_amp):
            emb_pt = self.model.emb_model(pos_t)
            emb_pq = self.model.emb_model(pos_q)
            emb_nt = self.model.emb_model(neg_t)
            emb_nq = self.model.emb_model(neg_q)

        n_pos = emb_pt.size(0)
        n_neg = emb_nt.size(0)
        labels = torch.cat([torch.ones(n_pos, device=self.device), torch.zeros(n_neg, device=self.device)])
        emb_a = torch.cat([emb_pt, emb_nt], dim=0)
        emb_b = torch.cat([emb_pq, emb_nq], dim=0)

        loss = order_embedding_loss(emb_a, emb_b, labels, margin=margin)

        e = self.model.predict((emb_a, emb_b))
        pred = self.model.clf_model(e.unsqueeze(1))
        pred_labels = torch.argmax(pred, dim=1)
        gt = labels.long()

        correct = (pred_labels == gt).float()
        accuracy = correct.mean().item()
        pos_correct = correct[:n_pos].mean().item() if n_pos > 0 else 0.0
        neg_correct = correct[n_pos:].mean().item() if n_neg > 0 else 0.0

        tp = int(((pred_labels == 1) & (gt == 1)).sum().item())
        fp = int(((pred_labels == 1) & (gt == 0)).sum().item())
        fn = int(((pred_labels == 0) & (gt == 1)).sum().item())
        tn = int(((pred_labels == 0) & (gt == 0)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        e_pos = e[:n_pos].mean().item() if n_pos > 0 else 0.0
        e_neg = e[n_pos:].mean().item() if n_neg > 0 else 0.0

        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(gt.cpu().numpy(), -e.cpu().numpy())
        except Exception:
            auc = 0.0

        self.model.train()
        return {
            "val_loss": loss.item(), "val_acc": accuracy,
            "precision": precision, "recall": recall, "f1": f1, "auc": auc,
            "pos_acc": pos_correct, "neg_acc": neg_correct,
            "mean_e_pos": e_pos, "mean_e_neg": e_neg,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        torch.save(model.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
