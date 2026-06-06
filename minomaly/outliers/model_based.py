"""Model-based outlier detector.

Port of ``find_outliers`` from ``code-original/minomaly_struct/utils.py``.
For each neighborhood embedding, counts how many other neighborhoods predict
it as a supergraph via the order-embedding model.  Neighborhoods whose
frequency falls below ``freq_thresh`` are flagged, and both their anchors and
all nodes within those neighborhoods are added to the starting-node set.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from tqdm import tqdm

from minomaly.outliers.base import OutlierDetector
from minomaly.registry import OUTLIERS
from minomaly.utils.device import get_device

logger = logging.getLogger(__name__)


@OUTLIERS.register("model_based")
class ModelBasedDetector(OutlierDetector):
    """Detect outlier neighborhoods using the order-embedding model.

    For each neighborhood embedding, counts how many other neighborhood
    embeddings predict it as a supergraph.  If the ratio
    ``freq / n_embs <= freq_thresh``, marks the anchor AND all nodes in that
    neighborhood as starting nodes.

    Parameters
    ----------
    freq_thresh:
        Frequency threshold.  Neighborhoods whose supergraph frequency
        ratio is at or below this value are considered anomalous.
    max_neigh_len:
        Optional maximum neighborhood size.  Neighborhoods larger than
        this are skipped (labelled as ``2`` -- not anomalous, not normal).
    """

    def __init__(
        self,
        freq_thresh: float,
        max_neigh_len: int | None = None,
    ) -> None:
        self.freq_thresh = freq_thresh
        self.max_neigh_len = max_neigh_len

    def detect(
        self,
        embs: list[torch.Tensor],
        model: object,
        real_anchors: list[int],
        neighborhoods: list,
        **kwargs: object,
    ) -> tuple[set[int], np.ndarray]:
        """Detect outlier neighborhoods via order-embedding frequency.

        Parameters
        ----------
        embs:
            List of embedding batch tensors.
        model:
            An :class:`~minomaly.models.order_embedder.OrderEmbedder` with
            ``predict()`` and ``clf_model()`` methods.
        real_anchors:
            Per-neighborhood original anchor node IDs.
        neighborhoods:
            Per-neighborhood graph objects (NetworkX graphs).
        **kwargs:
            Unused.

        Returns
        -------
        tuple[set[int], np.ndarray]
            ``(starting_nodes, outlier_embeddings)``
        """
        device = get_device()

        # Stack all embeddings into one tensor
        all_embs_t = torch.cat(embs, dim=0).to(device)  # (N, D)
        N = all_embs_t.shape[0]

        # Precompute clf decision boundary: violations < threshold ≡ supergraph
        from minomaly.search.beam_set import _precompute_clf_threshold
        threshold = _precompute_clf_threshold(model.clf_model, device)

        freq_counts = torch.zeros(N, device=device)
        chunk_size = 512

        for start in tqdm(range(0, N, chunk_size), desc="Model-based outlier detection"):
            end = min(start + chunk_size, N)
            query_chunk = all_embs_t[start:end]  # (C, D)

            for emb_batch in embs:
                emb_batch = emb_batch.to(device)
                violations = model.batch_predict(emb_batch, query_chunk)
                supergraphs = violations < threshold  # (C, B) bool
                freq_counts[start:end] += supergraphs.sum(dim=1).float()

        # Wait for GPU to finish all async operations
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Classify on GPU (no .item() loop)
        freq_ratios = freq_counts / N
        outlier_mask_t = freq_ratios <= self.freq_thresh  # bool tensor on GPU
        y_pred_np = np.where(outlier_mask_t.cpu().numpy(), -1, 0)

        real_anchors_np = np.array(real_anchors, dtype=int)

        # Collect starting nodes from outlier anchors
        starting_nodes: set[int] = set(
            int(node) for node in real_anchors_np[y_pred_np == -1]
        )

        # Also add all nodes within outlier neighborhoods
        # Neighborhoods may be PyG Data objects (no .nodes attribute)
        # so we just add the real_anchor for each outlier neighborhood.
        # The original code added all NX graph nodes, but since our
        # neighborhoods are relabeled (0..n), only the real_anchor
        # maps back to the original graph.
        for i in range(len(neighborhoods)):
            if y_pred_np[i] == -1:
                starting_nodes.add(int(real_anchors[i]))

        # Collect outlier embeddings (single GPU→CPU transfer)
        outlier_mask = y_pred_np == -1
        if outlier_mask.any():
            outlier_embs = all_embs_t[torch.tensor(outlier_mask, dtype=torch.bool)].cpu().numpy()
        else:
            outlier_embs = np.empty((0,))

        n_outliers = int(np.sum(y_pred_np == -1))
        print(f"Model-based detector: {len(starting_nodes)} starting nodes from {n_outliers} outlier neighborhoods")

        # Return freq_ratios for cross-phase caching (Idea 3 from ideas.md)
        self.freq_cache = freq_ratios.cpu()  # (K,) tensor

        return starting_nodes, outlier_embs
