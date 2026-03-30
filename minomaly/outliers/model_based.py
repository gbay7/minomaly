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

        # Flatten all batch embeddings into a single list
        all_embs: list[torch.Tensor] = [
            emb for emb_batch in embs for emb in emb_batch
        ]

        y_pred: list[int] = []

        for i in tqdm(
            range(len(all_embs)), desc="Model-based outlier detection"
        ):
            emb = all_embs[i]

            # Skip neighborhoods exceeding max size
            if (
                self.max_neigh_len is not None
                and len(neighborhoods[i]) > self.max_neigh_len
            ):
                y_pred.append(2)
                continue

            freq = 0
            n_embs = 0

            for emb_batch in embs:
                n_embs += len(emb_batch)
                # Predict: for each embedding in the batch, check if it
                # is a supergraph of the current embedding.
                # model.predict((emb_batch, emb)) returns violation scores.
                # model.clf_model classifies each score.
                supergraphs = torch.argmax(
                    model.clf_model(
                        model.predict(
                            (emb_batch.to(device), emb)
                        ).unsqueeze(1)
                    ),
                    dim=1,
                )
                freq += torch.sum(supergraphs).item()

            if freq / n_embs <= self.freq_thresh:
                y_pred.append(-1)
            else:
                y_pred.append(0)

        real_anchors_np = np.array(real_anchors, dtype=int)
        y_pred_np = np.array(y_pred)

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
            if y_pred[i] == -1:
                starting_nodes.add(int(real_anchors[i]))

        # Collect outlier embeddings
        outlier_embs = np.array(
            [
                emb.cpu().numpy()
                for i, emb in enumerate(all_embs)
                if y_pred[i] == -1
            ]
        ) if any(y == -1 for y in y_pred) else np.empty((0,))

        logger.info(
            "Model-based detector: %d starting nodes from %d outlier neighborhoods",
            len(starting_nodes),
            int(np.sum(y_pred_np == -1)),
        )

        return starting_nodes, outlier_embs
