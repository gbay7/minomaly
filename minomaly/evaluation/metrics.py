"""Detection metrics for anomaly evaluation.

Ported from ``code-original/minomaly_struct/utils.py:get_stat_results``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_stat_results(
    anomalous_nodes: Iterable[int],
    verified_beams: list[Any],
    all_nodes: list[int],
) -> dict[str, Any]:
    """Compute detection metrics.

    Parameters
    ----------
    anomalous_nodes : iterable of int
        Ground-truth anomalous node IDs.
    verified_beams : list
        Verified beam objects. Each beam should expose an ``.anchor()``
        method returning the anchor node ID, and a ``.score`` attribute
        with the anomaly score (lower = more anomalous).
    all_nodes : list of int
        All node IDs in the graph.

    Returns
    -------
    dict
        Dictionary containing accuracy, recall, precision, F1, AUROC, AP,
        confusion-matrix entries (tp/tn/fp/fn), optimal threshold, ROC
        curve data, and correctly predicted node IDs.
    """
    anomalous_set = set(anomalous_nodes)

    # Extract anchor for each verified beam
    verified_anchors: list[int] = [beam.anchor() for beam in verified_beams]

    true_pred_nodes: list[int] = [
        anchor for anchor in verified_anchors if anchor in anomalous_set
    ]

    verified_anchor_set = set(verified_anchors)

    # Binary ground-truth and predictions for every node
    y_true = [node in anomalous_set for node in all_nodes]
    y_pred = [node in verified_anchor_set for node in all_nodes]

    # Continuous anomaly scores: for verified nodes use (1 - beam.score),
    # for non-verified nodes use 0.
    anchor_to_score: dict[int, float] = {}
    for beam in verified_beams:
        a = beam.anchor()
        s = beam.score if beam.score is not None else 0.0
        score = 1.0 - s
        if a not in anchor_to_score or score > anchor_to_score[a]:
            anchor_to_score[a] = score
    y_score = [anchor_to_score.get(node, 0.0) for node in all_nodes]

    # ROC curve and optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = 1.0 - float(thresholds[optimal_idx])

    cm = confusion_matrix(y_true, y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_score))
    ap = float(average_precision_score(y_true, y_score))

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "ap": ap,
        "tp": int(cm[1][1]),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "true_pred_nodes": true_pred_nodes,
        "optimal_threshold": optimal_threshold,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }
