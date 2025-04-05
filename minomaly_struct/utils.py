import os
import pickle
import json
import json_numpy
import math

import torch

from common import utils
import random
from collections import defaultdict

from tqdm import tqdm

os.environ["MPLBACKEND"] = "agg"
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm
import matplotlib.pyplot as plt
# import tikzplotlib

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)

import networkx as nx


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


def embed_neighs(emb_model, neighs, anchors, batch_size, node_anchored=True):
    embs = []
    if len(neighs) % batch_size != 0:
        print("WARNING: number of graphs not multiple of batch size")
    for i in tqdm(range(len(neighs) // batch_size)):
        # top = min(len(neighs), (i+1)*args.batch_size)
        top = (i + 1) * batch_size
        with torch.no_grad():
            batch = utils.batch_nx_graphs(
                neighs[i * batch_size : top], anchors=anchors if node_anchored else None
            )
            emb = emb_model(batch)
            emb = emb.to(utils.get_device())
        embs.append(emb)
    return embs


def get_uniq_patterns(counts, out_batch_size):
    cand_patterns_uniq = []
    for pattern_size in counts.keys():
        for _, neighs in list(
            sorted(counts[pattern_size].items(), key=lambda x: len(x[1]), reverse=True)
        )[:out_batch_size]:
            print(" - Pattern size", pattern_size, "count", len(neighs))
            cand_patterns_uniq.append(random.choice(neighs))
    return cand_patterns_uniq


def concat_counts(counts1, counts2):
    for key1 in counts2:
        for key2 in counts2[key1]:
            counts1[key1][key2].extend(counts2[key1][key2])


def scatter_embs(embs_np, labels=None, legend=None, reduce_dim=None, out_path=None):
    if reduce_dim == "PCA":
        embs_np = PCA(n_components=2).fit_transform(embs_np)
    elif reduce_dim == "TSNE":
        embs_np = TSNE(n_components=2).fit_transform(embs_np)

    if labels is not None:
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx = np.where(labels == label)
            plt.scatter(
                embs_np[idx, 0], embs_np[idx, 1], label=legend[label], cmap=cm.tab20
            )
    else:
        plt.scatter(embs_np[:, 0], embs_np[:, 1], label=legend, cmap=cm.tab20)

    if out_path is not None:
        plt.legend()
        # tikzplotlib.save(f"{out_path}.tex")
        plt.savefig(out_path)
        plt.close()


def export_patterns(
    out_graphs,
    graphs_out_path="plots/cluster",
    results_out_path="results/out-patterns.p",
    node_anchored=True,
    count_by_anchor_=False,
):
    if count_by_anchor_:
        count_by_anchor = defaultdict(int)
    else:
        count_by_size = defaultdict(int)

    print("Saving graph patterns to", graphs_out_path)
    for pattern, anchor in out_graphs:
        node_colors = [
            "red" if node == anchor and node_anchored else "blue"
            for node in pattern.nodes
        ]
        if node_anchored:
            nx.draw(pattern, with_labels=True, node_color=node_colors)
        else:
            nx.draw(pattern, node_color=node_colors)

        if count_by_anchor_:
            path = os.path.join(
                graphs_out_path, "{}-{}.pdf".format(anchor, count_by_anchor[anchor])
            )
        else:
            path = os.path.join(
                graphs_out_path,
                "{}-{}.pdf".format(len(pattern), count_by_size[len(pattern)]),
            )
        plt.savefig(path)
        plt.close()
        if count_by_anchor_:
            count_by_anchor[anchor] += 1
        else:
            count_by_size[len(pattern)] += 1

    print("Saving pattern results to", results_out_path)
    if not os.path.exists(os.path.dirname(results_out_path)):
        os.makedirs(os.path.dirname(results_out_path))
    with open(results_out_path, "wb") as f:
        pickle.dump(out_graphs, f)


def find_outliers(
    neigh_embs, model, freq_thresh, real_anchors, neighs, max_neigh_len=None
):
    y_pred = []
    embs = [emb for emb_batch in neigh_embs for emb in emb_batch]

    for i in tqdm(range(len(embs)), desc="Finding outlier neighborhoods"):
        emb = embs[i]
        if max_neigh_len is not None and len(neighs[i]) > max_neigh_len:
            y_pred.append(2)
            continue
        freq, n_embs = 0, 0
        for emb_batch in neigh_embs:
            n_embs += len(emb_batch)
            supergraphs = torch.argmax(
                model.clf_model(
                    model.predict((emb_batch.to(utils.get_device()), emb)).unsqueeze(1)
                ),
                axis=1,
            )
            freq += torch.sum(supergraphs).item()
        if freq / n_embs <= freq_thresh:
            y_pred.append(-1)
        else:
            y_pred.append(0)
    real_anchors = np.array(real_anchors, dtype=int)
    starting_nodes = set(real_anchors[np.array(y_pred) == -1])
    starting_nodes = {node.item() for node in starting_nodes}

    for i, neigh in enumerate(neighs):
        if y_pred[i] == -1:
            neighbors = list(neigh.nodes)
            starting_nodes.update(neighbors)

    if_embs = np.array(
        [emb.cpu().numpy() for i, emb in enumerate(embs) if y_pred[i] == -1]
    )
    return starting_nodes, if_embs


def get_outlier_neighs(embs_np, real_anchors, neighs, max_neigh_len, contamination=0.2):
    clf = IsolationForest(contamination=contamination, random_state=42)
    neigh_len_cond = np.array([len(n) <= max_neigh_len for n in neighs])
    embs_np = embs_np[neigh_len_cond]
    y_pred = clf.fit_predict(embs_np)
    if_embs_np = embs_np[y_pred == -1]
    real_anchors = np.array(real_anchors, dtype=int)
    starting_nodes = set((real_anchors[neigh_len_cond])[y_pred == -1])
    starting_nodes = {node.item() for node in starting_nodes}
    return starting_nodes, if_embs_np


def write_verified(file_path, verified, dict_stats):
    with open(file_path, "w") as f:
        if dict_stats.get("true_anomalies") is not None:
            for verified_beam in verified:
                verified_beam.is_true = (
                    verified_beam.anchor() in dict_stats["true_anomalies"]
                )
        f.write(
            json_numpy.dumps(
                {
                    "stats": dict_stats,
                    "anomalies": [verified.to_dict() for verified in verified],
                },
                indent=4,
            )
        )


def find_n_neighborhoods(a, b):
    k = math.ceil(a / b)
    x = b * k - a
    return x


def get_stat_results(anomalous_nodes, verified, all_nodes):
    true_pred_nodes = [
        beam.anchor() for beam in verified if beam.anchor() in anomalous_nodes
    ]

    y_nodes_true = [node in anomalous_nodes for node in all_nodes]
    y_nodes_pred = [node in [beam.anchor() for beam in verified] for node in all_nodes]
    y_nodes_score = [
        next(((1 - beam.score) for beam in verified if beam.anchor() == node), 0)
        for node in all_nodes
    ]

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_nodes_true, y_nodes_score)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = 1 - float(thresholds[optimal_idx])

    cm = confusion_matrix(y_nodes_true, y_nodes_pred)
    print(cm)

    # Calculate the metrics
    accuracy = accuracy_score(y_nodes_true, y_nodes_pred)
    recall = recall_score(y_nodes_true, y_nodes_pred)
    precision = precision_score(y_nodes_true, y_nodes_pred)
    f1 = f1_score(y_nodes_true, y_nodes_pred)
    auroc = roc_auc_score(y_nodes_true, y_nodes_score)
    ap = average_precision_score(y_nodes_true, y_nodes_score)

    print(f"optimal_threshold: {optimal_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"AUROC: {auroc}")
    print(f"AP: {ap}")

    stat_results = {
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

    return stat_results
