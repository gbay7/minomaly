"""Dataset loading utilities for PyGOD anomaly-detection benchmarks.

Handles downloading, caching, and converting PyGOD datasets
(``inj_cora``, ``inj_amazon``, ``inj_flickr``, etc.) as well as
extracting structural anomaly labels and converting PyG Data to NetworkX.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import networkx as nx
import requests
import torch
import torch_geometric.utils as pyg_utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PYGOD_DATA_URL = "https://github.com/pygod-team/data/raw/main/{name}.pt.zip"
_DGL_FRAUD_URL = "https://data.dgl.ai/dataset/{name}.zip"
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pygod", "data")

_FRAUD_DATASETS = {
    "fraud_amazon": {"zip_name": "FraudAmazon", "mat_name": "Amazon.mat",
                     "relations": ["net_upu", "net_usu", "net_uvu", "homo"]},
    "fraud_yelp": {"zip_name": "FraudYelp", "mat_name": "YelpChi.mat",
                   "relations": ["net_rur", "net_rtr", "net_rsr", "homo"]},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pygod_dataset(
    name: str,
    cache_dir: Optional[str] = None,
) -> tuple:
    """Load a PyGOD anomaly-detection dataset, downloading if necessary.

    Parameters
    ----------
    name:
        Dataset identifier, e.g. ``"inj_cora"``, ``"inj_amazon"``,
        ``"inj_flickr"``.
    cache_dir:
        Directory for storing downloaded files.  Defaults to
        ``~/.pygod/data/``.

    Returns
    -------
    tuple
        ``(data, task_name)`` where *data* is the loaded PyG-style object
        (typically a :class:`torch_geometric.data.Data`) and *task_name*
        is a string describing the task (always ``"node"`` for PyGOD).
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    pt_path = cache_path / f"{name}.pt"

    if not pt_path.exists():
        zip_path = cache_path / f"{name}.pt.zip"
        url = _PYGOD_DATA_URL.format(name=name)

        print(f"Downloading {name} from {url} ...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        # Extract the .pt file from the zip archive.
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path)

        # Clean up the zip.
        zip_path.unlink(missing_ok=True)

        if not pt_path.exists():
            raise FileNotFoundError(
                f"Expected {pt_path} after extraction, but it was not found. "
                f"Contents of cache dir: {list(cache_path.iterdir())}"
            )

    data = torch.load(pt_path, weights_only=False)
    return data, "node"


def load_fraud_dataset(
    name: str,
    relation: str = "upu",
    cache_dir: Optional[str] = None,
) -> tuple:
    """Load a DGL fraud detection dataset as a homogeneous PyG graph.

    Bipartite fraud datasets (Amazon reviews, Yelp) are stored as
    multi-relational user/review graphs.  This loader extracts a single
    relation (default: co-purchase ``upu``) and converts it to a
    standard PyG ``Data`` object with PyGOD-style anomaly labels.

    Parameters
    ----------
    name:
        Dataset identifier: ``"fraud_amazon"`` or ``"fraud_yelp"``.
    relation:
        Which relation to use.  Short forms (``"upu"``, ``"usu"``,
        ``"uvu"``) are expanded automatically to ``"net_upu"`` etc.
        ``"homo"`` uses all relations combined.
    cache_dir:
        Download directory.  Defaults to ``~/.pygod/data/``.
    """
    import scipy.io as sio
    from torch_geometric.data import Data

    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    meta = _FRAUD_DATASETS[name]
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    mat_path = cache_path / meta["mat_name"]

    if not mat_path.exists():
        zip_path = cache_path / f"{meta['zip_name']}.zip"
        url = _DGL_FRAUD_URL.format(name=meta["zip_name"])
        print(f"Downloading {name} from {url} ...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_path)
        zip_path.unlink(missing_ok=True)

    mat = sio.loadmat(str(mat_path))

    rel_key = relation if relation in mat else f"net_{relation}"
    if rel_key not in mat:
        available = [k for k in mat if not k.startswith("_")]
        raise ValueError(f"Relation '{relation}' not found. Available: {available}")

    adj = mat[rel_key]
    import numpy as np
    coo = adj.tocoo()
    edge_index = torch.tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long
    )

    labels = torch.tensor(mat["label"].flatten(), dtype=torch.long)
    y = labels * 2

    num_nodes = len(labels)
    data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)

    n_anom = (labels == 1).sum().item()
    n_edges = edge_index.size(1)
    print(
        f"Loaded {name} ({rel_key}): {num_nodes:,} nodes, "
        f"{n_edges:,} edges, {n_anom:,} fraud anomalies"
    )
    return data, "node"


def load_pyg_dataset(
    name: str,
    cache_dir: Optional[str] = None,
    injection: str = "clique",
    injection_ratio: float = 0.05,
    dice_perturb: float = 0.5,
    group_size: int = 10,
    drop_prob: float = 0.0,
    seed: int = 42,
    n_outliers: Optional[int] = None,
) -> tuple:
    """Load a standard PyG dataset and inject structural outliers.

    Supported datasets: cora, citeseer, pubmed, amazon_computers,
    squirrel, chameleon, actor.

    Parameters
    ----------
    name:
        Dataset identifier.
    cache_dir:
        Root directory for PyG dataset downloads.
    injection:
        Outlier injection method: clique, star, biclique, cycle, path,
        tree, near_clique, dice, or none.
    injection_ratio:
        Fraction of nodes to inject as outliers.
    dice_perturb:
        DICE-n edge perturbation ratio.
    group_size:
        Size of each outlier group (for topology injectors).
    drop_prob:
        Edge drop probability (for near_clique).
    seed:
        Random seed for injection.

    Returns
    -------
    tuple
        ``(data, task_name)`` where anomaly labels are encoded in
        ``data.y`` using PyGOD bit format (bit 1 = structural).
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".pyg_datasets")

    from minomaly.data.injection import _PRESET_OUTLIER_NUM, inject_outliers

    _PLANETOID = {"cora", "citeseer", "pubmed"}
    _WIKI = {"squirrel", "chameleon"}

    if name in _PLANETOID:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=cache_dir, name=name.capitalize())
        data = dataset[0]
    elif name == "amazon_computers":
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=cache_dir, name="Computers")
        data = dataset[0]
    elif name in _WIKI:
        from torch_geometric.datasets import WikipediaNetwork
        dataset = WikipediaNetwork(root=cache_dir, name=name)
        data = dataset[0]
    elif name == "actor":
        from torch_geometric.datasets import Actor
        dataset = Actor(root=cache_dir)
        data = dataset[0]
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Supported: "
            f"{sorted(_PLANETOID | _WIKI | {'amazon_computers', 'actor'})}"
        )

    class_labels = data.y.clone()

    if injection == "none":
        data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    else:
        if n_outliers is None:
            n_outliers = _PRESET_OUTLIER_NUM.get(name)
        if n_outliers is None:
            n_outliers = int(data.num_nodes * injection_ratio)
        data, y_outlier = inject_outliers(
            data,
            method=injection,
            n_outliers=n_outliers,
            ratio=injection_ratio,
            seed=seed,
            dice_perturb=dice_perturb,
            group_size=group_size,
            drop_prob=drop_prob,
        )
        data.y = y_outlier * 2

    data.class_labels = class_labels

    n_anom = (data.y > 0).sum().item()
    print(
        f"Loaded {name}: {data.num_nodes:,} nodes, "
        f"{data.edge_index.size(1):,} edges, {n_anom:,} {injection} outliers"
    )
    return data, "node"


def load_organic_dataset(
    name: str,
    cache_dir: Optional[str] = None,
) -> tuple:
    """Load an organic anomaly dataset (e.g. ``elliptic``, ``mgtab``).

    These datasets are prepared by ``datasets/prepare_datasets.py`` and
    stored in the same cache directory as PyGOD datasets.  They use
    identical PyGOD bit-encoding for labels:
    ``(data.y >> 1) & 1 == 1`` for structural anomalies.

    Parameters
    ----------
    name:
        Dataset name (``"elliptic"`` or ``"mgtab"``).
    cache_dir:
        Directory containing the ``.pt`` file.  Defaults to
        ``~/.pygod/data/``.

    Returns
    -------
    tuple
        ``(data, task_name)`` matching :func:`load_pygod_dataset` signature.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    pt_path = Path(cache_dir) / f"{name}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {pt_path}. "
            f"Run: python datasets/prepare_datasets.py --dataset {name}"
        )

    data = torch.load(pt_path, weights_only=False)
    n_anom = ((data.y >> 1) & 1).sum().item()
    print(
        f"Loaded {name}: {data.num_nodes:,} nodes, "
        f"{data.edge_index.size(1):,} edges, {n_anom:,} anomalies"
    )
    return data, "node"


def load_molecules_dataset(
    name: str,
    anomaly_class: int = 0,
    n_anomaly_molecules: int = 20,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> tuple:
    """Load a TUDataset of molecules and combine into one graph.

    Each molecule becomes a connected component. Molecules from
    ``anomaly_class`` are subsampled to ``n_anomaly_molecules`` to
    create a rare structural anomaly signal.  Node features are
    discarded (topology only).

    Returns ``(data, "node")`` with PyGOD-style labels (bit 1 = structural).
    """
    import random
    from torch_geometric.data import Data
    from torch_geometric.datasets import TUDataset

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".pyg_datasets")

    _TU_NAMES = {
        "aids": "AIDS", "mutag": "MUTAG", "nci1": "NCI1", "nci109": "NCI109",
        "ptc_mr": "PTC_MR", "imdb-binary": "IMDB-BINARY", "imdb-multi": "IMDB-MULTI",
        "enzymes": "ENZYMES", "proteins": "PROTEINS", "dd": "DD",
        "reddit-binary": "REDDIT-BINARY", "collab": "COLLAB",
        "syntheticnew": "SYNTHETICnew", "synthie": "Synthie",
    }
    tu_name = _TU_NAMES.get(name, name.upper())
    ds = TUDataset(root=cache_dir, name=tu_name)

    normal = [d for d in ds if int(d.y) != anomaly_class]
    anomalous = [d for d in ds if int(d.y) == anomaly_class]

    rng = random.Random(seed)
    rng.shuffle(anomalous)
    anomalous = anomalous[:n_anomaly_molecules]

    all_graphs = normal + anomalous
    rng.shuffle(all_graphs)

    edge_indices = []
    labels = []
    offset = 0

    for g in all_graphs:
        ei = g.edge_index + offset
        edge_indices.append(ei)
        is_anom = int(g.y) == anomaly_class
        labels.extend([2 if is_anom else 0] * g.num_nodes)
        offset += g.num_nodes

    combined_edge_index = torch.cat(edge_indices, dim=1)
    combined_y = torch.tensor(labels, dtype=torch.long)

    data = Data(
        edge_index=combined_edge_index,
        y=combined_y,
        num_nodes=offset,
    )

    n_anom_nodes = (combined_y > 0).sum().item()
    n_anom_mols = n_anomaly_molecules
    print(
        f"Loaded {name}: {len(normal)} normal + {n_anom_mols} anomaly molecules "
        f"→ {offset:,} nodes, {combined_edge_index.size(1):,} edges, "
        f"{n_anom_nodes} anomalous nodes"
    )
    return data, "node"


def extract_anomaly_labels(data, task: str = "struct-anomaly") -> torch.Tensor:
    """Extract anomaly labels from a PyGOD dataset.

    PyGOD encodes anomaly types via bit-flags in ``data.y``:
    - bit 0: contextual anomaly
    - bit 1: structural anomaly

    Parameters
    ----------
    data:
        A PyG-style data object with a ``.y`` attribute.
    task:
        Which anomalies to extract:
        - ``"struct-anomaly"``: structural only (bit 1)
        - ``"context-anomaly"``: contextual only (bit 0)
        - ``"all-anomaly"``: any anomaly (y > 0)

    Returns
    -------
    torch.Tensor
        A binary tensor of shape ``(num_nodes,)`` where ``1`` indicates
        an anomaly of the requested type.
    """
    if task == "struct-anomaly":
        return (data.y >> 1) & 1
    elif task == "context-anomaly":
        return data.y & 1
    elif task == "all-anomaly":
        return (data.y > 0).long()
    else:
        raise ValueError(f"Unknown task: {task}. Use 'struct-anomaly', 'context-anomaly', or 'all-anomaly'.")


def pyg_data_to_nx(data) -> nx.Graph:
    """Convert a PyG :class:`Data` object to an undirected NetworkX graph.

    Parameters
    ----------
    data:
        A PyG ``Data`` object with at least an ``edge_index`` attribute.

    Returns
    -------
    nx.Graph
        An undirected NetworkX graph.
    """
    G = pyg_utils.to_networkx(data, to_undirected=True)
    return G
