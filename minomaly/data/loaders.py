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
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pygod", "data")


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


def extract_anomaly_labels(data) -> torch.Tensor:
    """Extract structural anomaly labels from a PyGOD dataset.

    PyGOD encodes anomaly types via bit-flags in ``data.y``.
    Structural anomalies are indicated by bit 1: ``(y >> 1) & 1``.

    Parameters
    ----------
    data:
        A PyG-style data object with a ``.y`` attribute containing the
        encoded labels.

    Returns
    -------
    torch.Tensor
        A binary tensor of shape ``(num_nodes,)`` where ``1`` indicates a
        structural anomaly.
    """
    return (data.y >> 1) & 1


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
