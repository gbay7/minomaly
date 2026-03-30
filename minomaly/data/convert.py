"""Convert between graph representations and PyG Data objects.

This module replaces the DeepSNAP dependency used in the original codebase.
It works with both NetworkX graphs (for backward compat / generators)
and the native :class:`~minomaly.data.graph.GraphData` / ``SubgraphView``.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import coalesce

if TYPE_CHECKING:
    from minomaly.data.graph import SubgraphView


def subgraph_to_pyg(
    view: SubgraphView,
    anchor_global: Optional[int] = None,
    node_anchored: bool = True,
    add_self_loop: bool = True,
) -> Data:
    """Convert a :class:`SubgraphView` to a PyG Data without NX round-trip."""
    return view.to_pyg(
        anchor_global=anchor_global,
        node_anchored=node_anchored,
        add_self_loop=add_self_loop,
    )


def batch_pyg_data(
    data_list: list[Data],
    device: Optional[torch.device] = None,
) -> Batch:
    """Batch pre-built PyG Data objects (from GraphData or SubgraphView)."""
    if device is None:
        from minomaly.utils.device import get_device
        device = get_device()
    batch = Batch.from_data_list(data_list)
    return batch.to(device)


def nx_to_pyg(
    g: nx.Graph,
    anchor: Optional[int] = None,
    node_anchored: bool = True,
    add_self_loop: bool = True,
) -> Data:
    """Convert a single NetworkX graph to a PyG :class:`Data` object.

    Parameters
    ----------
    g:
        An undirected NetworkX graph.  Node labels may be any hashable type;
        they will be relabelled to contiguous integers ``0 .. n-1``.
    anchor:
        The node id (in the *original* labelling of *g*) that acts as the
        anchor.  Ignored when *node_anchored* is ``False``.
    node_anchored:
        When ``True`` the node feature ``x`` is set to a 1-d anchor indicator
        (``1.0`` for the anchor node, ``0.0`` elsewhere).  When ``False`` all
        node features are ``1.0``.
    add_self_loop:
        When ``True`` **and** an anchor is given, a self-loop is added on the
        anchor node (matching the original sampling convention).

    Returns
    -------
    Data
        A PyG ``Data`` with ``x``, ``edge_index``, and ``num_nodes`` set.
    """
    # Relabel nodes to contiguous integers so edge_index values are valid.
    mapping = {v: i for i, v in enumerate(g.nodes())}
    g_relabeled = nx.relabel_nodes(g, mapping)
    num_nodes = g_relabeled.number_of_nodes()

    # Resolve anchor in the new labelling.
    anchor_idx: Optional[int] = None
    if node_anchored and anchor is not None:
        anchor_idx = mapping[anchor]

    # --- Build edge_index (both directions for undirected) ----------------
    edges = list(g_relabeled.edges())
    if len(edges) > 0:
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
    else:
        src, dst = [], []

    # Optionally add self-loop on anchor.
    if add_self_loop and anchor_idx is not None:
        src.append(anchor_idx)
        dst.append(anchor_idx)

    if len(src) > 0:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Node features: GLASS labeling (ICLR 2022) -------------------------
    # Channel 0: anchor indicator (1 for anchor, 0 otherwise)
    # Channel 1: inside-subgraph indicator (1 for all nodes in this subgraph)
    x = torch.zeros(num_nodes, 2)
    x[:, 1] = 1.0  # all nodes are inside this subgraph
    if node_anchored and anchor_idx is not None:
        x[anchor_idx, 0] = 1.0

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def batch_nx_graphs(
    graphs: list[nx.Graph],
    anchors: Optional[list[int]] = None,
    node_anchored: bool = True,
    add_self_loop: bool = True,
    device: Optional[torch.device] = None,
) -> Batch:
    """Batch multiple NetworkX graphs into a single PyG :class:`Batch`.

    This is the drop-in replacement for the old DeepSNAP-based
    ``batch_nx_graphs`` in ``common/utils.py``.

    Parameters
    ----------
    graphs:
        List of NetworkX graphs.
    anchors:
        Optional per-graph anchor node ids (in the original labelling of
        each graph).  Must be the same length as *graphs* when provided.
    node_anchored:
        Forwarded to :func:`nx_to_pyg`.
    add_self_loop:
        Forwarded to :func:`nx_to_pyg`.
    device:
        Target device.  When ``None``, :func:`minomaly.utils.device.get_device`
        is called to auto-select.

    Returns
    -------
    Batch
        A batched PyG ``Batch`` object ready for model consumption.
    """
    if device is None:
        from minomaly.utils.device import get_device
        device = get_device()

    data_list: list[Data] = []
    for i, g in enumerate(graphs):
        anchor = anchors[i] if anchors is not None else None
        data_list.append(
            nx_to_pyg(g, anchor=anchor, node_anchored=node_anchored,
                       add_self_loop=add_self_loop)
        )

    batch = Batch.from_data_list(data_list)
    return batch.to(device)
