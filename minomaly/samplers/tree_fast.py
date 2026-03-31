"""Fast tree sampler — uses NetworkX for neighbor iteration.

10-50x faster than the PyG-native tree sampler because NX dict-of-dict
adjacency provides O(1) Python-level neighbor lookups without tensor
conversion overhead.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from tqdm import tqdm

from minomaly.data.graph import GraphData
from minomaly.registry import SAMPLERS
from minomaly.samplers.base import Sampler, SamplingResult


def _sample_neigh_nx(
    nx_graphs: list[nx.Graph],
    size: int,
) -> tuple[int, nx.Graph, list[int]]:
    """Sample a connected neighborhood via random frontier walk on NX.

    Same algorithm as SPMiner's original sample_neigh — fast because NX
    uses dict-of-dict adjacency.
    """
    ps = np.array([len(g) for g in nx_graphs], dtype=np.float32)
    ps /= ps.sum()
    dist = stats.rv_discrete(values=(np.arange(len(nx_graphs)), ps))
    while True:
        idx = dist.rvs()
        graph = nx_graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = {start_node}
        while len(neigh) < size and frontier:
            new_node = random.choice(frontier)
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return idx, graph, neigh


def _neigh_to_pyg(
    graph: nx.Graph,
    neigh_nodes: list[int],
    node_anchored: bool,
    add_self_loop: bool,
) -> Data:
    """Convert NX neighborhood to relabeled PyG Data."""
    anchor = neigh_nodes[0]
    subg = graph.subgraph(neigh_nodes)

    mapping = {anchor: 0}
    mapping.update({n: i + 1 for i, n in enumerate(set(subg.nodes) - {anchor})})
    num_nodes = len(mapping)

    edges = list(subg.edges())
    if edges:
        src = [mapping[u] for u, v in edges] + [mapping[v] for u, v in edges]
        dst = [mapping[v] for u, v in edges] + [mapping[u] for u, v in edges]
    else:
        src, dst = [], []

    if add_self_loop:
        src.append(0)
        dst.append(0)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # GLASS labeling: [anchor_indicator, inside_subgraph]
    x = torch.zeros(num_nodes, 2)
    x[:, 1] = 1.0
    if node_anchored:
        x[0, 0] = 1.0

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


@SAMPLERS.register("tree_fast")
class TreeFastSampler(Sampler):
    """Fast tree sampler using NX for neighbor iteration.

    Converts GraphData to NX once, then samples using Python-native
    dict lookups. 10-50x faster than the PyG-native TreeSampler.
    """

    def __init__(
        self,
        n_neighborhoods: int = 10_000,
        min_size: int = 1,
        max_size: int = 30,
    ) -> None:
        self.n_neighborhoods = n_neighborhoods
        self.min_size = min_size
        self.max_size = max_size

    def sample(
        self,
        graphs: list[GraphData],
        node_anchored: bool = True,
        add_self_loop: bool = True,
    ) -> SamplingResult:
        nx_graphs = [g.to_nx() for g in graphs]

        result = SamplingResult()
        for _ in tqdm(range(self.n_neighborhoods), desc="TreeFastSampler"):
            size = random.randint(self.min_size, self.max_size)
            idx, graph, neigh_nodes = _sample_neigh_nx(nx_graphs, size)
            data = _neigh_to_pyg(graph, neigh_nodes, node_anchored, add_self_loop)
            result.neighborhoods.append(data)
            result.real_anchors.append(neigh_nodes[0])
            if node_anchored:
                result.anchors.append(0)
        return result
