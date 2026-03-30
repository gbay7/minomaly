"""Tree (random-walk) neighborhood sampler — PyG-native."""

from __future__ import annotations

import random

import numpy as np
import scipy.stats as stats
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from minomaly.data.graph import GraphData
from minomaly.registry import SAMPLERS
from minomaly.samplers.base import Sampler, SamplingResult


def sample_neigh_pyg(
    graphs: list[GraphData],
    size: int,
) -> tuple[GraphData, list[int]]:
    """Sample a connected neighborhood of *size* nodes via random frontier walk.

    Operates on :class:`GraphData` using tensor-based neighbor lookups.
    Returns (graph, node_list) where node_list[0] is the anchor.
    """
    ps = np.array([g.num_nodes for g in graphs], dtype=np.float64)
    ps /= ps.sum()
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        start_node = random.randint(0, graph.num_nodes - 1)
        neigh = [start_node]
        neigh_set = {start_node}
        frontier_set = set(graph.neighbors(start_node).tolist()) - neigh_set
        frontier = list(frontier_set)
        while len(neigh) < size and frontier:
            new_node = random.choice(frontier)
            neigh.append(new_node)
            neigh_set.add(new_node)
            new_neighbors = set(graph.neighbors(new_node).tolist())
            frontier_set = (frontier_set | new_neighbors) - neigh_set
            frontier = list(frontier_set)
        if len(neigh) == size:
            return graph, neigh


def _make_subgraph_data(
    graph: GraphData,
    neigh_nodes: list[int],
    node_anchored: bool,
    add_self_loop: bool,
) -> tuple[Data, int]:
    """Extract subgraph as a PyG Data. Anchor is neigh_nodes[0] → relabeled to 0."""
    anchor = neigh_nodes[0]
    nodes_t = torch.tensor(neigh_nodes, dtype=torch.long, device=graph.device)
    view = graph.subgraph(nodes_t)
    data = view.to_pyg(
        anchor_global=anchor,
        node_anchored=node_anchored,
        add_self_loop=add_self_loop,
    )
    return data, anchor


@SAMPLERS.register("tree")
class TreeSampler(Sampler):
    """Random-walk (tree) neighborhood sampler."""

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
        result = SamplingResult()
        for _ in tqdm(range(self.n_neighborhoods), desc="TreeSampler"):
            size = random.randint(self.min_size, self.max_size)
            graph, neigh_nodes = sample_neigh_pyg(graphs, size)
            data, real_anchor = _make_subgraph_data(
                graph, neigh_nodes, node_anchored, add_self_loop,
            )
            result.neighborhoods.append(data)
            result.real_anchors.append(real_anchor)
            if node_anchored:
                result.anchors.append(0)  # anchor is always relabeled to 0
        return result
