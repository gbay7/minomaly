"""Fast radial (BFS) sampler — uses NetworkX for BFS.

Faster than the PyG-native radial sampler because NX BFS and subgraph
operations use Python dict adjacency without tensor overhead.
"""

from __future__ import annotations

import random

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from tqdm import tqdm

from minomaly.data.graph import GraphData
from minomaly.registry import SAMPLERS
from minomaly.samplers.base import Sampler, SamplingResult


def _neigh_to_pyg(
    graph: nx.Graph,
    neigh_nodes: list[int],
    anchor: int,
    node_anchored: bool,
    add_self_loop: bool,
) -> Data:
    """Convert NX neighborhood to relabeled PyG Data."""
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

    x = torch.zeros(num_nodes, 2)
    x[:, 1] = 1.0
    if node_anchored:
        x[0, 0] = 1.0

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


@SAMPLERS.register("radial_fast")
class RadialFastSampler(Sampler):
    """Fast BFS radial sampler using NX for neighbor iteration."""

    def __init__(
        self,
        radius: int = 2,
        subsample_size: int = 0,
        nodes: list[list[int]] | None = None,
    ) -> None:
        self.radius = radius
        self.subsample_size = subsample_size
        self.nodes = nodes

    def sample(
        self,
        graphs: list[GraphData],
        node_anchored: bool = True,
        add_self_loop: bool = True,
    ) -> SamplingResult:
        nx_graphs = [g.to_nx() for g in graphs]
        result = SamplingResult()

        for i, nx_graph in enumerate(nx_graphs):
            node_list = (
                list(nx_graph.nodes)
                if self.nodes is None
                else self.nodes[i]
            )
            for node in tqdm(node_list, desc=f"RadialFastSampler graph {i}"):
                neigh = list(
                    nx.single_source_shortest_path_length(
                        nx_graph, node, cutoff=self.radius
                    ).keys()
                )
                if self.subsample_size > 0:
                    others = [n for n in neigh if n != node]
                    k = min(len(others), self.subsample_size)
                    neigh = [node] + random.sample(others, k)

                if len(neigh) <= 1:
                    continue

                subg = nx_graph.subgraph(neigh)

                if self.subsample_size > 0:
                    for comp in nx.connected_components(subg):
                        if node in comp:
                            neigh = list(comp)
                            break

                if len(neigh) <= 1:
                    continue

                data = _neigh_to_pyg(nx_graph, neigh, node, node_anchored, add_self_loop)
                result.neighborhoods.append(data)
                result.real_anchors.append(node)
                if node_anchored:
                    result.anchors.append(0)

        return result
