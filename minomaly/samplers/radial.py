"""Radial (BFS) neighborhood sampler — PyG-native."""

from __future__ import annotations

import random
from collections import deque

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from minomaly.data.graph import GraphData
from minomaly.registry import SAMPLERS
from minomaly.samplers.base import Sampler, SamplingResult


def _bfs_neighborhood(graph: GraphData, node: int, radius: int) -> list[int]:
    """BFS up to *radius* hops from *node*, returning all reached node ids."""
    visited = {node}
    queue = deque([(node, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= radius:
            continue
        for nb in graph.neighbors(current).tolist():
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, depth + 1))
    return list(visited)


def _connected_component_containing(
    edge_index: torch.Tensor, num_nodes: int, target: int,
) -> set[int]:
    """Find the connected component containing *target* via BFS on edge_index."""
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].append(v)
        adj[v].append(u)
    visited = {target}
    queue = deque([target])
    while queue:
        n = queue.popleft()
        for nb in adj[n]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return visited


@SAMPLERS.register("radial")
class RadialSampler(Sampler):
    """BFS-based radial neighborhood sampler (PyG-native).

    For each specified node, extracts all nodes reachable within *radius* hops,
    optionally subsamples, and keeps only the connected component containing
    the anchor.
    """

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
        result = SamplingResult()
        for i, graph in enumerate(graphs):
            node_list = (
                list(range(graph.num_nodes))
                if self.nodes is None
                else self.nodes[i]
            )
            for node in tqdm(node_list, desc=f"RadialSampler graph {i}"):
                neigh = _bfs_neighborhood(graph, node, self.radius)

                if self.subsample_size > 0:
                    others = [n for n in neigh if n != node]
                    k = min(len(others), self.subsample_size)
                    neigh = [node] + random.sample(others, k)

                if len(neigh) <= 1:
                    continue

                nodes_t = torch.tensor(neigh, dtype=torch.long, device=graph.device)
                view = graph.subgraph(nodes_t)

                # Keep only the connected component containing the anchor
                if self.subsample_size > 0:
                    anchor_local = view.anchor_local(node)
                    comp = _connected_component_containing(
                        view.edge_index, view.num_nodes, anchor_local,
                    )
                    if len(comp) < view.num_nodes:
                        # Re-extract with only the component nodes
                        comp_global = [neigh[l] for l in sorted(comp)]
                        nodes_t = torch.tensor(
                            comp_global, dtype=torch.long, device=graph.device,
                        )
                        view = graph.subgraph(nodes_t)

                data = view.to_pyg(
                    anchor_global=node,
                    node_anchored=node_anchored,
                    add_self_loop=add_self_loop,
                )
                result.neighborhoods.append(data)
                result.real_anchors.append(node)
                if node_anchored:
                    result.anchors.append(0)
        return result
