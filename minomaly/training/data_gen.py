"""Training pair generator for order-embedding GNN training.

Generates positive and negative graph pairs for the order-embedding loss.
Positive pairs: (target, subgraph_of_target) -- true subgraph relation.
Negative pairs: (graph_a, random_graph_b) -- no guaranteed relation.
Hard negatives: take a real subgraph and add random edges to break the relation.
"""

from __future__ import annotations

import random
from typing import Optional

import networkx as nx
import torch
from torch_geometric.data import Batch

from minomaly.data.convert import batch_nx_graphs
from minomaly.generators.base import GraphGenerator
from minomaly.utils.device import get_device


class TrainingPairGenerator:
    """Generate pos/neg graph pairs for order embedding training.

    Positive: (target, subgraph_of_target) -- true subgraph relation.
    Negative: (graph_a, random_graph_b) -- no guaranteed relation.
    Hard negatives: take a real subgraph and add random edges to break relation.

    Parameters
    ----------
    generator:
        A :class:`GraphGenerator` (or :class:`EnsembleGenerator`) that produces
        random connected NetworkX graphs.
    min_size:
        Minimum graph size (number of nodes) for generated graphs.
    max_size:
        Maximum graph size (number of nodes) for generated graphs.
    node_anchored:
        Whether to use node-anchored subgraph extraction (anchor indicator
        feature on node 0).
    hard_neg_ratio:
        Fraction of negative pairs that should be hard negatives (subgraphs
        with extra random edges added).
    """

    def __init__(
        self,
        generator: GraphGenerator,
        min_size: int = 5,
        max_size: int = 29,
        node_anchored: bool = True,
        hard_neg_ratio: float = 0.5,
    ) -> None:
        self.generator = generator
        self.min_size = min_size
        self.max_size = max_size
        self.node_anchored = node_anchored
        self.hard_neg_ratio = hard_neg_ratio

    def generate_batch(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> tuple[Batch, Batch, Batch, Batch]:
        """Generate a batch of positive and negative graph pairs.

        Parameters
        ----------
        batch_size:
            Number of positive pairs (and negative pairs) to generate.
            The total number of pairs is ``2 * batch_size``.
        device:
            Target device for the PyG Batch objects.  When ``None``,
            :func:`get_device` is used.

        Returns
        -------
        tuple[Batch, Batch, Batch, Batch]
            ``(pos_target, pos_query, neg_target, neg_query)`` as PyG Batches.
        """
        if device is None:
            device = get_device()

        pos_targets: list[nx.Graph] = []
        pos_queries: list[nx.Graph] = []
        pos_target_anchors: list[int] = []
        pos_query_anchors: list[int] = []

        neg_targets: list[nx.Graph] = []
        neg_queries: list[nx.Graph] = []
        neg_target_anchors: list[int] = []
        neg_query_anchors: list[int] = []

        for _ in range(batch_size):
            # Positive pair
            t, q, a_t, a_q = self._positive_pair()
            pos_targets.append(t)
            pos_queries.append(q)
            pos_target_anchors.append(a_t)
            pos_query_anchors.append(a_q)

            # Negative pair
            hard = random.random() < self.hard_neg_ratio
            t, q, a_t, a_q = self._negative_pair(hard=hard)
            neg_targets.append(t)
            neg_queries.append(q)
            neg_target_anchors.append(a_t)
            neg_query_anchors.append(a_q)

        # Convert NX graphs to PyG Batches via batch_nx_graphs
        kwargs: dict = dict(node_anchored=self.node_anchored, device=device)

        pos_t_batch = batch_nx_graphs(
            pos_targets,
            anchors=pos_target_anchors if self.node_anchored else None,
            **kwargs,
        )
        pos_q_batch = batch_nx_graphs(
            pos_queries,
            anchors=pos_query_anchors if self.node_anchored else None,
            **kwargs,
        )
        neg_t_batch = batch_nx_graphs(
            neg_targets,
            anchors=neg_target_anchors if self.node_anchored else None,
            **kwargs,
        )
        neg_q_batch = batch_nx_graphs(
            neg_queries,
            anchors=neg_query_anchors if self.node_anchored else None,
            **kwargs,
        )

        return pos_t_batch, pos_q_batch, neg_t_batch, neg_q_batch

    def _positive_pair(self) -> tuple[nx.Graph, nx.Graph, int, int]:
        """Sample a graph and a true subgraph of it.

        Uses the frontier-walk pattern from the original code's
        ``sample_neigh``: pick a random start node, grow by randomly picking
        frontier neighbours until the desired subgraph size is reached.

        Returns
        -------
        tuple[nx.Graph, nx.Graph, int, int]
            ``(target_graph, subgraph, anchor_in_target, anchor_in_subgraph)``
        """
        target_size = random.randint(self.min_size + 1, self.max_size)
        target = self._gen_connected(target_size)

        nodes = list(target.nodes())
        anchor = random.choice(nodes)

        # Subgraph size: at least 1, strictly smaller than target
        max_sub = max(len(target) - 1, 1)
        sub_size = random.randint(1, max_sub)

        # Random frontier walk to extract a connected subgraph (sample_neigh
        # pattern from code-original/common/utils.py)
        neigh = [anchor]
        frontier = list(set(target.neighbors(anchor)) - set(neigh))
        visited = {anchor}

        while len(neigh) < sub_size and frontier:
            new_node = random.choice(frontier)
            neigh.append(new_node)
            visited.add(new_node)
            frontier += [
                n for n in target.neighbors(new_node) if n not in visited
            ]
            frontier = [x for x in frontier if x not in visited]

        subgraph = target.subgraph(neigh).copy()

        return target, subgraph, anchor, anchor

    def _negative_pair(
        self, hard: bool = False
    ) -> tuple[nx.Graph, nx.Graph, int, int]:
        """Generate a negative pair.

        Parameters
        ----------
        hard:
            If ``True``, generate a hard negative by taking a real subgraph
            and adding random edges to break the subgraph relation.
            If ``False``, generate two independent random graphs.

        Returns
        -------
        tuple[nx.Graph, nx.Graph, int, int]
            ``(graph_a, graph_b, anchor_a, anchor_b)``
        """
        if hard:
            # Hard negative: take a real subgraph and corrupt it by adding
            # random edges to break containment
            target, subgraph, anchor_t, anchor_q = self._positive_pair()

            sub_nodes = list(subgraph.nodes())
            if len(sub_nodes) >= 2:
                non_edges = list(nx.non_edges(subgraph))
                if non_edges:
                    n_add = random.randint(1, min(len(non_edges), 5))
                    subgraph = subgraph.copy()
                    for u, v in random.sample(non_edges, n_add):
                        subgraph.add_edge(u, v)

            return target, subgraph, anchor_t, anchor_q
        else:
            # Easy negative: two independent random graphs
            size_a = random.randint(self.min_size, self.max_size)
            size_b = random.randint(self.min_size, self.max_size)
            graph_a = self._gen_connected(size_a)
            graph_b = self._gen_connected(size_b)

            anchor_a = random.choice(list(graph_a.nodes()))
            anchor_b = random.choice(list(graph_b.nodes()))

            return graph_a, graph_b, anchor_a, anchor_b

    def _gen_connected(self, size: int) -> nx.Graph:
        """Generate a connected graph, retrying until successful.

        Parameters
        ----------
        size:
            Number of nodes in the target graph.

        Returns
        -------
        nx.Graph
            A connected NetworkX graph with ``size`` nodes.
        """
        for _ in range(100):
            g = self.generator.generate(size=size)
            if nx.is_connected(g):
                return g
        # Fallback: force connectivity by bridging disconnected components
        g = self.generator.generate(size=size)
        if not nx.is_connected(g):
            components = list(nx.connected_components(g))
            for i in range(1, len(components)):
                u = random.choice(list(components[i - 1]))
                v = random.choice(list(components[i]))
                g.add_edge(u, v)
        return g
