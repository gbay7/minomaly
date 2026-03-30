"""Barabasi-Albert preferential attachment graph generator."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from minomaly.generators.base import GraphGenerator
from minomaly.registry import GENERATORS

logger = logging.getLogger(__name__)


@GENERATORS.register("barabasi_albert")
class BAGenerator(GraphGenerator):
    """Generate connected dual Barabasi-Albert graphs.

    Uses ``nx.dual_barabasi_albert_graph`` with random *m*, *p*, and *q*
    values.  Regenerates until the result is connected.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        max_p: float = 0.2,
        max_q: float = 0.2,
        size_prob: np.ndarray | None = None,
    ) -> None:
        super().__init__(sizes, size_prob=size_prob)
        self.max_p = max_p
        self.max_q = max_q

    def generate(self, size: int | None = None) -> nx.Graph:
        num_nodes = self._get_size(size)
        max_m = max(int(2 * np.log2(num_nodes)), 2)

        for _ in range(100):
            m1 = int(np.random.randint(1, max_m)) + 1
            m2 = int(np.random.randint(1, max(m1, 2)))
            p = float(np.minimum(np.random.exponential(20), self.max_p))
            # Clamp m1, m2 to valid range
            m1 = min(m1, num_nodes - 1)
            m2 = min(m2, num_nodes - 1)
            m1 = max(m1, 1)
            m2 = max(m2, 1)
            try:
                graph = nx.dual_barabasi_albert_graph(num_nodes, m1, m2, p)
                if nx.is_connected(graph):
                    return graph
            except nx.NetworkXError:
                continue

        # Fallback to simple BA
        graph = nx.barabasi_albert_graph(num_nodes, min(2, num_nodes - 1))
        return graph

        logger.debug(
            "Generated %d-node dual B-A graph with max m: %d",
            num_nodes,
            max_m,
        )
        return graph
