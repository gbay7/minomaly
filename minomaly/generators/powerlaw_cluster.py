"""Power-law cluster graph generator."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from minomaly.generators.base import GraphGenerator
from minomaly.registry import GENERATORS

logger = logging.getLogger(__name__)


@GENERATORS.register("powerlaw_cluster")
class PowerLawClusterGenerator(GraphGenerator):
    """Generate connected power-law cluster graphs.

    Uses ``nx.powerlaw_cluster_graph`` with random attachment count *m* and
    triangle probability *p*.  Regenerates until the result is connected.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        max_p: float = 0.5,
        size_prob: np.ndarray | None = None,
    ) -> None:
        super().__init__(sizes, size_prob=size_prob)
        self.max_p = max_p

    def generate(self, size: int | None = None) -> nx.Graph:
        num_nodes = self._get_size(size)
        max_m = int(2 * np.log2(num_nodes))
        max_m = max(max_m, 1)

        m = np.random.choice(max_m) + 1
        p = np.random.uniform(high=self.max_p)

        graph = nx.powerlaw_cluster_graph(num_nodes, m, p)
        while not nx.is_connected(graph):
            m = np.random.choice(max_m) + 1
            p = np.random.uniform(high=self.max_p)
            graph = nx.powerlaw_cluster_graph(num_nodes, m, p)

        logger.debug(
            "Generated %d-node powerlaw cluster graph with max m: %d",
            num_nodes,
            max_m,
        )
        return graph
