"""Erdos-Renyi random graph generator."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from minomaly.generators.base import GraphGenerator
from minomaly.registry import GENERATORS

logger = logging.getLogger(__name__)


@GENERATORS.register("erdos_renyi")
class ERGenerator(GraphGenerator):
    """Generate connected Erdos-Renyi random graphs.

    Edge probability is sampled from a Beta distribution whose mean is
    ``log2(n) / n``, controlled by *p_alpha*.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        p_alpha: float = 1.3,
        size_prob: np.ndarray | None = None,
    ) -> None:
        super().__init__(sizes, size_prob=size_prob)
        self.p_alpha = p_alpha

    def generate(self, size: int | None = None) -> nx.Graph:
        num_nodes = self._get_size(size)
        alpha = self.p_alpha
        mean = np.log2(num_nodes) / num_nodes
        beta = alpha / mean - alpha

        p = np.random.beta(alpha, beta)
        graph = nx.gnp_random_graph(num_nodes, p)

        while not nx.is_connected(graph):
            p = np.random.beta(alpha, beta)
            graph = nx.gnp_random_graph(num_nodes, p)

        logger.debug(
            "Generated %d-node E-R graph with average p: %.4f",
            num_nodes,
            mean,
        )
        return graph
