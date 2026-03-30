"""Watts-Strogatz small-world graph generator."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from minomaly.generators.base import GraphGenerator
from minomaly.registry import GENERATORS

logger = logging.getLogger(__name__)


@GENERATORS.register("watts_strogatz")
class WSGenerator(GraphGenerator):
    """Generate connected Watts-Strogatz small-world graphs.

    Density (nearest-neighbour count *k*) is sampled from a Beta distribution
    with mean ``log2(n) / n``, scaled by *n*.  Rewiring probability is sampled
    from ``Beta(rewire_alpha, rewire_beta)``.
    """

    def __init__(
        self,
        sizes: np.ndarray,
        density_alpha: float = 1.3,
        rewire_alpha: float = 2.0,
        rewire_beta: float = 2.0,
        size_prob: np.ndarray | None = None,
    ) -> None:
        super().__init__(sizes, size_prob=size_prob)
        self.density_alpha = density_alpha
        self.rewire_alpha = rewire_alpha
        self.rewire_beta = rewire_beta

    def generate(self, size: int | None = None) -> nx.Graph:
        num_nodes = self._get_size(size)

        density_alpha = self.density_alpha
        density_mean = np.log2(num_nodes) / num_nodes
        density_beta = density_alpha / density_mean - density_alpha

        rewire_alpha = self.rewire_alpha
        rewire_beta = self.rewire_beta

        while True:
            k = int(np.random.beta(density_alpha, density_beta) * num_nodes)
            k = max(k, 2)
            p = np.random.beta(rewire_alpha, rewire_beta)
            try:
                graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
                break
            except Exception:
                pass

        logger.debug(
            "Generated %d-node W-S graph with average density: %.4f",
            num_nodes,
            density_mean,
        )
        return graph
