"""Ensemble generator that delegates to a weighted mix of sub-generators."""

from __future__ import annotations

import numpy as np
import networkx as nx

from minomaly.generators.base import GraphGenerator
from minomaly.generators.erdos_renyi import ERGenerator
from minomaly.generators.watts_strogatz import WSGenerator
from minomaly.generators.barabasi_albert import BAGenerator
from minomaly.generators.powerlaw_cluster import PowerLawClusterGenerator
from minomaly.generators.dense_er import DenseERGenerator
from minomaly.generators.structural import (
    BiCliqueGenerator,
    GridGenerator,
    CycleGenerator,
    NearCliqueGenerator,
    WheelGenerator,
    HypercubeGenerator,
)
from minomaly.registry import GENERATORS


@GENERATORS.register("ensemble")
class EnsembleGenerator(GraphGenerator):
    """Randomly select a sub-generator and delegate graph generation.

    Parameters
    ----------
    generators:
        Concrete :class:`GraphGenerator` instances to choose from.
    gen_prob:
        Selection probabilities (uniform if *None*).
    """

    def __init__(
        self,
        generators: list[GraphGenerator],
        gen_prob: list[float] | None = None,
    ) -> None:
        # EnsembleGenerator does not need its own sizes/size_prob --
        # each sub-generator carries its own.  We initialise the ABC
        # with a dummy value so that _get_size is never called directly
        # on this instance.
        if generators:
            super().__init__(generators[0].sizes, size_prob=generators[0].size_prob)
        else:
            super().__init__(np.array([1]))

        self.generators = generators
        if gen_prob is not None:
            prob = np.asarray(gen_prob, dtype=np.float64)
            self.gen_prob: np.ndarray = prob / prob.sum()
        else:
            self.gen_prob = np.ones(len(generators)) / len(generators)

    def generate(self, size: int | None = None) -> nx.Graph:
        idx = int(np.random.choice(len(self.generators), p=self.gen_prob))
        return self.generators[idx].generate(size=size)


def build_default_ensemble(
    sizes: np.ndarray,
    include_dense: bool = True,
    include_structural: bool = False,
) -> EnsembleGenerator:
    """Create the default ensemble of ER + WS + BA + PowerLaw (+ DenseER) generators.

    When *include_structural* is True, adds biclique, grid, cycle, and
    near-clique generators so the model learns these topologies.
    """
    generators: list[GraphGenerator] = [
        ERGenerator(sizes),
        WSGenerator(sizes),
        BAGenerator(sizes),
        PowerLawClusterGenerator(sizes),
    ]
    if include_dense:
        generators.append(DenseERGenerator(sizes))
    if include_structural:
        generators.extend([
            BiCliqueGenerator(sizes),
            GridGenerator(sizes),
            CycleGenerator(sizes),
            NearCliqueGenerator(sizes),
            WheelGenerator(sizes),
            HypercubeGenerator(sizes),
        ])
    return EnsembleGenerator(generators)
