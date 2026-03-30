"""Abstract base class for random graph generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np


class GraphGenerator(ABC):
    """ABC for random graph generators."""

    def __init__(
        self,
        sizes: np.ndarray,
        size_prob: np.ndarray | None = None,
    ) -> None:
        self.sizes = np.asarray(sizes)
        if size_prob is not None:
            size_prob = np.asarray(size_prob, dtype=np.float64)
            self.size_prob: np.ndarray = size_prob / size_prob.sum()
        else:
            self.size_prob = np.ones(len(self.sizes)) / len(self.sizes)

    def _get_size(self, size: int | None = None) -> int:
        if size is not None:
            return size
        return int(np.random.choice(self.sizes, p=self.size_prob))

    @abstractmethod
    def generate(self, size: int | None = None) -> nx.Graph: ...
