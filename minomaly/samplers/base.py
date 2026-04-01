"""Base classes for neighborhood sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    from minomaly.data.graph import GraphData


@dataclass
class SamplingResult:
    """Output of a sampling pass.

    All neighborhoods are stored as PyG ``Data`` objects (not NetworkX).
    ``real_anchors`` contains the original node ids in the source graph.
    """

    neighborhoods: list[Data] = field(default_factory=list)
    anchors: list[int] = field(default_factory=list)
    real_anchors: list[int] = field(default_factory=list)
    node_lists: list[list[int]] = field(default_factory=list)  # original node IDs per neighborhood (for contextual embedding)


class Sampler(ABC):
    """ABC for neighborhood sampling strategies.

    Samplers receive :class:`~minomaly.data.graph.GraphData` objects
    (PyG-native) and produce PyG ``Data`` objects for each neighborhood.
    """

    @abstractmethod
    def sample(
        self,
        graphs: list[GraphData],
        node_anchored: bool = True,
        add_self_loop: bool = True,
    ) -> SamplingResult:
        """Sample neighborhoods from *graphs*.

        Args:
            graphs: source graphs (GPU-friendly GraphData).
            node_anchored: if True, record anchor indices and set anchor feature.
            add_self_loop: if True, add a self-loop on the anchor node.
        """
        ...
