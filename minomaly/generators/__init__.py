"""Graph generators -- import all to trigger registry registration."""

from minomaly.generators.erdos_renyi import ERGenerator
from minomaly.generators.watts_strogatz import WSGenerator
from minomaly.generators.barabasi_albert import BAGenerator
from minomaly.generators.powerlaw_cluster import PowerLawClusterGenerator
from minomaly.generators.ensemble import EnsembleGenerator

__all__ = [
    "ERGenerator",
    "WSGenerator",
    "BAGenerator",
    "PowerLawClusterGenerator",
    "EnsembleGenerator",
]
