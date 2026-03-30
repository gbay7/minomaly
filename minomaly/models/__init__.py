"""Models sub-package — import all to trigger registry registration."""

from minomaly.models.skip_last_gnn import SkipLastGNN
from minomaly.models.order_embedder import OrderEmbedder
from minomaly.models.hyperbolic_gnn import HyperbolicGNN
from minomaly.models.poincare_embedder import PoincareEmbedder
from minomaly.models.gatv2_encoder import GATv2Encoder
from minomaly.models.contrastive_embedder import ContrastiveEmbedder
from minomaly.models.hybrid_embedder import HybridEmbedder

__all__ = [
    "SkipLastGNN", "OrderEmbedder", "HyperbolicGNN",
    "PoincareEmbedder", "GATv2Encoder", "ContrastiveEmbedder",
    "HybridEmbedder",
]
