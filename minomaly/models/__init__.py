"""Models sub-package — import all to trigger registry registration."""

from minomaly.models.skip_last_gnn import SkipLastGNN
from minomaly.models.order_embedder import OrderEmbedder

__all__ = ["SkipLastGNN", "OrderEmbedder"]
