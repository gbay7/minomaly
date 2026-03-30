"""Training sub-package for order-embedding GNN."""

from minomaly.training.losses import order_embedding_loss
from minomaly.training.data_gen import TrainingPairGenerator
from minomaly.training.trainer import OrderEmbeddingTrainer

__all__ = [
    "order_embedding_loss",
    "TrainingPairGenerator",
    "OrderEmbeddingTrainer",
]
