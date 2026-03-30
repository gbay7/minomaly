"""Minomaly — Unsupervised Graph Anomaly Detection Framework."""

__version__ = "0.2.0"

# Import all subpackages to trigger registry decorators
import minomaly.generators  # noqa: F401
import minomaly.samplers  # noqa: F401
import minomaly.scoring  # noqa: F401
import minomaly.models  # noqa: F401
import minomaly.outliers  # noqa: F401
import minomaly.search  # noqa: F401

# Public API
from minomaly.config.schema import MinomalyConfig
from minomaly.config.loader import load_config
from minomaly.pipeline import MinomalyPipeline

__all__ = ["MinomalyConfig", "load_config", "MinomalyPipeline", "__version__"]
