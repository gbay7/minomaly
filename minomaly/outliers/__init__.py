"""Outlier detectors -- import all to trigger registry registration."""

from minomaly.outliers.base import OutlierDetector
from minomaly.outliers.model_based import ModelBasedDetector
from minomaly.outliers.isolation_forest import IsolationForestDetector
from minomaly.outliers.combined import CombinedDetector

__all__ = [
    "OutlierDetector",
    "ModelBasedDetector",
    "IsolationForestDetector",
    "CombinedDetector",
]
