"""Abstract base classes for scoring functions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ScoringFunction(ABC):
    """Score a beam during search (lower is more anomalous)."""

    @abstractmethod
    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float: ...


class VerificationScoringFunction(ABC):
    """Score a beam for verification (incorporates score dynamics)."""

    @abstractmethod
    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float: ...
