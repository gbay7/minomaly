"""Abstract base classes for scoring functions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ScoringFunction(ABC):
    """Score a beam during search (lower is more anomalous).

    Subclasses that need beam-level structural features (edge density,
    degree distribution, etc.) can accept ``beam=`` via **kwargs.
    When called without a beam (e.g. for max_strength threshold), the
    scorer should fall back to frequency-only behavior.
    """

    @abstractmethod
    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
        **kwargs,
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
        **kwargs,
    ) -> float: ...
