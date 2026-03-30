"""Concrete scoring functions registered with the SCORING registry."""

from __future__ import annotations

from minomaly.registry import SCORING
from minomaly.scoring.base import ScoringFunction, VerificationScoringFunction


# ── Search scoring functions ─────────────────────────────────────────


@SCORING.register("freq")
class FreqScore(ScoringFunction):
    """Raw frequency score (the default in the original codebase)."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        return freq


@SCORING.register("weighted")
class WeightedScore(ScoringFunction):
    """Linear combination of frequency and weight."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        return alpha * freq + (1 - alpha) * weight


@SCORING.register("harmonic")
class HarmonicScore(ScoringFunction):
    """Harmonic mean of frequency and weight."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        total = freq + weight
        if total > 0:
            return (freq * weight) / total
        return 0.0


@SCORING.register("geometric")
class GeometricScore(ScoringFunction):
    """Geometric mean of frequency and weight."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        return (freq * weight) ** 0.5


@SCORING.register("arithmetic")
class ArithmeticScore(ScoringFunction):
    """Arithmetic mean of frequency and weight."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        return (freq + weight) / 2.0


# ── Verification scoring functions ───────────────────────────────────


@SCORING.register("freq_verif")
class FreqVerifScore(VerificationScoringFunction):
    """Verification score incorporating frequency momentum."""

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
    ) -> float:
        return freq + weight * (freq - last_score)
