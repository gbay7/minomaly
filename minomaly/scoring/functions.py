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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
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
        **kwargs,
    ) -> float:
        return (freq + weight) / 2.0


# ── Beam-aware scoring functions ─────────────────────────────────────


@SCORING.register("density_aware")
class DensityAwareScore(ScoringFunction):
    """Score = freq / edge_density.

    Normalizes frequency by edge density so that sparse patterns (grids,
    cycles) with moderate frequency score lower (= more anomalous) than
    dense patterns at the same frequency. Falls back to freq when no
    beam is provided.
    """

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
        **kwargs,
    ) -> float:
        beam = kwargs.get("beam")
        if beam is None:
            return freq
        density = beam.edge_density()
        if density < 1e-6:
            return freq
        return freq / density


@SCORING.register("freq_gradient")
class FreqGradientScore(ScoringFunction):
    """Score = freq + alpha * |df/ds|.

    Combines raw frequency with the magnitude of the frequency gradient.
    Patterns whose frequency drops sharply as they grow (distinctive
    structures) get a bonus (lower score). Falls back to freq when no
    beam is provided.
    """

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
        **kwargs,
    ) -> float:
        beam = kwargs.get("beam")
        if beam is None:
            return freq
        grad = beam.freq_gradient()
        return freq + alpha * grad


@SCORING.register("structural")
class StructuralScore(ScoringFunction):
    """Multi-feature score combining freq, density, and gradient.

    score = alpha * freq_norm + (1-alpha) * density_penalty
    where density_penalty = freq / max(density, eps).

    This separates dense anomalies (low freq, high density → low penalty)
    from sparse anomalies (moderate freq, low density → lower penalty
    than raw freq would suggest). Falls back to freq without a beam.
    """

    def __call__(
        self,
        freq: float,
        weight: float,
        alpha: float = 0.5,
        last_score: float = float("inf"),
        **kwargs,
    ) -> float:
        beam = kwargs.get("beam")
        if beam is None:
            return freq
        density = beam.edge_density()
        grad = beam.freq_gradient()
        density_term = freq / max(density, 0.01)
        grad_term = freq + 0.5 * grad
        return alpha * density_term + (1 - alpha) * grad_term


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
        **kwargs,
    ) -> float:
        return freq + weight * (freq - last_score)
