"""Poincare ball model math primitives.

Implements numerically stable operations on the Poincare ball
B_c^d = {x in R^d : c||x||^2 < 1} with curvature -1/c.  When c = 1 this is
the standard unit Poincare ball.

References
----------
- Ganea et al., "Hyperbolic Neural Networks" (NeurIPS 2018)
- Chami et al., "Hyperbolic Graph Convolutional Neural Networks" (NeurIPS 2019)
"""

from __future__ import annotations

import torch

MIN_NORM: float = 1e-15
BALL_EPS: float = 1e-5


def project(
    x: torch.Tensor, c: torch.Tensor, eps: float = BALL_EPS
) -> torch.Tensor:
    """Project *x* onto the Poincare ball so that ``||x|| < 1/sqrt(c) - eps``.

    Parameters
    ----------
    x:
        Arbitrary-shape tensor whose last dimension is the embedding dimension.
    c:
        Positive curvature scalar tensor.
    eps:
        Small margin to keep points strictly inside the ball boundary.

    Returns
    -------
    torch.Tensor
        Projected tensor with the same shape as *x*.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    max_norm = (1.0 / torch.sqrt(c)) - eps
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def lambda_x(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    r"""Conformal factor :math:`\lambda_x = 2 / (1 - c \|x\|^2)`.

    Parameters
    ----------
    x:
        Point(s) in the Poincare ball.
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Conformal factor with the last dimension kept for broadcasting.
    """
    x_sqnorm = (x * x).sum(dim=-1, keepdim=True).clamp(
        max=1.0 / c.item() - BALL_EPS
    )
    return 2.0 / (1.0 - c * x_sqnorm).clamp(min=MIN_NORM)


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    r"""Mobius addition :math:`x \oplus_c y` in the Poincare ball.

    Parameters
    ----------
    x, y:
        Points in the Poincare ball (broadcastable shapes).
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Result of the Mobius addition, projected back onto the ball.
    """
    x_sqnorm = (x * x).sum(dim=-1, keepdim=True)
    y_sqnorm = (y * y).sum(dim=-1, keepdim=True)
    xy_inner = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy_inner + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
    denom = 1 + 2 * c * xy_inner + c ** 2 * x_sqnorm * y_sqnorm
    return project(num / denom.clamp(min=MIN_NORM), c)


def exp_map_zero(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exponential map at the origin: tangent space -> Poincare ball.

    Parameters
    ----------
    v:
        Tangent vector(s) at the origin.
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Corresponding point(s) on the Poincare ball.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    sqrt_c = torch.sqrt(c)
    return project(torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm), c)


def log_map_zero(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map at the origin: Poincare ball -> tangent space.

    Parameters
    ----------
    y:
        Point(s) on the Poincare ball.
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Tangent vector(s) at the origin corresponding to *y*.
    """
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    sqrt_c = torch.sqrt(c)
    return (
        torch.atanh(sqrt_c * y_norm.clamp(max=1.0 / sqrt_c.item() - BALL_EPS))
        * y
        / (sqrt_c * y_norm)
    )


def exp_map(
    x: torch.Tensor, v: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Exponential map at point *x*: tangent space at *x* -> Poincare ball.

    Parameters
    ----------
    x:
        Base point on the Poincare ball.
    v:
        Tangent vector(s) at *x*.
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Point(s) on the Poincare ball reached by following the geodesic
        from *x* in direction *v*.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=MIN_NORM)
    sqrt_c = torch.sqrt(c)
    lam = lambda_x(x, c)
    second_term = (
        torch.tanh(sqrt_c * lam * v_norm / 2) * v / (sqrt_c * v_norm)
    )
    return project(mobius_add(x, second_term, c), c)


def poincare_distance(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    r"""Geodesic distance between *x* and *y* in the Poincare ball.

    .. math::
        d_c(x, y) = \frac{2}{\sqrt{c}} \operatorname{arctanh}\bigl(
            \sqrt{c}\,\lVert (-x) \oplus_c y \rVert
        \bigr)

    Parameters
    ----------
    x, y:
        Points in the Poincare ball (broadcastable shapes).
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Scalar distance(s); the last dimension is reduced.
    """
    sqrt_c = torch.sqrt(c)
    add_result = mobius_add(-x, y, c)
    dist = add_result.norm(dim=-1).clamp(min=MIN_NORM)
    # Clamp for numerical stability of arctanh
    dist = dist.clamp(max=1.0 / sqrt_c.item() - BALL_EPS)
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * dist)


def poincare_distance_batch(
    x: torch.Tensor, y: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """Pairwise distances between *x* ``(N, D)`` and *y* ``(B, D)`` -> ``(N, B)``.

    Broadcasts *x* and *y* along new dimensions to compute all N * B pairs
    in a single batched call.

    Parameters
    ----------
    x:
        Tensor of shape ``(N, D)``.
    y:
        Tensor of shape ``(B, D)``.
    c:
        Positive curvature scalar tensor.

    Returns
    -------
    torch.Tensor
        Distance matrix of shape ``(N, B)``.
    """
    x_exp = x.unsqueeze(1)  # (N, 1, D)
    y_exp = y.unsqueeze(0)  # (1, B, D)
    return poincare_distance(x_exp, y_exp, c)  # (N, B)
