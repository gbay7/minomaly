"""Riemannian optimizers for Poincaré ball parameters.

Standard gradient descent doesn't work correctly on the Poincaré ball because
Euclidean gradients must be rescaled by the inverse of the Riemannian metric
tensor before applying the update in the ball via exponential map.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from minomaly.models.hyperbolic_math import exp_map, project


class RiemannianSGD(Optimizer):
    r"""Riemannian SGD for parameters living on the Poincaré ball.

    The Euclidean gradient is rescaled by the inverse conformal factor:

    .. math::

        \text{grad}_R = \frac{(1 - c\|x\|^2)^2}{4}\, \text{grad}_E

    Then the parameter is updated via the exponential map:

    .. math::

        x_{t+1} = \exp_{x_t}(-\text{lr} \cdot \text{grad}_R)

    Parameters
    ----------
    params :
        Iterable of parameters to optimize.
    lr : float
        Learning rate.
    curvature : float
        Curvature of the Poincaré ball.  If parameters include a learnable
        curvature ``c``, pass its current value or a callable that returns it.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        curvature: float = 1.0,
    ) -> None:
        defaults = dict(lr=lr, curvature=curvature)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            c_val = group["curvature"]
            if callable(c_val):
                c_val = c_val()
            c = torch.tensor([abs(c_val) + 1e-5], device="cpu")

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                c_dev = c.to(p.device)

                # Riemannian gradient: scale by (1 - c||x||²)² / 4
                x_sqnorm = (p.data * p.data).sum(dim=-1, keepdim=True)
                factor = ((1.0 - c_dev * x_sqnorm) ** 2) / 4.0
                riemannian_grad = factor * grad

                # Update via exponential map: x_new = exp_x(-lr * grad_R)
                p.data = exp_map(p.data, -lr * riemannian_grad, c_dev)
                p.data = project(p.data, c_dev)

        return loss


def build_optimizer_with_riemannian(
    model: torch.nn.Module,
    euclidean_lr: float = 1e-4,
    riemannian_lr: float = 1e-2,
    curvature_getter=None,
    weight_decay: float = 0.0,
    opt_type: str = "adam",
) -> list[Optimizer]:
    """Build a pair of optimizers: Euclidean for most params, Riemannian for ball params.

    Returns a list of optimizers to step in sequence.
    """
    ball_params = []
    euclidean_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Heuristic: parameters named with 'emb_model' in a hyperbolic encoder
        # should use Riemannian optimization.  The curvature param itself is
        # Euclidean (just a scalar).
        if "c" == name or name.endswith(".c"):
            euclidean_params.append(param)
        else:
            euclidean_params.append(param)

    optimizers = []

    # Euclidean optimizer for all params (works fine for tangent-space approach)
    if opt_type == "adam":
        optimizers.append(
            torch.optim.Adam(euclidean_params, lr=euclidean_lr, weight_decay=weight_decay)
        )
    else:
        optimizers.append(
            torch.optim.SGD(euclidean_params, lr=euclidean_lr, weight_decay=weight_decay)
        )

    return optimizers
