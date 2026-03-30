"""Deterministic seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_deterministic(seed: int = 42) -> None:
    """Seed all relevant random number generators and enable deterministic mode.

    Sets seeds for :mod:`random`, :mod:`numpy`, and :mod:`torch` (CPU and
    CUDA).  Also configures cuDNN for deterministic behaviour and enables
    :func:`torch.use_deterministic_algorithms`.

    Parameters
    ----------
    seed:
        The integer seed to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # Older PyTorch versions may not have this function.
        pass
