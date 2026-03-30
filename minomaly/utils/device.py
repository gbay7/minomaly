"""Device resolution and caching utilities."""

from __future__ import annotations

import torch

_device_cache: torch.device | None = None


def get_device() -> torch.device:
    """Return a cached :class:`torch.device`, selecting CUDA when available.

    The result is computed once and cached for subsequent calls.
    """
    global _device_cache
    if _device_cache is None:
        _device_cache = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return _device_cache


def resolve_device(config_device: str) -> torch.device:
    """Resolve a device string from configuration into a :class:`torch.device`.

    Parameters
    ----------
    config_device:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
        When ``"auto"`` is given, :func:`get_device` is used.
    """
    if config_device == "auto":
        return get_device()
    return torch.device(config_device)
