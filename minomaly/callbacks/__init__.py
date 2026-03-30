"""Callback subsystem for Minomaly.

Importing this package triggers registration of all built-in callbacks
with the global ``CALLBACKS`` registry.
"""

from minomaly.callbacks.base import Callback
from minomaly.callbacks.composite import CallbackList

# Import concrete callbacks to trigger their @CALLBACKS.register decorators.
from minomaly.callbacks import checkpoint as _checkpoint  # noqa: F401
from minomaly.callbacks import evaluation as _evaluation  # noqa: F401
from minomaly.callbacks import logging_cb as _logging_cb  # noqa: F401
from minomaly.callbacks import visualization as _visualization  # noqa: F401

__all__ = ["Callback", "CallbackList"]
