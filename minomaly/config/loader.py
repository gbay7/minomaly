"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .schema import (
    DatasetConfig,
    DeviceConfig,
    MinomalyConfig,
    ModelConfig,
    OutlierConfig,
    SamplingConfig,
    SearchConfig,
    TrainingConfig,
    VisualizationConfig,
)

# Map of top-level field names to their dataclass types, used for manual
# dict unpacking when dacite is not available.
_SUB_CONFIG_TYPES: Dict[str, type] = {
    "device": DeviceConfig,
    "model": ModelConfig,
    "sampling": SamplingConfig,
    "search": SearchConfig,
    "outlier": OutlierConfig,
    "training": TrainingConfig,
    "visualization": VisualizationConfig,
    "dataset": DatasetConfig,
}


def _manual_from_dict(data: Dict[str, Any]) -> MinomalyConfig:
    """Build a :class:`MinomalyConfig` without dacite by manually
    unpacking nested dictionaries."""
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _SUB_CONFIG_TYPES and isinstance(value, dict):
            kwargs[key] = _SUB_CONFIG_TYPES[key](**value)
        else:
            kwargs[key] = value
    return MinomalyConfig(**kwargs)


def config_from_dict(d: Dict[str, Any]) -> MinomalyConfig:
    """Create a :class:`MinomalyConfig` from a plain dictionary.

    Uses *dacite* for recursive dataclass construction when available,
    falling back to manual unpacking otherwise.
    """
    try:
        import dacite

        return dacite.from_dict(data_class=MinomalyConfig, data=d)
    except ImportError:
        return _manual_from_dict(d)


def load_config(path: Union[str, Path]) -> MinomalyConfig:
    """Load a YAML configuration file and return a :class:`MinomalyConfig`.

    Parameters
    ----------
    path:
        Path to a ``.yaml`` / ``.yml`` file.
    """
    path = Path(path)
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    return config_from_dict(data)


def _set_nested_attr(obj: Any, dotted_key: str, value: Any) -> None:
    """Set an attribute on a (possibly nested) dataclass using dot notation.

    For example ``_set_nested_attr(cfg, "search.n_beams", 4)`` sets
    ``cfg.search.n_beams = 4``.
    """
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    final_key = parts[-1]

    # Attempt to coerce the value to the field's declared type so that
    # string overrides from CLI args (e.g. "4") are converted properly.
    if is_dataclass(obj):
        for f in fields(obj):
            if f.name == final_key:
                target_type = f.type
                # Resolve simple built-in types for coercion.
                _COERCIBLE = {"int": int, "float": float, "bool": _str_to_bool, "str": str}
                if target_type in _COERCIBLE:
                    value = _COERCIBLE[target_type](value)
                elif isinstance(target_type, type) and target_type in (int, float, str, bool):
                    if target_type is bool:
                        value = _str_to_bool(value)
                    else:
                        value = target_type(value)
                break

    setattr(obj, final_key, value)


def _str_to_bool(v: Any) -> bool:
    """Convert a string value to bool, accepting common truthy/falsy strings."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("true", "1", "yes", "on"):
            return True
        if v.lower() in ("false", "0", "no", "off"):
            return False
    return bool(v)


def merge_cli_overrides(config: MinomalyConfig, overrides: Dict[str, Any]) -> MinomalyConfig:
    """Apply dot-notation overrides to *config* and return the mutated object.

    Parameters
    ----------
    config:
        The base configuration to modify.
    overrides:
        A mapping of dotted keys to values, e.g.
        ``{"search.n_beams": 4, "seed": 123}``.

    Returns
    -------
    MinomalyConfig
        The same *config* instance, modified in-place for convenience.
    """
    config = copy.deepcopy(config)
    for key, value in overrides.items():
        _set_nested_attr(config, key, value)
    return config
