"""JSON serialization helpers with NumPy type support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom :class:`json.JSONEncoder` that handles NumPy data types.

    Supported conversions:

    * Integer scalars -> :class:`int`
    * Floating-point scalars -> :class:`float`
    * Complex scalars -> ``{"real": ..., "imag": ...}``
    * :class:`numpy.ndarray` -> nested :class:`list`
    * :class:`numpy.bool_` -> :class:`bool`
    * :class:`numpy.void` -> ``None``
    """

    def default(self, obj: Any) -> Any:
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.void):
            return None

        return super().default(obj)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Serialize *data* to a JSON file using :class:`NumpyEncoder`.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    data:
        The dictionary to serialize.
    path:
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, cls=NumpyEncoder, indent=4)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file and return the parsed dictionary.

    Parameters
    ----------
    path:
        Path to the JSON file.
    """
    path = Path(path)
    with open(path, "r") as fh:
        return json.load(fh)
