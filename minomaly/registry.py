"""Generic registry system for swappable components.

Usage:
    from minomaly.registry import SAMPLERS

    @SAMPLERS.register("tree")
    class TreeSampler(Sampler): ...

    sampler = SAMPLERS.build("tree", n_neighborhoods=10000)
"""

from __future__ import annotations

from typing import TypeVar, Generic, Callable

T = TypeVar("T")


class Registry(Generic[T]):
    """A typed dictionary mapping string keys to component classes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class under *key*."""
        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry:
                raise ValueError(
                    f"Registry '{self.name}': key '{key}' already registered "
                    f"to {self._registry[key].__name__}"
                )
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, key: str, **kwargs) -> T:
        """Instantiate the class registered under *key*."""
        cls = self[key]
        return cls(**kwargs)

    def __getitem__(self, key: str) -> type[T]:
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(
                f"Registry '{self.name}': '{key}' not found. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def list_available(self) -> list[str]:
        return sorted(self._registry)

    def __repr__(self) -> str:
        return f"Registry('{self.name}', keys={self.list_available()})"


# ── Top-level registries ──────────────────────────────────────────────
SAMPLERS: Registry    = Registry("samplers")
ENCODERS: Registry    = Registry("encoders")
EMBEDDERS: Registry   = Registry("embedders")
SCORING: Registry     = Registry("scoring")
SEARCH: Registry      = Registry("search")
OUTLIERS: Registry    = Registry("outliers")
GENERATORS: Registry  = Registry("generators")
CALLBACKS: Registry   = Registry("callbacks")
