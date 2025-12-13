"""Backend registry for pluggable tabular classifiers.

Strict typing; no optional fallbacks. Backends are registered explicitly.
"""

from __future__ import annotations

from typing import Protocol

from ..types import BackendName
from .protocol import BackendCapabilities, ClassifierBackend


class BackendFactory(Protocol):
    """Factory protocol to construct a backend implementation."""

    def __call__(self) -> ClassifierBackend: ...


class BackendRegistration:
    """Registration record holding a factory and cached capabilities."""

    def __init__(self, factory: BackendFactory) -> None:
        self._factory = factory
        self._capabilities_cache: BackendCapabilities | None = None

    def factory(self) -> BackendFactory:
        return self._factory

    def capabilities(self) -> BackendCapabilities:
        if self._capabilities_cache is None:
            backend = self._factory()
            self._capabilities_cache = backend.capabilities()
        return self._capabilities_cache


class ClassifierRegistry:
    """Registry of classifier backends keyed by name."""

    def __init__(self) -> None:
        self._map: dict[BackendName, BackendRegistration] = {}

    def register(self, name: BackendName, registration: BackendRegistration) -> None:
        self._map[name] = registration

    def list_backends(self) -> list[BackendName]:
        return sorted(self._map.keys())

    def get(self, name: BackendName) -> ClassifierBackend:
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: BackendName) -> BackendCapabilities:
        return self._map[name].capabilities()


def default_registry() -> ClassifierRegistry:
    """Build the default registry with supported backends.

    Includes:
    - xgboost: wraps existing XGBoost trainer
    - mlp: torch-based MLP backend (package provided separately)
    """
    reg = ClassifierRegistry()

    # XGBoost backend
    xgb_mod = __import__(
        "covenant_ml.backends.xgboost_backend",
        fromlist=["create_xgboost_backend"],
    )
    create_xgboost_backend: BackendFactory = xgb_mod.create_xgboost_backend
    reg.register("xgboost", BackendRegistration(create_xgboost_backend))

    # MLP backend (present, implemented in separate subpackage)
    mlp_pkg = __import__(
        "covenant_ml.backends.mlp",
        fromlist=["create_mlp_backend"],
    )
    create_mlp_backend: BackendFactory = mlp_pkg.create_mlp_backend
    reg.register("mlp", BackendRegistration(create_mlp_backend))

    return reg


__all__ = [
    "BackendFactory",
    "BackendRegistration",
    "ClassifierRegistry",
    "default_registry",
]
