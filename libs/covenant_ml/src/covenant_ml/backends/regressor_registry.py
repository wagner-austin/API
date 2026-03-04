"""Backend registry for pluggable tabular regressors.

Parallel to registry.py for classifiers. Same pattern: explicit
registration, cached capabilities, lazy instantiation.
Strict typing; no optional fallbacks.
"""

from __future__ import annotations

from typing import Protocol

from ..types import RegressorBackendName
from .protocol import BackendCapabilities
from .regressor_protocol import RegressorBackend


class RegressorBackendFactory(Protocol):
    """Factory protocol to construct a regressor backend implementation."""

    def __call__(self) -> RegressorBackend: ...


class RegressorBackendRegistration:
    """Registration record holding a factory and cached capabilities.

    Parallel to BackendRegistration for classifiers.
    """

    def __init__(self, factory: RegressorBackendFactory) -> None:
        self._factory = factory
        self._capabilities_cache: BackendCapabilities | None = None

    def factory(self) -> RegressorBackendFactory:
        """Return the factory callable.

        Returns:
            The factory that creates RegressorBackend instances.
        """
        return self._factory

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities, creating a backend to query if not cached.

        Returns:
            BackendCapabilities for this regressor backend.
        """
        if self._capabilities_cache is None:
            backend = self._factory()
            self._capabilities_cache = backend.capabilities()
        return self._capabilities_cache


class RegressorRegistry:
    """Registry of regressor backends keyed by name.

    Parallel to ClassifierRegistry. Same API surface.
    """

    def __init__(self) -> None:
        self._map: dict[RegressorBackendName, RegressorBackendRegistration] = {}

    def register(
        self, name: RegressorBackendName, registration: RegressorBackendRegistration
    ) -> None:
        """Register a regressor backend.

        Args:
            name: Backend name literal.
            registration: Registration record with factory.
        """
        self._map[name] = registration

    def list_backends(self) -> list[RegressorBackendName]:
        """List all registered backend names, sorted.

        Returns:
            Sorted list of registered backend names.
        """
        return sorted(self._map.keys())

    def get(self, name: RegressorBackendName) -> RegressorBackend:
        """Create and return a backend instance by name.

        Args:
            name: Backend name to look up.

        Returns:
            A fresh RegressorBackend instance.

        Raises:
            KeyError: If the backend name is not registered.
        """
        reg = self._map[name]
        return reg.factory()()

    def get_capabilities(self, name: RegressorBackendName) -> BackendCapabilities:
        """Return capabilities for a registered backend.

        Args:
            name: Backend name to look up.

        Returns:
            BackendCapabilities for the named backend.

        Raises:
            KeyError: If the backend name is not registered.
        """
        return self._map[name].capabilities()


def default_regressor_registry() -> RegressorRegistry:
    """Build the default registry with supported regressor backends.

    Returns:
        A RegressorRegistry with all available regressor backends.
    """
    from .lightgbm.regressor import create_lightgbm_regressor_backend
    from .xgboost.regressor import create_xgboost_regressor_backend

    reg = RegressorRegistry()
    reg.register(
        "xgboost_reg",
        RegressorBackendRegistration(create_xgboost_regressor_backend),
    )
    reg.register(
        "lightgbm_reg",
        RegressorBackendRegistration(create_lightgbm_regressor_backend),
    )
    return reg


__all__ = [
    "RegressorBackendFactory",
    "RegressorBackendRegistration",
    "RegressorRegistry",
    "default_regressor_registry",
]
