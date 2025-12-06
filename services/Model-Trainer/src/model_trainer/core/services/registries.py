from __future__ import annotations

from collections.abc import Callable, Mapping

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from ..contracts.dataset import DatasetBuilder
from ..contracts.model import BackendCapabilities, ModelBackend
from ..contracts.tokenizer import TokenizerBackend

BackendFactory = Callable[[DatasetBuilder], ModelBackend]


class BackendRegistration:
    """Registration entry for a model backend with factory and capabilities."""

    factory: BackendFactory
    capabilities: BackendCapabilities

    def __init__(
        self: BackendRegistration,
        factory: BackendFactory,
        capabilities: BackendCapabilities,
    ) -> None:
        self.factory = factory
        self.capabilities = capabilities


class ModelRegistry:
    """Registry for model backends with lazy loading.

    Backends are instantiated on first access, not at registry creation time.
    Capabilities can be queried without instantiating backends.
    """

    _registrations: Mapping[str, BackendRegistration]
    _dataset_builder: DatasetBuilder
    _cache: dict[str, ModelBackend]

    def __init__(
        self: ModelRegistry,
        registrations: Mapping[str, BackendRegistration],
        dataset_builder: DatasetBuilder,
    ) -> None:
        self._registrations = registrations
        self._dataset_builder = dataset_builder
        self._cache = {}

    def get(self: ModelRegistry, name: str) -> ModelBackend:
        """Get a backend by name, instantiating it if not already cached."""
        if name not in self._registrations:
            raise AppError(
                ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
                f"Unknown model backend: {name}",
                model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
            )
        if name not in self._cache:
            self._cache[name] = self._registrations[name].factory(self._dataset_builder)
        return self._cache[name]

    def list_backends(self: ModelRegistry) -> list[str]:
        """List all registered backend names."""
        return list(self._registrations.keys())

    def get_capabilities(self: ModelRegistry, name: str) -> BackendCapabilities:
        """Get capabilities for a backend without instantiation."""
        if name not in self._registrations:
            raise AppError(
                ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
                f"Unknown model backend: {name}",
                model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
            )
        return self._registrations[name].capabilities


class TokenizerRegistry:
    backends: Mapping[str, TokenizerBackend]

    def __init__(self: TokenizerRegistry, backends: Mapping[str, TokenizerBackend]) -> None:
        self.backends = backends

    def get(self: TokenizerRegistry, method: str) -> TokenizerBackend:
        if method not in self.backends:
            raise AppError(
                ModelTrainerErrorCode.UNSUPPORTED_BACKEND,
                f"Unknown tokenizer backend: {method}",
                model_trainer_status_for(ModelTrainerErrorCode.UNSUPPORTED_BACKEND),
            )
        return self.backends[method]
