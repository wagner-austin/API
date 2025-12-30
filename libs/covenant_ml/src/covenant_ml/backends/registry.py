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
    - mlp: torch-based MLP backend
    - lstm: torch-based LSTM backend for temporal sequences
    - lightgbm: LightGBM gradient boosting backend
    - cleargbm: numpy-based gradient boosting with built-in interpretability
    - logreg: sklearn Logistic Regression (interpretable baseline)
    - random_forest: sklearn Random Forest (bagging ensemble)
    """
    reg = ClassifierRegistry()

    # XGBoost backend
    xgb_mod = __import__(
        "covenant_ml.backends.xgboost",
        fromlist=["create_xgboost_backend"],
    )
    create_xgboost_backend: BackendFactory = xgb_mod.create_xgboost_backend
    reg.register("xgboost", BackendRegistration(create_xgboost_backend))

    # MLP backend
    mlp_pkg = __import__(
        "covenant_ml.backends.mlp",
        fromlist=["create_mlp_backend"],
    )
    create_mlp_backend: BackendFactory = mlp_pkg.create_mlp_backend
    reg.register("mlp", BackendRegistration(create_mlp_backend))

    # LSTM backend
    lstm_pkg = __import__(
        "covenant_ml.backends.lstm",
        fromlist=["create_lstm_backend"],
    )
    create_lstm_backend: BackendFactory = lstm_pkg.create_lstm_backend
    reg.register("lstm", BackendRegistration(create_lstm_backend))

    # LightGBM backend
    lgbm_pkg = __import__(
        "covenant_ml.backends.lightgbm",
        fromlist=["create_lightgbm_backend"],
    )
    create_lightgbm_backend: BackendFactory = lgbm_pkg.create_lightgbm_backend
    reg.register("lightgbm", BackendRegistration(create_lightgbm_backend))

    # ClearGBM backend
    cgbm_pkg = __import__(
        "covenant_ml.backends.cleargbm",
        fromlist=["create_cleargbm_backend"],
    )
    create_cleargbm_backend: BackendFactory = cgbm_pkg.create_cleargbm_backend
    reg.register("cleargbm", BackendRegistration(create_cleargbm_backend))

    # Logistic Regression backend
    logreg_pkg = __import__(
        "covenant_ml.backends.logreg",
        fromlist=["create_logreg_backend"],
    )
    create_logreg_backend: BackendFactory = logreg_pkg.create_logreg_backend
    reg.register("logreg", BackendRegistration(create_logreg_backend))

    # Random Forest backend
    rf_pkg = __import__(
        "covenant_ml.backends.random_forest",
        fromlist=["create_random_forest_backend"],
    )
    create_random_forest_backend: BackendFactory = rf_pkg.create_random_forest_backend
    reg.register("random_forest", BackendRegistration(create_random_forest_backend))

    return reg


__all__ = [
    "BackendFactory",
    "BackendRegistration",
    "ClassifierRegistry",
    "default_registry",
]
