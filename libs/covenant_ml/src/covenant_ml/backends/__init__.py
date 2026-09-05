"""Backends package for covenant_ml.

Exports only registry/protocols; concrete backends live in submodules.
"""

from __future__ import annotations

from .lightgbm.regressor import (
    LIGHTGBM_REGRESSOR_CAPABILITIES,
    LightGBMRegressorBackend,
    create_lightgbm_regressor_backend,
)
from .protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier
from .registry import BackendFactory, BackendRegistration, ClassifierRegistry, default_registry
from .regressor_protocol import PreparedRegressor, RegressorBackend
from .regressor_registry import (
    RegressorBackendFactory,
    RegressorBackendRegistration,
    RegressorRegistry,
    default_regressor_registry,
)
from .xgboost.regressor import (
    XGBOOST_REGRESSOR_CAPABILITIES,
    XGBoostRegressorBackend,
    create_xgboost_regressor_backend,
)

__all__ = [
    "LIGHTGBM_REGRESSOR_CAPABILITIES",
    "XGBOOST_REGRESSOR_CAPABILITIES",
    "BackendCapabilities",
    "BackendFactory",
    "BackendRegistration",
    "ClassifierBackend",
    "ClassifierRegistry",
    "LightGBMRegressorBackend",
    "PreparedClassifier",
    "PreparedRegressor",
    "RegressorBackend",
    "RegressorBackendFactory",
    "RegressorBackendRegistration",
    "RegressorRegistry",
    "XGBoostRegressorBackend",
    "create_lightgbm_regressor_backend",
    "create_xgboost_regressor_backend",
    "default_registry",
    "default_regressor_registry",
]
