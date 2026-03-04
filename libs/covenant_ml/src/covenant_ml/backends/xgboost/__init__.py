"""Namespace for XGBoost backends (classifier and regressor).

Exports factory symbols to be used by registries. Implementations
are provided by backend.py (classifier) and regressor.py in this package.
"""

from __future__ import annotations

from .backend import XGBOOST_CAPABILITIES, XGBoostBackend, create_xgboost_backend
from .regressor import (
    XGBOOST_REGRESSOR_CAPABILITIES,
    XGBoostRegressorBackend,
    create_xgboost_regressor_backend,
)

__all__ = [
    "XGBOOST_CAPABILITIES",
    "XGBOOST_REGRESSOR_CAPABILITIES",
    "XGBoostBackend",
    "XGBoostRegressorBackend",
    "create_xgboost_backend",
    "create_xgboost_regressor_backend",
]
