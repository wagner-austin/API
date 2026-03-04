"""Namespace for LightGBM backend.

Exports factory symbols to be used by the registry. Implementation is
provided by backend.py (classifier) and regressor.py (regressor).
"""

from __future__ import annotations

from .backend import LIGHTGBM_CAPABILITIES, LightGBMBackend, create_lightgbm_backend
from .regressor import (
    LIGHTGBM_REGRESSOR_CAPABILITIES,
    LightGBMRegressorBackend,
    create_lightgbm_regressor_backend,
)

__all__ = [
    "LIGHTGBM_CAPABILITIES",
    "LIGHTGBM_REGRESSOR_CAPABILITIES",
    "LightGBMBackend",
    "LightGBMRegressorBackend",
    "create_lightgbm_backend",
    "create_lightgbm_regressor_backend",
]
