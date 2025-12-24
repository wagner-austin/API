"""Namespace for XGBoost backend.

Exports factory symbol to be used by the registry. Implementation is
provided by backend.py in this package.
"""

from __future__ import annotations

from .backend import XGBOOST_CAPABILITIES, XGBoostBackend, create_xgboost_backend

__all__ = [
    "XGBOOST_CAPABILITIES",
    "XGBoostBackend",
    "create_xgboost_backend",
]
