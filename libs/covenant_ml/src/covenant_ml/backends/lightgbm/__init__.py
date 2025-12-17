"""Namespace for LightGBM backend.

Exports factory symbol to be used by the registry. Implementation is
provided by backend.py in this package.
"""

from __future__ import annotations

from .backend import LIGHTGBM_CAPABILITIES, LightGBMBackend, create_lightgbm_backend

__all__ = [
    "LIGHTGBM_CAPABILITIES",
    "LightGBMBackend",
    "create_lightgbm_backend",
]
