"""Namespace for MLP backends (classifier and regressor).

Exports factory symbols to be used by the registry. Implementations
provided by backend.py (classifier) and regressor.py (regressor).
"""

from __future__ import annotations

from .backend import MLP_CAPABILITIES, create_mlp_backend
from .regressor import MLP_REGRESSOR_CAPABILITIES, create_mlp_regressor_backend

__all__ = [
    "MLP_CAPABILITIES",
    "MLP_REGRESSOR_CAPABILITIES",
    "create_mlp_backend",
    "create_mlp_regressor_backend",
]
