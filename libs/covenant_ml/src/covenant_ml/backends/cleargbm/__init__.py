"""Namespace for ClearGBM backend.

Exports factory symbol to be used by the registry. Implementation is
provided by backend.py in this package.
"""

from __future__ import annotations

from .backend import (
    CLEARGBM_CAPABILITIES,
    ClearGBMBackend,
    create_cleargbm_backend,
    try_extract_cleargbm_model,
)

__all__ = [
    "CLEARGBM_CAPABILITIES",
    "ClearGBMBackend",
    "create_cleargbm_backend",
    "try_extract_cleargbm_model",
]
