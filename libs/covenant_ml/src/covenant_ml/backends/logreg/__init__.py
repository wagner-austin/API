"""Logistic Regression backend for covenant_ml.

Exports the backend factory and capabilities.
"""

from __future__ import annotations

from .backend import LOGREG_CAPABILITIES, LogRegBackend, create_logreg_backend

__all__ = [
    "LOGREG_CAPABILITIES",
    "LogRegBackend",
    "create_logreg_backend",
]
