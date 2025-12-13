"""Backends package for covenant_ml.

Exports only registry/protocols; concrete backends live in submodules.
"""

from __future__ import annotations

from .protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback
from .registry import BackendFactory, BackendRegistration, ClassifierRegistry, default_registry

__all__ = [
    "BackendCapabilities",
    "BackendFactory",
    "BackendRegistration",
    "ClassifierBackend",
    "ClassifierRegistry",
    "PreparedClassifier",
    "ProgressCallback",
    "default_registry",
]
