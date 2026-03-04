"""Tabular MLP model definition (shapes only; no torch import here).

The actual implementation relies on platform_ml.torch_types for Protocols and
dynamic import; this file defines a thin adapter-friendly API.
"""

from __future__ import annotations

from typing import Protocol

from platform_ml.torch_types import TrainableModel


class MLPFactory(Protocol):
    """Factory Protocol to create a TrainableModel without importing torch."""

    def __call__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: tuple[int, ...],
    ) -> TrainableModel: ...


__all__ = ["MLPFactory"]
