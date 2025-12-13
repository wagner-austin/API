"""Unified entrypoint for tabular training across backends.

Leverages ClassifierRegistry to select backend and run training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .backends.protocol import ProgressCallback
from .backends.registry import ClassifierRegistry
from .types import BackendName, ClassifierTrainConfig, TrainOutcome


class BaseTabularTrainer:
    """Unified trainer for tabular datasets using pluggable backends."""

    def __init__(self, registry: ClassifierRegistry) -> None:
        self._registry = registry

    def train(
        self,
        *,
        backend: BackendName,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        impl = self._registry.get(backend)
        return impl.train(
            x_features=x_features,
            y_labels=y_labels,
            feature_names=feature_names,
            config=config,
            output_dir=output_dir,
            progress=progress,
        )


__all__ = ["BaseTabularTrainer"]
