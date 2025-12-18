"""Test utilities for the pluggable dataset loading system.

Provides fake implementations for testing without real filesystem access.
These are public utilities exported for consumers to use in their tests.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types import (
    DatasetConfig,
    DatasetMeta,
    LoadedDataset,
)


class FakeDatasetLoader:
    """Fake dataset loader for testing.

    Returns deterministic synthetic datasets based on config.
    Does not access the filesystem.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 10,
        positive_ratio: float = 0.3,
        random_state: int = 42,
    ) -> None:
        """Initialize fake loader with dataset parameters.

        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features per sample.
            positive_ratio: Fraction of positive class samples.
            random_state: Random seed for reproducibility.
        """
        self._n_samples = n_samples
        self._n_features = n_features
        self._positive_ratio = positive_ratio
        self._random_state = random_state

    def load(
        self,
        config: DatasetConfig,
        external_dir: Path,
    ) -> LoadedDataset:
        """Generate a synthetic dataset based on config.

        Args:
            config: Dataset configuration (used for metadata).
            external_dir: Ignored for fake loader.

        Returns:
            LoadedDataset with synthetic data matching config name.
        """
        rng = np.random.default_rng(self._random_state)

        # Generate features
        x_array: NDArray[np.float64] = rng.standard_normal(
            (self._n_samples, self._n_features)
        ).astype(np.float64)

        # Generate labels with specified positive ratio
        n_positive = int(self._n_samples * self._positive_ratio)
        n_negative = self._n_samples - n_positive

        y_array: NDArray[np.int64] = np.zeros(self._n_samples, dtype=np.int64)
        y_array[:n_positive] = 1
        rng.shuffle(y_array)

        # Generate feature names
        feature_names = tuple(f"feature_{i}" for i in range(self._n_features))

        meta = DatasetMeta(
            name=config["name"],
            n_samples=self._n_samples,
            n_features=self._n_features,
            n_positive=n_positive,
            n_negative=n_negative,
            positive_ratio=self._positive_ratio,
            feature_names=feature_names,
            categorical_encodings=(),  # Fake loader generates numeric data only
        )

        return LoadedDataset(meta=meta, x=x_array, y=y_array)


def create_fake_dataset_loader(
    n_samples: int = 100,
    n_features: int = 10,
    positive_ratio: float = 0.3,
    random_state: int = 42,
) -> FakeDatasetLoader:
    """Factory function for creating fake dataset loader.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features per sample.
        positive_ratio: Fraction of positive class samples.
        random_state: Random seed for reproducibility.

    Returns:
        FakeDatasetLoader configured with specified parameters.
    """
    return FakeDatasetLoader(
        n_samples=n_samples,
        n_features=n_features,
        positive_ratio=positive_ratio,
        random_state=random_state,
    )


__all__ = [
    "FakeDatasetLoader",
    "create_fake_dataset_loader",
]
