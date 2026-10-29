"""Learnable imbalanced data shared by real classifier integration tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_binary_dataset(
    n_samples: int = 100,
    n_features: int = 4,
    pos_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Generate two separated Gaussian classes with fixed class counts.

    Args:
        n_samples: Number of observations.
        n_features: Number of signal-bearing features.
        pos_ratio: Positive class fraction before integer rounding.
        seed: Seed for reproducible noise and label ordering.

    Returns:
        Features, binary labels and ordered feature names. Each positive
        class feature has mean six instead of zero, so a learning assertion
        tests a real relationship instead of accidental ranking of noise.
    """
    rng = np.random.default_rng(seed)
    x: NDArray[np.float64] = rng.standard_normal((n_samples, n_features))
    y: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    y[: int(n_samples * pos_ratio)] = 1
    rng.shuffle(y)
    signal: NDArray[np.float64] = y[:, None].astype(np.float64)
    signal *= 6.0
    x += signal
    return x, y, [f"f{i}" for i in range(n_features)]
