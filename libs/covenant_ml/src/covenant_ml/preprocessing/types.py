"""Preprocessing types for automatic data cleaning.

Provides TypedDicts for preprocessing configuration and state.
All types are immutable (total=True) and strictly typed.

Preprocessing handles:
- Outlier detection and capping (percentile-based)
- Special code detection (96, 98, 999 → treat as missing)
- Missing value imputation (median/mean)
- Z-score normalization
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

# Imputation strategy literals
ImputationStrategy = Literal["median", "mean", "zero"]


class OutlierBounds(TypedDict, total=True):
    """Per-feature outlier bounds for capping.

    Values below lower or above upper are clipped to the bounds.
    Bounds are computed from training data percentiles.

    Attributes:
        feature_idx: Zero-based feature column index.
        lower: Lower bound (values below are clipped up).
        upper: Upper bound (values above are clipped down).
    """

    feature_idx: int
    lower: float
    upper: float


class SpecialCodeSpec(TypedDict, total=True):
    """Per-feature special codes to treat as missing.

    Special codes are domain-specific values that indicate missing data
    but are encoded as regular numbers (e.g., 96, 98, 999, -1).

    Attributes:
        feature_idx: Zero-based feature column index.
        codes: Tuple of values to treat as missing.
    """

    feature_idx: int
    codes: tuple[float, ...]


class ImputationSpec(TypedDict, total=True):
    """Per-feature imputation value for missing data.

    Computed from training data only (excluding NaN and special codes).

    Attributes:
        feature_idx: Zero-based feature column index.
        impute_value: Value to replace missing/NaN with.
    """

    feature_idx: int
    impute_value: float


class PreprocessingState(TypedDict, total=True):
    """Immutable fitted preprocessing state.

    Captured from training data during fit(), applied to all data during
    transform(). This separation prevents data leakage from validation/test
    sets into the preprocessing statistics.

    Attributes:
        n_features: Number of features in the original data.
        outlier_bounds: Per-feature outlier bounds for capping.
        special_codes: Per-feature special codes to replace with NaN.
        imputation_values: Per-feature imputation values for NaN replacement.
        feature_means: Per-feature mean values for z-score normalization.
        feature_stds: Per-feature std values for z-score normalization.
    """

    n_features: int
    outlier_bounds: tuple[OutlierBounds, ...]
    special_codes: tuple[SpecialCodeSpec, ...]
    imputation_values: tuple[ImputationSpec, ...]
    feature_means: NDArray[np.float64]
    feature_stds: NDArray[np.float64]


# Default special codes commonly used to indicate missing data
DEFAULT_SPECIAL_CODES: frozenset[float] = frozenset(
    {
        96.0,
        98.0,
        99.0,
        999.0,
        9999.0,
        -1.0,
        -9.0,
        -99.0,
        -999.0,
    }
)

# Default percentiles for outlier bounds
DEFAULT_OUTLIER_PERCENTILES: tuple[float, float] = (1.0, 99.0)

# Default imputation strategy
DEFAULT_IMPUTATION_STRATEGY: ImputationStrategy = "median"


__all__ = [
    "DEFAULT_IMPUTATION_STRATEGY",
    "DEFAULT_OUTLIER_PERCENTILES",
    "DEFAULT_SPECIAL_CODES",
    "ImputationSpec",
    "ImputationStrategy",
    "OutlierBounds",
    "PreprocessingState",
    "SpecialCodeSpec",
]
