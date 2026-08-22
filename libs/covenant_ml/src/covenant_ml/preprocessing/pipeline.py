"""Auto-preprocessing pipeline for data cleaning.

Provides automatic detection and handling of data quality issues:
- Outliers (percentile-based capping)
- Special codes (domain values representing missing data)
- Missing values (imputation with median/mean)
- Feature scaling (z-score normalization)

All statistics are computed from training data only to prevent leakage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.preprocessing._numeric import (
    _finite_mask,
    _isclose_mask,
    _nan_mask,
    _replace_zeros_with_one,
    _safe_mean,
    _safe_median,
    _safe_nanmean,
    _safe_nanstd,
    _safe_percentile,
)
from covenant_ml.preprocessing.types import (
    DEFAULT_IMPUTATION_STRATEGY,
    DEFAULT_OUTLIER_PERCENTILES,
    DEFAULT_SPECIAL_CODES,
    ImputationSpec,
    ImputationStrategy,
    OutlierBounds,
    PreprocessingState,
    SpecialCodeSpec,
)

_log = get_logger(__name__)


# =============================================================================
# Type-safe numpy helpers
# =============================================================================


# =============================================================================
# Detection Functions
# =============================================================================


def detect_outlier_bounds(
    x_train: NDArray[np.float64],
    percentiles: tuple[float, float] = DEFAULT_OUTLIER_PERCENTILES,
) -> tuple[OutlierBounds, ...]:
    """Detect outlier bounds per feature using percentiles.

    Computes lower and upper bounds from training data. Values outside
    these bounds are considered outliers and will be capped during transform.

    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).
        percentiles: Tuple of (lower_percentile, upper_percentile).
            Default is (1.0, 99.0) meaning 1st and 99th percentiles.

    Returns:
        Tuple of OutlierBounds, one per feature.
    """
    lower_pct, upper_pct = percentiles
    n_features: int = int(x_train.shape[1])
    bounds: list[OutlierBounds] = []

    for feat_idx in range(n_features):
        col: NDArray[np.float64] = x_train[:, feat_idx]
        # Exclude NaN values from percentile computation
        valid_mask: NDArray[np.bool_] = _finite_mask(col)
        valid_values: NDArray[np.float64] = col[valid_mask]

        if len(valid_values) == 0:
            # No valid values - use 0.0 as placeholder bounds
            lower = 0.0
            upper = 0.0
        else:
            lower = _safe_percentile(valid_values, lower_pct)
            upper = _safe_percentile(valid_values, upper_pct)

        bounds.append(
            OutlierBounds(
                feature_idx=feat_idx,
                lower=lower,
                upper=upper,
            )
        )

    return tuple(bounds)


def detect_special_codes(
    x_train: NDArray[np.float64],
    known_codes: frozenset[float] = DEFAULT_SPECIAL_CODES,
    min_frequency: float = 0.001,
) -> tuple[SpecialCodeSpec, ...]:
    """Detect special codes that represent missing data.

    Scans each feature for known special code values (96, 98, 999, etc.)
    that appear frequently enough to be intentional missing indicators.

    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).
        known_codes: Set of values to check for special codes.
        min_frequency: Minimum frequency (fraction of samples) for a code
            to be considered a special code. Default 0.001 (0.1%).

    Returns:
        Tuple of SpecialCodeSpec for features with detected special codes.
    """
    n_samples: int = int(x_train.shape[0])
    n_features: int = int(x_train.shape[1])
    min_count: int = max(1, int(n_samples * min_frequency))
    specs: list[SpecialCodeSpec] = []

    for feat_idx in range(n_features):
        col = x_train[:, feat_idx]
        detected_codes: list[float] = []

        for code in known_codes:
            # Count occurrences of this code value
            count = int(np.sum(np.isclose(col, code, rtol=1e-9, atol=1e-9)))
            if count >= min_count:
                detected_codes.append(code)

        if detected_codes:
            specs.append(
                SpecialCodeSpec(
                    feature_idx=feat_idx,
                    codes=tuple(sorted(detected_codes)),
                )
            )

    return tuple(specs)


def compute_imputation_values(
    x_train: NDArray[np.float64],
    special_codes: tuple[SpecialCodeSpec, ...],
    strategy: ImputationStrategy = DEFAULT_IMPUTATION_STRATEGY,
) -> tuple[ImputationSpec, ...]:
    """Compute imputation values per feature from training data.

    Excludes NaN values and detected special codes when computing
    the imputation statistic (median or mean).

    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).
        special_codes: Previously detected special codes to exclude.
        strategy: Imputation strategy ("median", "mean", or "zero").

    Returns:
        Tuple of ImputationSpec, one per feature.
    """
    n_features: int = int(x_train.shape[1])

    # Build lookup of special codes per feature
    codes_by_feature: dict[int, frozenset[float]] = {}
    for spec in special_codes:
        codes_by_feature[spec["feature_idx"]] = frozenset(spec["codes"])

    specs: list[ImputationSpec] = []

    for feat_idx in range(n_features):
        col: NDArray[np.float64] = x_train[:, feat_idx]

        # Build mask excluding NaN and special codes
        valid_mask: NDArray[np.bool_] = _finite_mask(col)

        feature_codes = codes_by_feature.get(feat_idx, frozenset())
        for code in feature_codes:
            code_mask: NDArray[np.bool_] = _isclose_mask(col, code)
            valid_mask = valid_mask & ~code_mask

        valid_values: NDArray[np.float64] = col[valid_mask]

        # Compute imputation value
        if len(valid_values) == 0:
            impute_value = 0.0
        elif strategy == "median":
            impute_value = _safe_median(valid_values)
        elif strategy == "mean":
            impute_value = _safe_mean(valid_values)
        else:  # "zero"
            impute_value = 0.0

        specs.append(
            ImputationSpec(
                feature_idx=feat_idx,
                impute_value=impute_value,
            )
        )

    return tuple(specs)


def compute_feature_stats(
    x_train: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute mean and std per feature for z-score normalization.

    Uses np.nan-aware functions to handle any remaining NaN values.

    Args:
        x_train: Training feature matrix of shape (n_samples, n_features).

    Returns:
        Tuple of (means, stds) arrays, each of shape (n_features,).
    """
    # Use nanmean/nanstd to ignore NaN values
    means: NDArray[np.float64] = _safe_nanmean(x_train, axis=0)
    stds: NDArray[np.float64] = _safe_nanstd(x_train, axis=0)

    # Replace zero stds with 1.0 to avoid division by zero
    stds = _replace_zeros_with_one(stds)

    return means, stds


# =============================================================================
# Transform Functions
# =============================================================================


def replace_special_codes(
    x: NDArray[np.float64],
    special_codes: tuple[SpecialCodeSpec, ...],
) -> NDArray[np.float64]:
    """Replace special code values with NaN.

    Args:
        x: Feature matrix to transform (modified in place).
        special_codes: Special codes to replace.

    Returns:
        Transformed feature matrix with special codes replaced by NaN.
    """
    for spec in special_codes:
        feat_idx = spec["feature_idx"]
        col = x[:, feat_idx]

        for code in spec["codes"]:
            mask = np.isclose(col, code, rtol=1e-9, atol=1e-9)
            col[mask] = np.nan

    return x


def cap_outliers(
    x: NDArray[np.float64],
    outlier_bounds: tuple[OutlierBounds, ...],
) -> NDArray[np.float64]:
    """Cap outlier values to bounds.

    Args:
        x: Feature matrix to transform (modified in place).
        outlier_bounds: Per-feature outlier bounds.

    Returns:
        Transformed feature matrix with outliers capped.
    """
    for bounds in outlier_bounds:
        feat_idx = bounds["feature_idx"]
        lower = bounds["lower"]
        upper = bounds["upper"]

        col = x[:, feat_idx]
        # Only clip if bounds are different (avoid no-op)
        if lower < upper:
            np.clip(col, lower, upper, out=col)

    return x


def impute_missing(
    x: NDArray[np.float64],
    imputation_values: tuple[ImputationSpec, ...],
) -> NDArray[np.float64]:
    """Impute missing (NaN) values with precomputed values.

    Args:
        x: Feature matrix to transform (modified in place).
        imputation_values: Per-feature imputation values.

    Returns:
        Transformed feature matrix with NaN values imputed.
    """
    for spec in imputation_values:
        feat_idx: int = spec["feature_idx"]
        impute_value: float = spec["impute_value"]

        col: NDArray[np.float64] = x[:, feat_idx]
        nan_indices: NDArray[np.bool_] = _nan_mask(col)
        col[nan_indices] = impute_value

    return x


def apply_zscore(
    x: NDArray[np.float64],
    means: NDArray[np.float64],
    stds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply z-score normalization: (x - mean) / std.

    Args:
        x: Feature matrix to transform (modified in place).
        means: Per-feature mean values.
        stds: Per-feature std values (zeros replaced with 1.0).

    Returns:
        Normalized feature matrix.
    """
    x -= means
    x /= stds
    return x


# =============================================================================
# AutoPreprocessor Class
# =============================================================================


class AutoPreprocessor:
    """Automatic preprocessor with fit/transform interface.

    Detects and handles common data quality issues:
    1. Special codes (96, 98, 999, etc.) → replaced with NaN
    2. Outliers → capped to percentile bounds
    3. Missing values (NaN) → imputed with median/mean
    4. Feature scaling → z-score normalization

    All statistics are computed from training data only during fit().
    The fitted state is immutable and applied consistently during transform().

    Example:
        >>> preprocessor = AutoPreprocessor()
        >>> state = preprocessor.fit(x_train, y_train)
        >>> x_train_clean = preprocessor.transform(x_train, state)
        >>> x_val_clean = preprocessor.transform(x_val, state)
    """

    def __init__(
        self,
        outlier_percentiles: tuple[float, float] = DEFAULT_OUTLIER_PERCENTILES,
        imputation_strategy: ImputationStrategy = DEFAULT_IMPUTATION_STRATEGY,
        special_codes: frozenset[float] = DEFAULT_SPECIAL_CODES,
    ) -> None:
        """Initialize preprocessor with configuration.

        Args:
            outlier_percentiles: Tuple of (lower, upper) percentiles for
                outlier detection. Default (1.0, 99.0).
            imputation_strategy: Strategy for imputing missing values.
                One of "median", "mean", or "zero". Default "median".
            special_codes: Set of values to detect as special codes.
                Default includes 96, 98, 999, -1, etc.
        """
        self._outlier_percentiles = outlier_percentiles
        self._imputation_strategy = imputation_strategy
        self._special_codes = special_codes

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> PreprocessingState:
        """Fit preprocessor on training data only.

        Detects data quality issues and computes all statistics needed
        for transformation. The returned state is immutable and should
        be passed to transform() for consistent preprocessing.

        Args:
            x_train: Training feature matrix of shape (n_samples, n_features).
            y_train: Training labels (unused, included for API consistency).

        Returns:
            Immutable PreprocessingState containing all fitted parameters.
        """
        _ = y_train  # Unused but kept for consistent API

        n_features = x_train.shape[1]

        # Step 1: Detect special codes
        special_codes = detect_special_codes(
            x_train,
            known_codes=self._special_codes,
        )

        if special_codes:
            _log.info(
                "Detected special codes",
                extra={
                    "n_features_with_codes": len(special_codes),
                    "total_features": n_features,
                },
            )

        # Step 2: Create working copy and replace special codes with NaN
        x_work = x_train.copy()
        replace_special_codes(x_work, special_codes)

        # Step 3: Detect outlier bounds (after special codes are NaN)
        outlier_bounds = detect_outlier_bounds(
            x_work,
            percentiles=self._outlier_percentiles,
        )

        # Step 4: Compute imputation values (excluding NaN/special codes)
        imputation_values = compute_imputation_values(
            x_work,
            special_codes,
            strategy=self._imputation_strategy,
        )

        # Step 5: Apply all transforms to working copy for stat computation
        cap_outliers(x_work, outlier_bounds)
        impute_missing(x_work, imputation_values)

        # Step 6: Compute z-score stats on fully cleaned data
        means, stds = compute_feature_stats(x_work)

        return PreprocessingState(
            n_features=n_features,
            outlier_bounds=outlier_bounds,
            special_codes=special_codes,
            imputation_values=imputation_values,
            feature_means=means,
            feature_stds=stds,
        )

    def transform(
        self,
        x: NDArray[np.float64],
        state: PreprocessingState,
    ) -> NDArray[np.float64]:
        """Transform data using fitted preprocessing state.

        Applies all preprocessing steps in order:
        1. Replace special codes with NaN
        2. Cap outliers to bounds
        3. Impute NaN values
        4. Apply z-score normalization

        Args:
            x: Feature matrix of shape (n_samples, n_features).
            state: Fitted preprocessing state from fit().

        Returns:
            Transformed feature matrix with same shape.

        Raises:
            ValueError: If x has wrong number of features.
        """
        n_features: int = int(x.shape[1])
        expected_features: int = state["n_features"]

        if n_features != expected_features:
            raise ValueError(
                f"Feature count mismatch: got {n_features}, expected {expected_features}"
            )

        # Create copy to avoid modifying input
        x_out: NDArray[np.float64] = x.copy()

        # Apply transforms in order
        replace_special_codes(x_out, state["special_codes"])
        cap_outliers(x_out, state["outlier_bounds"])
        impute_missing(x_out, state["imputation_values"])
        apply_zscore(x_out, state["feature_means"], state["feature_stds"])

        return x_out


def create_auto_preprocessor(
    outlier_percentiles: tuple[float, float] = DEFAULT_OUTLIER_PERCENTILES,
    imputation_strategy: ImputationStrategy = DEFAULT_IMPUTATION_STRATEGY,
    special_codes: frozenset[float] = DEFAULT_SPECIAL_CODES,
) -> AutoPreprocessor:
    """Factory function to create an AutoPreprocessor.

    Args:
        outlier_percentiles: Tuple of (lower, upper) percentiles.
        imputation_strategy: Strategy for imputing missing values.
        special_codes: Set of values to detect as special codes.

    Returns:
        Configured AutoPreprocessor instance.
    """
    return AutoPreprocessor(
        outlier_percentiles=outlier_percentiles,
        imputation_strategy=imputation_strategy,
        special_codes=special_codes,
    )


__all__ = [
    "AutoPreprocessor",
    "apply_zscore",
    "cap_outliers",
    "compute_feature_stats",
    "compute_imputation_values",
    "create_auto_preprocessor",
    "detect_outlier_bounds",
    "detect_special_codes",
    "impute_missing",
    "replace_special_codes",
]
