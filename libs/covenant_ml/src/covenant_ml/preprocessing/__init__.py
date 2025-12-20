"""Automatic preprocessing for data cleaning.

Provides pluggable preprocessing that automatically handles:
- Outlier detection and capping (percentile-based)
- Special code detection (96, 98, 999 → treat as missing)
- Missing value imputation (median/mean)
- Z-score normalization

All statistics are computed from training data only to prevent leakage.

Example:
    >>> from covenant_ml.preprocessing import AutoPreprocessor
    >>> preprocessor = AutoPreprocessor()
    >>> state = preprocessor.fit(x_train, y_train)
    >>> x_train_clean = preprocessor.transform(x_train, state)
    >>> x_val_clean = preprocessor.transform(x_val, state)
"""

from covenant_ml.preprocessing.pipeline import (
    AutoPreprocessor,
    apply_zscore,
    cap_outliers,
    compute_feature_stats,
    compute_imputation_values,
    create_auto_preprocessor,
    detect_outlier_bounds,
    detect_special_codes,
    impute_missing,
    replace_special_codes,
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

__all__ = [
    "DEFAULT_IMPUTATION_STRATEGY",
    "DEFAULT_OUTLIER_PERCENTILES",
    "DEFAULT_SPECIAL_CODES",
    "AutoPreprocessor",
    "ImputationSpec",
    "ImputationStrategy",
    "OutlierBounds",
    "PreprocessingState",
    "SpecialCodeSpec",
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
