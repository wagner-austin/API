"""XGBoost model training for covenant breach prediction.

Enhanced trainer with:
- Train/validation/test splits
- Early stopping based on validation AUC
- Comprehensive metrics (loss, AUC, accuracy, precision, recall, F1)
- Progress callbacks for monitoring
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from collections.abc import Callable as TypingCallable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .metrics import (
    compute_all_metrics,
    compute_all_regression_metrics,
    format_metrics_str,
    format_regression_metrics_str,
)
from .preprocessing import AutoPreprocessor, PreprocessingState
from .types import (
    DMatrixFactory,
    DMatrixProtocol,
    FeatureImportance,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RequestedDevice,
    ResolvedDevice,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
    XGBClassifierFactory,
    XGBModelProtocol,
    XGBRegressorFactory,
    XGBRegressorModelProtocol,
)

_log = get_logger(__name__)

# Type for progress callback
ProgressCallback = Callable[[TrainProgress], None]


class SplitData(Protocol):
    """Protocol for split dataset."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]


class FeatureScaler:
    """Stores mean/std per feature column for standardization.

    Computes statistics from training data only (to avoid data leakage).
    Applies: x_normalized = (x - mean) / std

    Attributes:
        mean: Per-column mean values, shape (n_features,)
        std: Per-column standard deviation values, shape (n_features,)
        n_features: Number of features
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
    ) -> None:
        if mean.shape != std.shape:
            raise ValueError(f"mean and std must have same shape: {mean.shape} vs {std.shape}")
        if mean.ndim != 1:
            raise ValueError(f"mean must be 1D array, got {mean.ndim}D")
        self._mean = mean
        self._std = std
        self._n_features = len(mean)

    @property
    def mean(self) -> NDArray[np.float64]:
        """Per-column mean values."""
        return self._mean

    @property
    def std(self) -> NDArray[np.float64]:
        """Per-column standard deviation values."""
        return self._std

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply standardization: (x - mean) / std.

        Args:
            x: Feature matrix, shape (n_samples, n_features)

        Returns:
            Normalized feature matrix with same shape

        Raises:
            ValueError: If x has wrong number of features
        """
        n_cols: int = int(x.shape[1])
        if n_cols != self._n_features:
            raise ValueError(f"Expected {self._n_features} features, got {n_cols}")
        return (x - self._mean) / self._std


def compute_feature_scaler(x_train: NDArray[np.float64]) -> FeatureScaler:
    """Compute feature scaling statistics from training data.

    Uses StandardScaler approach: mean=0, std=1 per column.
    Handles zero-variance columns by setting std=1.0 (no scaling).

    Args:
        x_train: Training feature matrix, shape (n_samples, n_features)

    Returns:
        FeatureScaler with mean/std computed from x_train
    """
    mean: NDArray[np.float64] = np.mean(x_train, axis=0)
    std: NDArray[np.float64] = np.std(x_train, axis=0)

    # Replace zero std with 1.0 to avoid division by zero
    # (columns with zero variance are effectively constant)
    zero_std_mask: NDArray[np.bool_] = std == 0.0
    std_safe: NDArray[np.float64] = np.where(zero_std_mask, 1.0, std)

    n_zero_std = int(np.count_nonzero(zero_std_mask))
    if n_zero_std > 0:
        _log.info(
            "Feature scaler found zero-variance columns",
            extra={"n_zero_variance_cols": n_zero_std},
        )

    return FeatureScaler(mean=mean, std=std_safe)


class DataSplits:
    """Container for train/val/test data splits."""

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


class RegressionDataSplits:
    """Container for regression train/val/test data splits.

    Parallel to DataSplits for classification. Uses float64 targets
    instead of int64 labels.
    """

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.float64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.float64],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        """Number of validation samples."""
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        """Number of test samples."""
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        """Total number of samples across all splits."""
        return self.n_train + self.n_val + self.n_test


class PreprocessedDataSplits:
    """Container for preprocessed train/val/test data splits.

    Full preprocessing is applied using statistics computed from
    training data only (to avoid data leakage). Includes:
    - Outlier capping (percentile-based)
    - Special code replacement (96, 98, 999 → NaN)
    - Missing value imputation (median)
    - Z-score normalization
    """

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
        state: PreprocessingState,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.state = state

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


def preprocess_data_splits(splits: DataSplits) -> PreprocessedDataSplits:
    """Preprocess data splits using statistics from training data only.

    Applies full preprocessing pipeline:
    1. Outlier capping (1st/99th percentile bounds)
    2. Special code replacement (96, 98, 999, etc. → NaN)
    3. Missing value imputation (median)
    4. Z-score normalization (mean=0, std=1)

    All statistics are computed from training data only to prevent leakage.

    Args:
        splits: Raw data splits.

    Returns:
        PreprocessedDataSplits with cleaned and normalized features.
    """
    preprocessor = AutoPreprocessor()
    state = preprocessor.fit(splits.x_train, splits.y_train)

    _log.info(
        "Preprocessing data splits",
        extra={
            "n_features": state["n_features"],
            "n_outlier_bounds": len(state["outlier_bounds"]),
            "n_special_codes": len(state["special_codes"]),
            "n_imputation_values": len(state["imputation_values"]),
            "n_train": splits.n_train,
            "n_val": splits.n_val,
            "n_test": splits.n_test,
        },
    )

    return PreprocessedDataSplits(
        x_train=preprocessor.transform(splits.x_train, state),
        y_train=splits.y_train,
        x_val=preprocessor.transform(splits.x_val, state),
        y_val=splits.y_val,
        x_test=preprocessor.transform(splits.x_test, state),
        y_test=splits.y_test,
        state=state,
    )


class _XGBCoreProto(Protocol):
    def build_info(self) -> dict[str, bool]: ...


class _XGBModuleProto(Protocol):
    core: _XGBCoreProto
    XGBClassifier: XGBClassifierFactory
    DMatrix: DMatrixFactory


class _XGBRegressorModuleProto(Protocol):
    """Protocol for the xgboost module used by regression training.

    Includes all _XGBModuleProto attributes for _resolve_device compatibility,
    plus the XGBRegressor factory. The real xgboost module satisfies both.
    """

    core: _XGBCoreProto
    XGBClassifier: XGBClassifierFactory
    XGBRegressor: XGBRegressorFactory
    DMatrix: DMatrixFactory


_cuda_available_hook: TypingCallable[[], bool] | None = None


def set_cuda_available_hook(hook: TypingCallable[[], bool] | None) -> None:
    """Set test hook for CUDA availability detection."""
    global _cuda_available_hook
    _cuda_available_hook = hook


def _detect_cuda_available(xgb_mod: _XGBModuleProto) -> bool:
    """Detect whether XGBoost was built with CUDA/GPU support.

    Checks XGBoost's build configuration via xgb.core.build_info()['USE_CUDA'].
    This indicates whether the XGBoost binary includes GPU acceleration code.

    Note: XGBoost 3.x does NOT accept device='auto' directly. This function
    is used by _resolve_device() to convert 'auto' → 'cuda' or 'cpu' before
    creating XGBClassifier instances.

    Returns:
        True if XGBoost was compiled with CUDA support, False otherwise.
    """
    info = xgb_mod.core.build_info()
    return bool(info.get("USE_CUDA", False))


def _cuda_is_available(xgb_mod: _XGBModuleProto) -> bool:
    """Check CUDA availability with optional test hook.

    Combines build-time CUDA check with optional runtime test hook.
    Used by _resolve_device() for 'auto' mode device selection.
    """
    if _cuda_available_hook is not None:
        return bool(_cuda_available_hook()) and _detect_cuda_available(xgb_mod)
    return _detect_cuda_available(xgb_mod)


def _resolve_device(requested: RequestedDevice, xgb_mod: _XGBModuleProto) -> ResolvedDevice:
    """Resolve 'auto' device to concrete 'cuda' or 'cpu' for XGBoost.

    IMPORTANT: XGBoost 3.x does NOT support device='auto' natively.
    This function MUST be called to convert 'auto' before passing to XGBClassifier.

    Resolution logic:
        - 'cpu'  → 'cpu' (passthrough)
        - 'cuda' → 'cuda' (validates CUDA available, raises if not)
        - 'auto' → 'cuda' if GPU available, else 'cpu'

    Args:
        requested: User-requested device ('cpu', 'cuda', or 'auto')
        xgb_mod: XGBoost module for CUDA availability check

    Returns:
        Resolved device: 'cpu' or 'cuda' (never 'auto')

    Raises:
        RuntimeError: If 'cuda' explicitly requested but not available

    Example:
        >>> resolved = _resolve_device('auto', xgb)
        >>> # resolved is 'cuda' on GPU systems, 'cpu' otherwise
        >>> model = xgb.XGBClassifier(device=resolved, ...)  # Valid!
    """
    if requested == "cpu":
        resolved: ResolvedDevice = "cpu"
    elif requested == "cuda":
        if not _cuda_is_available(xgb_mod):
            raise RuntimeError("CUDA requested but not available")
        resolved = "cuda"
    else:
        # requested == "auto": auto-detect based on CUDA availability
        resolved = "cuda" if _cuda_is_available(xgb_mod) else "cpu"

    # Log device resolution for visibility
    _log.info(
        "Device resolution",
        extra={
            "requested": requested,
            "resolved": resolved,
            "cuda_available": _cuda_is_available(xgb_mod),
        },
    )
    return resolved


def stratified_split(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    groups: NDArray[np.int64] | None = None,
) -> DataSplits:
    """Split data into train/val/test with stratification.

    Maintains class proportions across all splits. When ``groups`` is given,
    the units being split are the groups, not the rows: every row of a group
    lands in the same split, and stratification is by each group's label.
    Rows within one group are correlated (e.g., 1,500 snapshots of one
    match), so a row-level split would place near-duplicates of training
    rows in the test set and score memorization as skill.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        train_ratio: Fraction for training (e.g., 0.7)
        val_ratio: Fraction for validation (e.g., 0.15)
        test_ratio: Fraction for test holdout (e.g., 0.15)
        random_state: Random seed for reproducibility
        groups: Optional group codes (n_samples,); rows sharing a code are
            one entity and are never separated across splits. A group's
            stratification label is its first row's label (constant per
            group for any honest grouped dataset).

    Returns:
        DataSplits container with train/val/test arrays

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance)
    """
    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    rng = np.random.default_rng(random_state)

    if groups is not None:
        return _grouped_split(x_features, y_labels, groups, train_ratio, val_ratio, test_ratio, rng)

    # Get indices for each class (np.where returns tuple, take first element)
    pos_mask: NDArray[np.bool_] = y_labels == 1
    neg_mask: NDArray[np.bool_] = y_labels == 0
    pos_indices: NDArray[np.intp] = np.flatnonzero(pos_mask)
    neg_indices: NDArray[np.intp] = np.flatnonzero(neg_mask)

    # Shuffle indices
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Calculate split points for each class
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    pos_train_end = int(n_pos * train_ratio)
    pos_val_end = int(n_pos * (train_ratio + val_ratio))

    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))

    # Split indices (test is remainder after train+val)
    train_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[:pos_train_end],
            neg_indices[:neg_train_end],
        ]
    )
    val_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_train_end:pos_val_end],
            neg_indices[neg_train_end:neg_val_end],
        ]
    )
    test_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_val_end:],
            neg_indices[neg_val_end:],
        ]
    )

    # Shuffle final indices
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    _log.info(
        "Data split complete",
        extra={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return DataSplits(
        x_train=x_features[train_idx],
        y_train=y_labels[train_idx],
        x_val=x_features[val_idx],
        y_val=y_labels[val_idx],
        x_test=x_features[test_idx],
        y_test=y_labels[test_idx],
    )


def _grouped_split(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    groups: NDArray[np.int64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> DataSplits:
    """Split whole groups into train/val/test, stratified by group label.

    The same ratio arithmetic as the row split, applied to unique groups: a
    group's label is its first row's, positive and negative groups are
    shuffled and cut by the ratios, and every row follows its group. Row
    order within each final split is shuffled so downstream batch-wise
    consumers do not see one group as one contiguous block.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        groups: Group codes (n_samples,)
        train_ratio: Fraction of groups for training
        val_ratio: Fraction of groups for validation
        test_ratio: Fraction of groups for test (remainder)
        rng: Seeded generator shared with the caller

    Returns:
        DataSplits container with train/val/test arrays
    """
    unique_groups: NDArray[np.int64]
    first_row: NDArray[np.intp]
    unique_groups, first_row = np.unique(groups, return_index=True)
    group_labels: NDArray[np.int64] = y_labels[first_row]

    positive_mask: NDArray[np.bool_] = group_labels == 1
    pos_groups: NDArray[np.int64] = unique_groups[positive_mask]
    neg_groups: NDArray[np.int64] = unique_groups[~positive_mask]
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    def cut(pool: NDArray[np.int64]) -> tuple[set[int], set[int], set[int]]:
        train_end = int(len(pool) * train_ratio)
        val_end = int(len(pool) * (train_ratio + val_ratio))
        as_ints: list[int] = []
        for i in range(len(pool)):
            code: np.int64 = pool[i]
            as_ints.append(int(code))
        return set(as_ints[:train_end]), set(as_ints[train_end:val_end]), set(as_ints[val_end:])

    pos_train, pos_val, pos_test = cut(pos_groups)
    neg_train, neg_val, neg_test = cut(neg_groups)

    def rows_of(members: set[int]) -> NDArray[np.intp]:
        keep: NDArray[np.bool_] = np.zeros(len(groups), dtype=np.bool_)
        for row_idx in range(len(groups)):
            row_code: np.int64 = groups[row_idx]
            keep[row_idx] = int(row_code) in members
        rows: NDArray[np.intp] = np.flatnonzero(keep)
        rng.shuffle(rows)
        return rows

    train_idx = rows_of(pos_train | neg_train)
    val_idx = rows_of(pos_val | neg_val)
    test_idx = rows_of(pos_test | neg_test)

    _log.info(
        "Grouped data split complete",
        extra={
            "n_groups": len(unique_groups),
            "n_train_groups": len(pos_train) + len(neg_train),
            "n_val_groups": len(pos_val) + len(neg_val),
            "n_test_groups": len(pos_test) + len(neg_test),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return DataSplits(
        x_train=x_features[train_idx],
        y_train=y_labels[train_idx],
        x_val=x_features[val_idx],
        y_val=y_labels[val_idx],
        x_test=x_features[test_idx],
        y_test=y_labels[test_idx],
    )


def regression_split(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> RegressionDataSplits:
    """Split regression data into train/val/test sets.

    Uses random shuffling (not stratified — regression has no classes).

    Args:
        x_features: Feature matrix (n_samples, n_features).
        y_targets: Continuous target values (n_samples,).
        train_ratio: Fraction for training (e.g., 0.7).
        val_ratio: Fraction for validation (e.g., 0.15).
        test_ratio: Fraction for test holdout (e.g., 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        RegressionDataSplits container with train/val/test arrays.

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    n = len(y_targets)
    rng = np.random.default_rng(random_state)
    indices: NDArray[np.intp] = np.arange(n, dtype=np.intp)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx: NDArray[np.intp] = indices[:train_end]
    val_idx: NDArray[np.intp] = indices[train_end:val_end]
    test_idx: NDArray[np.intp] = indices[val_end:]

    _log.info(
        "Regression data split complete",
        extra={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return RegressionDataSplits(
        x_train=x_features[train_idx],
        y_train=y_targets[train_idx],
        x_val=x_features[val_idx],
        y_val=y_targets[val_idx],
        x_test=x_features[test_idx],
        y_test=y_targets[test_idx],
    )


def _get_probabilities(
    model: XGBModelProtocol,
    x_features: NDArray[np.float64],
    xgb_module: _XGBModuleProto,
) -> NDArray[np.float64]:
    """Get probability predictions for class 1 (breach).

    Uses DMatrix and booster.predict() - works on both CPU and GPU.
    """
    dmatrix: DMatrixProtocol = xgb_module.DMatrix(x_features)
    booster = model.get_booster()
    raw_preds: NDArray[np.float32] = booster.predict(dmatrix)
    return np.asarray(raw_preds, dtype=np.float64)


def _get_regression_predictions(
    model: XGBRegressorModelProtocol,
    x_features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Get regression predictions from a trained model.

    Args:
        model: Trained XGBoost regressor.
        x_features: Feature matrix, shape (n_samples, n_features).

    Returns:
        Predicted values, shape (n_samples,).
    """
    return model.predict(x_features)


def _compute_scale_pos_weight(
    y_labels: NDArray[np.int64],
    config_value: float | None,
) -> float:
    """Compute scale_pos_weight from labels or return provided value.

    Args:
        y_labels: Binary labels array
        config_value: Optional provided scale_pos_weight value

    Returns:
        scale_pos_weight value (provided or auto-calculated)

    Raises:
        ValueError: If no positive samples exist and no value provided
    """
    if config_value is not None:
        _log.info(
            "Using provided scale_pos_weight",
            extra={"scale_pos_weight": config_value},
        )
        return config_value

    pos_mask: NDArray[np.bool_] = y_labels == 1
    neg_mask: NDArray[np.bool_] = y_labels == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples (bankruptcies)")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


def extract_feature_importances(
    model: XGBModelProtocol | XGBRegressorModelProtocol,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract feature importances from trained model.

    Works with both classifier and regressor XGBoost models.

    Args:
        model: Trained XGBoost model (classifier or regressor)
        feature_names: List of feature names (must match number of model features)

    Returns:
        List of FeatureImportance sorted by importance (descending)

    Raises:
        ValueError: If feature_names length doesn't match model features
    """
    raw_importances = model.feature_importances_
    n_features = len(raw_importances)

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match model features ({n_features})"
        )

    names = feature_names

    # Create unsorted list with importances
    # Use flat iterator with item() to get typed float values
    unsorted: list[tuple[str, float]] = []
    for i, imp in enumerate(raw_importances.flat):
        imp_float: float = float(imp.item())
        unsorted.append((names[i], imp_float))

    # Sort by importance descending
    def get_importance(pair: tuple[str, float]) -> float:
        return pair[1]

    sorted_by_importance = sorted(unsorted, key=get_importance, reverse=True)

    # Build result with ranks
    result: list[FeatureImportance] = []
    for rank, (name, importance) in enumerate(sorted_by_importance, start=1):
        result.append(
            FeatureImportance(
                name=name,
                importance=importance,
                rank=rank,
            )
        )

    return result


def train_model_with_validation(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
    output_dir: Path,
    feature_names: list[str],
    progress_callback: ProgressCallback | None = None,
    groups: NDArray[np.int64] | None = None,
) -> TrainOutcome:
    """Train XGBoost classifier with validation and early stopping.

    Implements proper early stopping based on validation AUC:
    - Trains for up to n_estimators rounds
    - Monitors validation AUC after each round
    - Stops if no improvement for early_stopping_rounds consecutive rounds
    - Restores best model based on validation AUC

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        config: Training configuration with hyperparameters
        output_dir: Directory to save model artifacts
        feature_names: List of feature names for importance reporting
        progress_callback: Optional callback for progress updates
        groups: Optional group codes; whole groups share a split

    Returns:
        TrainOutcome with complete training results, metrics, and feature importances
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBModuleProto = xgb
    classifier_factory: XGBClassifierFactory = xgb_module.XGBClassifier
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    # Split data first (needed for auto-calculating scale_pos_weight)
    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_state=config["random_state"],
        groups=groups,
    )

    # Calculate scale_pos_weight from training set if not provided
    scale_pos_weight_computed = _compute_scale_pos_weight(
        splits.y_train, config.get("scale_pos_weight")
    )

    def _build_classifier(total_estimators: int) -> XGBModelProtocol:
        return classifier_factory(
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            n_estimators=total_estimators,
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            random_state=config["random_state"],
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=n_jobs,
            tree_method="hist",
            device=resolved_device,
            scale_pos_weight=scale_pos_weight_computed,
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
        )

    n_estimators = config["n_estimators"]
    early_stopping_rounds = config["early_stopping_rounds"]

    # Track training history
    train_loss_history: list[float] = []
    train_auc_history: list[float] = []
    val_loss_history: list[float] = []
    val_auc_history: list[float] = []

    # Track best model state
    best_val_auc = 0.0
    best_round = 0
    rounds_no_improve = 0
    early_stopped = False

    # We train incrementally by setting n_estimators=1 and using warm_start
    # This allows us to evaluate after each round and implement early stopping
    model: XGBModelProtocol | None = None
    current_round = 0  # Will be updated by loop; 0 if loop never runs

    for current_round in range(1, n_estimators + 1):
        if model is None:
            # First round: create model with 1 estimator
            model = _build_classifier(1)
            model.fit(splits.x_train, splits.y_train, verbose=False)
        else:
            # Subsequent rounds: create new model with more estimators
            # XGBoost doesn't support true warm_start, so we retrain with more trees
            # This is the standard approach for manual early stopping
            model = _build_classifier(current_round)
            model.fit(splits.x_train, splits.y_train, verbose=False)

        # Evaluate on train and validation sets
        train_proba = _get_probabilities(model, splits.x_train, xgb_module)
        val_proba = _get_probabilities(model, splits.x_val, xgb_module)

        train_metrics = compute_all_metrics(splits.y_train, train_proba)
        val_metrics = compute_all_metrics(splits.y_val, val_proba)

        # Record history
        train_loss_history.append(train_metrics["loss"])
        train_auc_history.append(train_metrics["auc"])
        val_loss_history.append(val_metrics["loss"])
        val_auc_history.append(val_metrics["auc"])

        # Report progress
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    round=current_round,
                    total_rounds=n_estimators,
                    train_loss=train_metrics["loss"],
                    train_auc=train_metrics["auc"],
                    val_loss=val_metrics["loss"],
                    val_auc=val_metrics["auc"],
                )
            )

        # Check for improvement (using AUC - higher is better)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_round = current_round
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # Log progress every 50 rounds (quiet by default)
        if current_round % 50 == 0:
            _log.debug(
                "Training progress",
                extra={
                    "round": current_round,
                    "total_rounds": n_estimators,
                    "train_auc": train_metrics["auc"],
                    "val_auc": val_metrics["auc"],
                    "best_val_auc": best_val_auc,
                    "best_round": best_round,
                    "rounds_no_improve": rounds_no_improve,
                },
            )

        # Early stopping check
        if rounds_no_improve >= early_stopping_rounds:
            early_stopped = True
            _log.info(
                "Early stopping triggered",
                extra={
                    "stopped_at_round": current_round,
                    "best_round": best_round,
                    "best_val_auc": best_val_auc,
                    "early_stopping_rounds": early_stopping_rounds,
                },
            )
            break

    # If early stopped, retrain with optimal number of estimators
    if early_stopped and best_round < current_round:
        _log.info(
            "Retraining with optimal estimators",
            extra={"best_round": best_round},
        )
        model = _build_classifier(best_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)
        actual_rounds = best_round
    else:
        actual_rounds = current_round

    # Model should never be None at this point (loop always runs at least once)
    if model is None:
        raise RuntimeError("Model not trained - n_estimators must be >= 1")

    # Final evaluation on all splits
    train_proba = _get_probabilities(model, splits.x_train, xgb_module)
    val_proba = _get_probabilities(model, splits.x_val, xgb_module)
    test_proba = _get_probabilities(model, splits.x_test, xgb_module)

    final_train_metrics = compute_all_metrics(splits.y_train, train_proba)
    final_val_metrics = compute_all_metrics(splits.y_val, val_proba)
    final_test_metrics = compute_all_metrics(splits.y_test, test_proba)

    _log.info(
        "Training complete",
        extra={
            "total_rounds_trained": actual_rounds,
            "early_stopped": early_stopped,
            "best_round": best_round,
            "train_metrics": format_metrics_str(final_train_metrics),
            "val_metrics": format_metrics_str(final_val_metrics),
            "test_metrics": format_metrics_str(final_test_metrics),
        },
    )

    # Save model
    model_id = str(uuid.uuid4())
    model_filename = f"covenant_model_{model_id[:8]}.ubj"
    model_path = output_dir / model_filename

    save_model(model, str(model_path))

    _log.info("Model saved", extra={"model_path": str(model_path)})

    # Extract feature importances
    importances = extract_feature_importances(model, feature_names)

    _log.info(
        "Feature importances extracted",
        extra={
            "top_features": [
                {"name": f["name"], "importance": f"{f['importance']:.4f}"} for f in importances[:3]
            ],
        },
    )

    return TrainOutcome(
        model_path=str(model_path),
        model_id=model_id,
        samples_total=splits.n_total,
        samples_train=splits.n_train,
        samples_val=splits.n_val,
        samples_test=splits.n_test,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        test_metrics=final_test_metrics,
        best_val_auc=best_val_auc,
        best_round=best_round,
        total_rounds=actual_rounds,
        early_stopped=early_stopped,
        config=config,
        feature_importances=importances,
        scale_pos_weight_computed=scale_pos_weight_computed,
    )


def train_regression_model_with_validation(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    config: TrainConfig,
    output_dir: Path,
    feature_names: list[str],
    progress_callback: Callable[[RegressionTrainProgress], None] | None = None,
) -> RegressionTrainOutcome:
    """Train XGBoost regressor with validation and early stopping.

    Parallel to train_model_with_validation for classification.
    Key differences:
    - objective='reg:squarederror', eval_metric='rmse'
    - No scale_pos_weight (regression has no class imbalance)
    - Early stopping on validation RMSE (lower is better)
    - Uses regression_split (random, not stratified)
    - Returns RegressionTrainOutcome with RegressionMetrics

    Args:
        x_features: Feature matrix (n_samples, n_features).
        y_targets: Continuous target values (n_samples,).
        config: Training configuration with hyperparameters.
        output_dir: Directory to save model artifacts.
        feature_names: List of feature names for importance reporting.
        progress_callback: Optional callback for progress updates.

    Returns:
        RegressionTrainOutcome with complete training results.
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBRegressorModuleProto = xgb
    regressor_factory: XGBRegressorFactory = xgb_module.XGBRegressor
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    splits = regression_split(
        x_features,
        y_targets,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_state=config["random_state"],
    )

    def _build_regressor(
        total_estimators: int,
    ) -> XGBRegressorModelProtocol:
        return regressor_factory(
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            n_estimators=total_estimators,
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            random_state=config["random_state"],
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=n_jobs,
            tree_method="hist",
            device=resolved_device,
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
        )

    n_estimators = config["n_estimators"]
    early_stopping_rounds = config["early_stopping_rounds"]

    # Track best model state (RMSE: lower is better)
    best_val_rmse = float("inf")
    best_round = 0
    rounds_no_improve = 0
    early_stopped = False

    model: XGBRegressorModelProtocol | None = None
    current_round = 0

    for current_round in range(1, n_estimators + 1):
        model = _build_regressor(current_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)

        # Evaluate on train and validation sets
        train_preds = _get_regression_predictions(
            model,
            splits.x_train,
        )
        val_preds = _get_regression_predictions(
            model,
            splits.x_val,
        )

        train_metrics = compute_all_regression_metrics(
            splits.y_train,
            train_preds,
        )
        val_metrics = compute_all_regression_metrics(
            splits.y_val,
            val_preds,
        )

        # Report progress
        if progress_callback is not None:
            progress_callback(
                RegressionTrainProgress(
                    round=current_round,
                    total_rounds=n_estimators,
                    train_rmse=train_metrics["rmse"],
                    val_rmse=val_metrics["rmse"],
                )
            )

        # Check for improvement (RMSE — lower is better)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_round = current_round
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # Log progress every 50 rounds
        if current_round % 50 == 0:
            _log.debug(
                "Regression training progress",
                extra={
                    "round": current_round,
                    "total_rounds": n_estimators,
                    "train_rmse": train_metrics["rmse"],
                    "val_rmse": val_metrics["rmse"],
                    "best_val_rmse": best_val_rmse,
                    "best_round": best_round,
                    "rounds_no_improve": rounds_no_improve,
                },
            )

        # Early stopping check
        if rounds_no_improve >= early_stopping_rounds:
            early_stopped = True
            _log.info(
                "Regression early stopping triggered",
                extra={
                    "stopped_at_round": current_round,
                    "best_round": best_round,
                    "best_val_rmse": best_val_rmse,
                    "early_stopping_rounds": early_stopping_rounds,
                },
            )
            break

    # If early stopped, retrain with optimal number of estimators
    if early_stopped and best_round < current_round:
        _log.info(
            "Retraining regressor with optimal estimators",
            extra={"best_round": best_round},
        )
        model = _build_regressor(best_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)
        actual_rounds = best_round
    else:
        actual_rounds = current_round

    if model is None:
        raise RuntimeError("Model not trained - n_estimators must be >= 1")

    # Final evaluation on all splits
    final_train_preds = _get_regression_predictions(
        model,
        splits.x_train,
    )
    final_val_preds = _get_regression_predictions(
        model,
        splits.x_val,
    )
    final_test_preds = _get_regression_predictions(
        model,
        splits.x_test,
    )

    final_train_metrics = compute_all_regression_metrics(
        splits.y_train,
        final_train_preds,
    )
    final_val_metrics = compute_all_regression_metrics(
        splits.y_val,
        final_val_preds,
    )
    final_test_metrics = compute_all_regression_metrics(
        splits.y_test,
        final_test_preds,
    )

    _log.info(
        "Regression training complete",
        extra={
            "total_rounds_trained": actual_rounds,
            "early_stopped": early_stopped,
            "best_round": best_round,
            "train_metrics": format_regression_metrics_str(
                final_train_metrics,
            ),
            "val_metrics": format_regression_metrics_str(
                final_val_metrics,
            ),
            "test_metrics": format_regression_metrics_str(
                final_test_metrics,
            ),
        },
    )

    # Save model
    model_id = str(uuid.uuid4())
    model_filename = f"covenant_reg_{model_id[:8]}.ubj"
    model_path = output_dir / model_filename

    save_model(model, str(model_path))

    _log.info(
        "Regression model saved",
        extra={"model_path": str(model_path)},
    )

    # Extract feature importances
    importances = extract_feature_importances(model, feature_names)

    _log.info(
        "Regression feature importances extracted",
        extra={
            "top_features": [
                {
                    "name": f["name"],
                    "importance": f"{f['importance']:.4f}",
                }
                for f in importances[:3]
            ],
        },
    )

    return RegressionTrainOutcome(
        model_path=str(model_path),
        model_id=model_id,
        samples_total=splits.n_total,
        samples_train=splits.n_train,
        samples_val=splits.n_val,
        samples_test=splits.n_test,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        test_metrics=final_test_metrics,
        best_val_rmse=best_val_rmse,
        best_round=best_round,
        total_rounds=actual_rounds,
        early_stopped=early_stopped,
        config=config,
        feature_importances=importances,
    )


def train_model(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
) -> XGBModelProtocol:
    """Train XGBoost classifier (simple API without validation).

    Auto-calculates scale_pos_weight if not provided.
    Prefer train_model_with_validation for production use.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        config: Training configuration

    Returns:
        Trained XGBClassifier model
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBModuleProto = xgb
    classifier_factory: XGBClassifierFactory = xgb_module.XGBClassifier
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    # Auto-calculate scale_pos_weight if not provided
    scale_pos_weight_computed = _compute_scale_pos_weight(y_labels, config.get("scale_pos_weight"))

    model = classifier_factory(
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=n_jobs,
        tree_method="hist",
        device=resolved_device,
        scale_pos_weight=scale_pos_weight_computed,
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
    )

    model.fit(x_features, y_labels)
    return model


def save_model(
    model: XGBModelProtocol | XGBRegressorModelProtocol,
    path: str,
) -> None:
    """Save trained model to file path.

    Works with both classifier and regressor XGBoost models.
    Uses get_booster().save_model() for XGBoost 3.x compatibility.
    """
    booster = model.get_booster()
    booster.save_model(path)


__all__ = [
    "DataSplits",
    "FeatureScaler",
    "PreprocessedDataSplits",
    "ProgressCallback",
    "RegressionDataSplits",
    "compute_feature_scaler",
    "extract_feature_importances",
    "preprocess_data_splits",
    "regression_split",
    "save_model",
    "stratified_split",
    "train_model",
    "train_model_with_validation",
    "train_regression_model_with_validation",
]
