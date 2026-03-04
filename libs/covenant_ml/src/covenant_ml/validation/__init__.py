"""Cross-validation module for model evaluation.

Provides stratified k-fold cross-validation with preprocessing isolation
to prevent data leakage. Each fold fits preprocessing on training data only.

Key components:
- CVSplitterProtocol: Protocol for pluggable CV strategies
- CVSplitterRegistry: Registry for CV strategy implementations
- stratified_kfold_split: Create stratified train/val splits
- group_stratified_kfold_split: Create group-aware splits (no entity leakage)
- run_cross_validation: Execute k-fold CV with a trainer function
- run_group_cross_validation: Execute group-aware CV (e.g., by customer_ID)
- OOF utilities: Work with out-of-fold predictions for stacking

Example:
    >>> from covenant_ml.validation import (
    ...     run_cross_validation,
    ...     compute_oof_metrics,
    ...     default_cv_registry,
    ... )
    >>> # Using registry for pluggable strategies
    >>> registry = default_cv_registry()
    >>> splitter = registry.get("stratified_kfold")
    >>> splits = splitter.split(y, n_folds=5, random_state=42)
    >>> # Or use direct functions
    >>> cv_result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=fn)
    >>> oof_metrics = compute_oof_metrics(y, cv_result)
    >>> oof_auc = oof_metrics['oof_auc']
"""

from covenant_ml.validation.oof import (
    OOFMetrics,
    combine_oof_predictions,
    compute_oof_auc,
    compute_oof_metrics,
    get_oof_for_stacking,
    validate_oof_coverage,
)
from covenant_ml.validation.protocol import (
    CVSplitterFactory,
    CVSplitterProtocol,
    CVStrategyCapabilities,
    CVStrategyName,
)
from covenant_ml.validation.registry import (
    CVSplitterRegistration,
    CVSplitterRegistry,
    default_cv_registry,
)
from covenant_ml.validation.regression_runner import (
    FoldRegressorTrainer,
    TrainedRegressor,
    get_regression_fold_data,
    kfold_split,
    run_regression_cross_validation,
)
from covenant_ml.validation.regression_types import (
    RegressionCVResult,
    RegressionFoldResult,
)
from covenant_ml.validation.runner import (
    FoldTrainer,
    TrainedModel,
    run_cross_validation,
    run_group_cross_validation,
)
from covenant_ml.validation.splitter import (
    get_fold_data,
    group_stratified_kfold_split,
    stratified_kfold_split,
)
from covenant_ml.validation.strategies import (
    GroupStratifiedKFoldSplitter,
    ShuffleSplitSplitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    create_group_stratified_kfold_splitter,
    create_shuffle_split_splitter,
    create_stratified_kfold_splitter,
    create_time_series_splitter,
)
from covenant_ml.validation.types import (
    CVResult,
    CVSplit,
    CVSplitInfo,
    FoldResult,
)

__all__ = [
    "CVResult",
    "CVSplit",
    "CVSplitInfo",
    "CVSplitterFactory",
    "CVSplitterProtocol",
    "CVSplitterRegistration",
    "CVSplitterRegistry",
    "CVStrategyCapabilities",
    "CVStrategyName",
    "FoldRegressorTrainer",
    "FoldResult",
    "FoldTrainer",
    "GroupStratifiedKFoldSplitter",
    "OOFMetrics",
    "RegressionCVResult",
    "RegressionFoldResult",
    "ShuffleSplitSplitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
    "TrainedModel",
    "TrainedRegressor",
    "combine_oof_predictions",
    "compute_oof_auc",
    "compute_oof_metrics",
    "create_group_stratified_kfold_splitter",
    "create_shuffle_split_splitter",
    "create_stratified_kfold_splitter",
    "create_time_series_splitter",
    "default_cv_registry",
    "get_fold_data",
    "get_oof_for_stacking",
    "get_regression_fold_data",
    "group_stratified_kfold_split",
    "kfold_split",
    "run_cross_validation",
    "run_group_cross_validation",
    "run_regression_cross_validation",
    "stratified_kfold_split",
    "validate_oof_coverage",
]
