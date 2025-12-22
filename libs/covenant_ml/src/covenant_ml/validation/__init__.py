"""Cross-validation module for model evaluation.

Provides stratified k-fold cross-validation with preprocessing isolation
to prevent data leakage. Each fold fits preprocessing on training data only.

Key components:
- stratified_kfold_split: Create stratified train/val splits
- group_stratified_kfold_split: Create group-aware splits (no entity leakage)
- run_cross_validation: Execute k-fold CV with a trainer function
- run_group_cross_validation: Execute group-aware CV (e.g., by customer_ID)
- OOF utilities: Work with out-of-fold predictions for stacking

Example:
    >>> from covenant_ml.validation import run_cross_validation, compute_oof_metrics
    >>> cv_result = run_cross_validation(x, y, n_folds=5, random_state=42, trainer=fn)
    >>> # Or use group-aware CV for time-series (no customer leakage)
    >>> cv_result = run_group_cross_validation(x, y, groups, n_folds=5, ...)
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
    "FoldResult",
    "FoldTrainer",
    "OOFMetrics",
    "TrainedModel",
    "combine_oof_predictions",
    "compute_oof_auc",
    "compute_oof_metrics",
    "get_fold_data",
    "get_oof_for_stacking",
    "group_stratified_kfold_split",
    "run_cross_validation",
    "run_group_cross_validation",
    "stratified_kfold_split",
    "validate_oof_coverage",
]
