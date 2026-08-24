"""Pytest configuration and fixtures for cleargbm tests."""

from __future__ import annotations

from cleargbm.types import GradientBoostingConfig, GrowthStrategy, Objective


def make_config(
    n_estimators: int = 10,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | None = None,
    colsample_bytree: float | None = None,
    categorical_features: tuple[int, ...] | None = None,
    n_classes: int | None = None,
    lambdarank_truncation_level: int | None = None,
    goss_top_rate: float | None = None,
    goss_other_rate: float | None = None,
    quantized_gradient_bins: int | None = None,
    max_bins: int = 64,
    subsample: float = 1.0,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    n_jobs: int = 1,
    early_stopping_rounds: int | None = None,
    growth_strategy: GrowthStrategy = "depth_wise",
    num_leaves: int | None = None,
    objective: Objective = "binary_log_loss",
    scale_pos_weight: float | None = 1.0,
) -> GradientBoostingConfig:
    """Create a test config.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Max features per split (None = all).
        colsample_bytree: Fraction of features per tree (None = all).
        categorical_features: Feature indices treated as categorical (None = all numeric).
        n_classes: Class count, required (>= 2) under ``"multiclass_softmax"``,
            ``None`` under every other objective.
        lambdarank_truncation_level: NDCG truncation position, required
            (>= 1) under ``"lambdarank"``, ``None`` otherwise.
        goss_top_rate: GOSS top rate, paired with goss_other_rate.
        goss_other_rate: GOSS other rate, paired with goss_top_rate.
        quantized_gradient_bins: Quantized-training bin count (None = float
            histograms); even, in [2, 126].
        max_bins: Number of histogram bins.
        subsample: Row subsampling ratio.
        random_state: Random seed.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        n_jobs: Number of parallel workers.
        early_stopping_rounds: Rounds without improvement before stopping (None = disabled).
        growth_strategy: Tree growth policy.
        num_leaves: Leaf budget, required under leaf-wise growth.
        objective: Training objective.
        scale_pos_weight: Positive-class weight; must be ``None`` under
            ``"squared_error"``.

    Returns:
        Test configuration.
    """
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        colsample_bytree=colsample_bytree,
        categorical_features=categorical_features,
        n_classes=n_classes,
        lambdarank_truncation_level=lambdarank_truncation_level,
        goss_top_rate=goss_top_rate,
        goss_other_rate=goss_other_rate,
        quantized_gradient_bins=quantized_gradient_bins,
        max_bins=max_bins,
        subsample=subsample,
        random_state=random_state,
        monotonic_constraints=None,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy=growth_strategy,
        num_leaves=num_leaves,
        objective=objective,
        scale_pos_weight=scale_pos_weight,
    )


def make_regression_config(
    n_estimators: int = 10,
    max_depth: int = 3,
    early_stopping_rounds: int | None = None,
) -> GradientBoostingConfig:
    """Create a test squared-error regression config.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        early_stopping_rounds: Rounds without improvement before stopping (None = disabled).

    Returns:
        Test configuration with the regression objective pairing.
    """
    return make_config(
        n_estimators=n_estimators,
        max_depth=max_depth,
        early_stopping_rounds=early_stopping_rounds,
        objective="squared_error",
        scale_pos_weight=None,
    )


# The former ``reset_test_hooks`` autouse fixture reset a Python-side random-
# state factory hook on ``_hooks_infra``. That module and its hook mechanism
# no longer exist — the Rust training loop owns all randomness end-to-end —
# so there is nothing left to reset between tests.
