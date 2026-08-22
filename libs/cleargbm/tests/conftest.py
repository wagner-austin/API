"""Pytest configuration and fixtures for cleargbm tests."""

from __future__ import annotations

from cleargbm.types import GradientBoostingConfig, GrowthStrategy


def make_config(
    n_estimators: int = 10,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: int | None = None,
    max_bins: int = 64,
    subsample: float = 1.0,
    random_state: int = 42,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    n_jobs: int = 1,
    early_stopping_rounds: int | None = None,
    growth_strategy: GrowthStrategy = "depth_wise",
    num_leaves: int | None = None,
) -> GradientBoostingConfig:
    """Create a test config.

    Args:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Max features per split (None = all).
        max_bins: Number of histogram bins.
        subsample: Row subsampling ratio.
        random_state: Random seed.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        n_jobs: Number of parallel workers.
        early_stopping_rounds: Rounds without improvement before stopping (None = disabled).
        growth_strategy: Tree growth policy.
        num_leaves: Leaf budget, required under leaf-wise growth.

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
        max_bins=max_bins,
        subsample=subsample,
        random_state=random_state,
        track_contributions=True,
        monotonic_constraints=None,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy=growth_strategy,
        num_leaves=num_leaves,
        scale_pos_weight=1.0,
    )


# The former ``reset_test_hooks`` autouse fixture reset a Python-side random-
# state factory hook on ``_hooks_infra``. That module and its hook mechanism
# no longer exist — the Rust training loop owns all randomness end-to-end —
# so there is nothing left to reset between tests.
