"""Pytest configuration and fixtures for cleargbm tests.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from cleargbm.types import GradientBoostingConfig


def make_config(
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
) -> GradientBoostingConfig:
    """Create a test config."""
    return GradientBoostingConfig(
        n_estimators=10,
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
    )


@pytest.fixture(autouse=True)
def reset_test_hooks() -> Generator[None, None, None]:
    """Reset test hooks after each test.

    Yields:
        None (test runs during yield).
    """
    from cleargbm import _test_hooks

    # Store original factory
    original_factory = _test_hooks._random_state_factory

    yield

    # Restore original factory
    _test_hooks._random_state_factory = original_factory
