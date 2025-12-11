"""Pytest configuration for platform_ml tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_ml import _test_hooks
from platform_ml.testing import reset_hooks as reset_wandb_hooks


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset all test hooks after each test to prevent state leakage."""
    yield
    _test_hooks.reset_hooks()
    reset_wandb_hooks()
