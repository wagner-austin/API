"""Pytest configuration for platform_ml tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_ml import torch_types
from platform_ml.testing import reset_hooks as reset_wandb_hooks


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset all test hooks after each test to prevent state leakage."""
    original_import_torch = torch_types._import_torch
    yield
    torch_types._import_torch = original_import_torch
    reset_wandb_hooks()
