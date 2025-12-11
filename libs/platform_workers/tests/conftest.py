"""Pytest configuration for platform_workers tests.

This module sets up test hooks before each test to ensure no test pollution
and provides shared fixtures.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_workers.testing import hooks


@pytest.fixture(autouse=True)
def reset_hooks() -> Generator[None, None, None]:
    """Reset all hooks before and after each test."""
    hooks.reset()
    yield
    hooks.reset()
