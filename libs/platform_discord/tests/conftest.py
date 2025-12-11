"""Pytest configuration for platform_discord tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_discord.testing import reset_hooks


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset all test hooks after each test to prevent state leakage."""
    yield
    reset_hooks()
