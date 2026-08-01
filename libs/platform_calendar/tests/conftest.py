"""Shared test fixtures for platform_calendar tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_calendar.testing import hooks


@pytest.fixture(autouse=True)
def _reset_hooks_fixture() -> Generator[None, None, None]:
    """Reset all hooks before and after each test."""
    hooks.reset()
    yield
    hooks.reset()
