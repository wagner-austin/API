"""Shared test fixtures for platform_email tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_email.testing import hooks


@pytest.fixture(autouse=True)
def _reset_hooks_fixture() -> Generator[None, None, None]:
    """Reset all hooks before and after each test.

    Called on the container rather than as a bare `reset_hooks()` so the
    isolation is attributable to `hooks`: tests below this conftest may assign
    `hooks.<attr>` knowing every attribute is restored around each test.
    """
    hooks.reset()
    yield
    hooks.reset()
