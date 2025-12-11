from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_music.testing import reset_hooks as _reset_hooks


@pytest.fixture(autouse=True)
def reset_hooks_fixture() -> Generator[None, None, None]:
    """Reset all test hooks to production defaults before and after each test."""
    _reset_hooks()
    yield
    _reset_hooks()
