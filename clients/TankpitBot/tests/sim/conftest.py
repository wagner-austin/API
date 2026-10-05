"""Fixtures shared by the sim tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tankpit_bot import _test_hooks
from tests.sim._fake_db import FakeDatabase, scripted_env


@pytest.fixture()
def fake_db() -> Generator[FakeDatabase, None, None]:
    """A scripted database behind ``connect_database``, its string in the test variable.

    Yields:
        The database.
    """
    database = FakeDatabase()
    real_connect, real_env = _test_hooks.connect_database, _test_hooks.get_env
    _test_hooks.connect_database = database.connect
    _test_hooks.get_env = scripted_env
    yield database
    _test_hooks.connect_database = real_connect
    _test_hooks.get_env = real_env
