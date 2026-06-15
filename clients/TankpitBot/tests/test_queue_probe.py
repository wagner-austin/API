"""Tests for scripts.queue_probe module."""

from __future__ import annotations

import runpy
import sys
import warnings
from collections.abc import Generator

import pytest
from scripts import _test_hooks
from scripts.queue_probe import main


@pytest.fixture(autouse=True)
def _isolate_hooks() -> Generator[None, None, None]:
    """Save and restore hooks around each test."""
    orig_setup = _test_hooks.setup_rich_logging
    yield
    _test_hooks.setup_rich_logging = orig_setup


def test_main_returns_zero() -> None:
    """main() returns 0."""
    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    result = main()
    assert result == 0


def test_main_module_entry() -> None:
    """Running as __main__ invokes main() and exits 0."""

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        pass

    _test_hooks.setup_rich_logging = _fake_setup
    sys.modules.pop("scripts.queue_probe", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.queue_probe", run_name="__main__")
    assert exc_info.value.code == 0
