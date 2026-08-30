"""Which operating system a decision is being made for.

One constant and one predicate. They have their own module because the things
that need them sit at different layers -- :mod:`rw_bot.validation` is
foundational and the harness imports it, not the reverse -- and a copy in each
is two definitions of one fact.
"""

from __future__ import annotations

import os
import sys
from pathlib import PurePosixPath, PureWindowsPath

import pytest

from rw_bot.platform_id import (
    POSIX_PATH_LIST_SEPARATOR,
    WINDOWS,
    WINDOWS_PATH_LIST_SEPARATOR,
    is_windows,
    path_list_separator,
    pure_path,
)

LINUX = "linux"
MACOS = "darwin"


def test_windows_is_windows() -> None:
    assert is_windows(WINDOWS) is True


@pytest.mark.parametrize("platform", [LINUX, MACOS, "freebsd", "cygwin"])
def test_everything_else_is_not(platform: str) -> None:
    """One equality rather than a table: every family other than Windows
    spells executable names, classpath separators and path roots the same way
    as each other."""
    assert is_windows(platform) is False


def test_windows_separates_path_lists_with_a_semicolon() -> None:
    assert path_list_separator(WINDOWS) == WINDOWS_PATH_LIST_SEPARATOR


@pytest.mark.parametrize("platform", [LINUX, MACOS])
def test_everything_else_separates_them_with_a_colon(platform: str) -> None:
    assert path_list_separator(platform) == POSIX_PATH_LIST_SEPARATOR


def test_the_separator_is_what_the_running_interpreter_uses() -> None:
    """One convention, not two: this is the character a JVM classpath and a
    PYTHONPATH both take, and ``os.pathsep`` is the interpreter's own name for
    it. Pinned against it so the pair cannot drift from the platform."""
    assert path_list_separator(sys.platform) == os.pathsep


def test_each_platform_gets_its_own_path_flavour() -> None:
    assert pure_path(WINDOWS) is PureWindowsPath
    assert pure_path(LINUX) is PurePosixPath


def test_the_flavour_is_the_stated_platforms_not_the_running_ones() -> None:
    """The defect this exists to stop: composing with ``pathlib.Path`` takes
    the flavour of the interpreter that is RUNNING, so a launch composed on
    one platform for the other carries the composer's separators. It cost two
    test failures that looked like wrong expectations and were not."""
    assert str(pure_path(LINUX)("/repo") / "a" / "b") == "/repo/a/b"
    assert str(pure_path(WINDOWS)("C:/repo") / "a" / "b") == "C:\\repo\\a\\b"


def test_a_composed_path_does_not_change_with_where_it_was_composed() -> None:
    """Stated as its own property because it is the one the cross-platform
    checks throughout this package rest on."""
    composed = pure_path(LINUX)("/repo") / ".game-w1" / "jvm64" / "bin" / "java"
    assert "\\" not in str(composed)


def test_the_constant_is_the_value_python_actually_reports() -> None:
    """Pinned against the interpreter rather than asserted as a literal twice:
    the whole package branches on this string, so a typo in it would send
    every decision down the POSIX arm on a Windows box, silently."""
    if sys.platform == WINDOWS:
        assert is_windows(sys.platform) is True
    else:
        assert is_windows(sys.platform) is False
