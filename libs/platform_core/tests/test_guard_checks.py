from __future__ import annotations

import io
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from pytest import raises
from scripts import guard as guard_mod


@pytest.fixture()
def restore_guard_is_dir() -> Generator[None, None, None]:
    """Restore ``guard_mod._is_dir`` after a test overrides the injection seam.

    Without this, a test that swaps the seam leaks it into every later test in
    the same worker process.

    Yields:
        None, for the duration of the test.
    """
    original = guard_mod._is_dir
    yield
    guard_mod._is_dir = original


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    rc = guard_mod.main(["--root", str(tmp_path)])
    assert rc == 0


def test_guard_main_unknown_flag_is_ignored(tmp_path: Path) -> None:
    rc = guard_mod.main(["--root", str(tmp_path), "ignored-flag"])
    assert rc == 0


def test_guard_main_verbose_flag_prints_exit_code(tmp_path: Path) -> None:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = guard_mod.main(["--root", str(tmp_path), "--verbose"])
        output = sys.stdout.getvalue()
        assert rc == 0
        assert "guard_exit_code code=0\n" in output
    finally:
        sys.stdout = old_stdout


def test_guard_find_monorepo_root_raises_without_libs(
    tmp_path: Path,
    restore_guard_is_dir: None,
) -> None:
    del restore_guard_is_dir

    def _always_false(_: Path) -> bool:
        return False

    guard_mod._is_dir = _always_false

    with raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard_mod._find_monorepo_root(tmp_path)
