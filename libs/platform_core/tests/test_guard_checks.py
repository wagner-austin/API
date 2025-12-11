from __future__ import annotations

import io
import sys
from pathlib import Path

from pytest import raises
from scripts import guard as guard_mod


def test_guard_main_entry_no_violations(tmp_path: Path) -> None:
    rc = guard_mod.main(["--root", str(tmp_path)])
    assert rc in (0, 2)


def test_guard_main_unknown_flag_is_ignored(tmp_path: Path) -> None:
    rc = guard_mod.main(["--root", str(tmp_path), "ignored-flag"])
    assert rc in (0, 2)


def test_guard_main_verbose_flag_prints_exit_code(tmp_path: Path) -> None:
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = guard_mod.main(["--root", str(tmp_path), "--verbose"])
        output = sys.stdout.getvalue()
        assert rc in (0, 2)
        assert f"guard_exit_code code={rc}\n" in output
    finally:
        sys.stdout = old_stdout


def test_guard_find_monorepo_root_raises_without_libs(tmp_path: Path) -> None:
    def _always_false(_: Path) -> bool:
        return False

    guard_mod._is_dir = _always_false

    with raises(RuntimeError, match="monorepo root with 'libs' directory not found"):
        guard_mod._find_monorepo_root(tmp_path)
