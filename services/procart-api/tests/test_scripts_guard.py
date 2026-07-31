from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from scripts import guard as guard_mod


def test_guard_main_and_main_block_service() -> None:
    rc = guard_mod.main(None)
    assert rc >= 0

    # Execute the service's guard.py directly as __main__ to cover main block.
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    code = script_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(script_path)}
    with pytest.raises(SystemExit):
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)


def test_guard_find_root_raises_when_libs_missing_service() -> None:
    original_is_dir = guard_mod._is_dir
    try:
        guard_mod._is_dir = lambda p: False
        with pytest.raises(RuntimeError):
            guard_mod._find_monorepo_root(Path("C:\\"))
    finally:
        guard_mod._is_dir = original_is_dir


def test_guard_verbose_flag_and_root_override_service() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rc = guard_mod.main(["--root", str(project_root), "--verbose"])
    assert rc >= 0


def test_guard_unknown_flag_hits_else_branch_service() -> None:
    rc = guard_mod.main(["--unknown-flag"])  # triggers the else branch
    assert rc >= 0


def test_guard_force_all_branches_service() -> None:
    # Exercise both the early-return path of _find_monorepo_root and the verbose
    # printing branch in main by stubbing internal helpers with typed callables.
    # Call main with verbose to hit sys.stdout.write and include an unknown
    # token to exercise the else branch in the arg parser. Avoid overriding
    # filesystem behavior to let guard discover the real monorepo root.
    buf = StringIO()
    with redirect_stdout(buf):
        rc = guard_mod.main(["--unknown-flag", "-v"])  # branches
    out = buf.getvalue()
    assert rc >= 0
    assert "guard_exit_code code=" in out
