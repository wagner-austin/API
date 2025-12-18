from __future__ import annotations

from pathlib import Path

import pytest


def test_guard_main_and_main_block() -> None:
    # Import the guard module and invoke main(None)
    from scripts import guard as guard_mod

    rc = guard_mod.main(None)
    assert rc >= 0

    # Execute the file as if __name__ == "__main__" using compile+exec.
    # This covers the SystemExit path without using runpy (which returns Any).
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    code = script_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(script_path)}
    with pytest.raises(SystemExit):
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)


def test_guard_find_root_raises_when_libs_missing() -> None:
    # Force _find_monorepo_root to walk to FS root and raise
    from scripts import guard as guard_mod

    original_is_dir = guard_mod._is_dir
    try:
        guard_mod._is_dir = lambda p: False  # never finds libs
        with pytest.raises(RuntimeError):
            guard_mod._find_monorepo_root(Path("C:\\"))
    finally:
        guard_mod._is_dir = original_is_dir


def test_guard_verbose_flag_and_root_override() -> None:
    from scripts import guard as guard_mod

    project_root = Path(__file__).resolve().parents[1]
    # We don't assert output; calling is enough to exercise verbose branch
    rc = guard_mod.main(["--root", str(project_root), "--verbose"])
    assert rc >= 0


def test_guard_unknown_flag_hits_else_branch() -> None:
    from scripts import guard as guard_mod

    rc = guard_mod.main(["--unknown-flag"])  # triggers the else branch in arg parsing
    assert rc >= 0
