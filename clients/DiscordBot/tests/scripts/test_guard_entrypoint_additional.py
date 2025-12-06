from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Protocol

import pytest


class _GuardModule(Protocol):
    def main(self, argv: list[str]) -> int: ...
    def _find_monorepo_root(self, start: Path) -> Path: ...


def _load_guard_module() -> _GuardModule:
    spec = importlib.util.spec_from_file_location("discordbot_scripts_guard", "scripts/guard.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    guard: _GuardModule = mod
    return guard


def test_find_monorepo_root_raises(tmp_path: Path) -> None:
    guard = _load_guard_module()
    with pytest.raises(RuntimeError):
        _ = guard._find_monorepo_root(tmp_path)


def test_guard_main_non_verbose_returns_code(tmp_path: Path) -> None:
    guard = _load_guard_module()
    code = guard.main(["--root", str(tmp_path)])
    assert type(code) is int


def test_guard_runs_as_main() -> None:
    # Execute the script under __main__ to cover the guard block
    with pytest.raises(SystemExit):
        import runpy

        runpy.run_path("scripts/guard.py", run_name="__main__")
