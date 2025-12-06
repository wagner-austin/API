from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Protocol

import pytest


class _GuardModule(Protocol):
    def main(self, argv: list[str]) -> int: ...


def _load_guard_module() -> _GuardModule:
    spec = importlib.util.spec_from_file_location("turkic_api_scripts_guard", "scripts/guard.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("guard module spec could not be loaded")
    mod: ModuleType = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert hasattr(loader, "exec_module")
    loader.exec_module(mod)
    guard_mod: _GuardModule = mod
    return guard_mod


def test_guard_main_verbose_and_unknown_token(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    guard = _load_guard_module()
    code = guard.main(["--verbose", "--root", str(tmp_path), "unknown-flag"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Guard rule summary:" in out
    assert "guard_exit_code code=0" in out
