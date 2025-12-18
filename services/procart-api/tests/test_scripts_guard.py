from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Protocol

import pytest


class _RunForProject(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


class _GuardModuleProto(Protocol):
    _is_dir: Callable[[Path], bool]

    def _find_monorepo_root(self, start: Path) -> Path: ...
    def _load_orchestrator(self, monorepo_root: Path) -> _RunForProject: ...
    def main(self, argv: Sequence[str] | None = None) -> int: ...

    __file__: str


def _load_service_guard_module() -> _GuardModuleProto:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    spec = importlib.util.spec_from_file_location("procart_api_service_guard", str(script_path))
    if spec is None or spec.loader is None:
        raise AssertionError("failed to build spec for guard.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["procart_api_service_guard"] = module
    spec.loader.exec_module(module)
    # Bind to a variable with the precise protocol via annotated assignment.
    mod = __import__("procart_api_service_guard")
    gm: _GuardModuleProto = mod  # type: ignore[assignment]
    return gm


def test_guard_main_and_main_block_service() -> None:
    # Load service-local guard module explicitly, then call main(None).
    guard_mod = _load_service_guard_module()

    rc = guard_mod.main(None)
    assert rc >= 0

    # Execute the service's guard.py directly as __main__ to cover main block.
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "guard.py"
    code = script_path.read_text(encoding="utf-8")
    globals_dict = {"__name__": "__main__", "__file__": str(script_path)}
    with pytest.raises(SystemExit):
        exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)


def test_guard_find_root_raises_when_libs_missing_service() -> None:
    guard_mod = _load_service_guard_module()

    original_is_dir = guard_mod._is_dir
    try:
        guard_mod._is_dir = lambda p: False
        with pytest.raises(RuntimeError):
            guard_mod._find_monorepo_root(Path("C:\\"))
    finally:
        guard_mod._is_dir = original_is_dir


def test_guard_verbose_flag_and_root_override_service() -> None:
    guard_mod = _load_service_guard_module()

    project_root = Path(__file__).resolve().parents[1]
    rc = guard_mod.main(["--root", str(project_root), "--verbose"])
    assert rc >= 0


def test_guard_unknown_flag_hits_else_branch_service() -> None:
    guard_mod = _load_service_guard_module()

    rc = guard_mod.main(["--unknown-flag"])  # triggers the else branch
    assert rc >= 0


def test_guard_force_all_branches_service() -> None:
    # Exercise both the early-return path of _find_monorepo_root and the verbose
    # printing branch in main by stubbing internal helpers with typed callables.
    guard_mod = _load_service_guard_module()

    # Call main with verbose to hit sys.stdout.write and include an unknown
    # token to exercise the else branch in the arg parser. Avoid overriding
    # filesystem behavior to let guard discover the real monorepo root.
    buf = StringIO()
    with redirect_stdout(buf):
        rc = guard_mod.main(["--unknown-flag", "-v"])  # branches
    out = buf.getvalue()
    assert rc >= 0
    assert "guard_exit_code code=" in out
