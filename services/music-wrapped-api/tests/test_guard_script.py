from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

from pytest import MonkeyPatch
from scripts import guard as guard_script


def test_find_monorepo_root_success(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    app_dir = root / "services" / "music-wrapped-api"
    (root / "libs").mkdir(parents=True)
    app_dir.mkdir(parents=True)
    out = guard_script._find_monorepo_root(app_dir)
    assert out == root


def test_find_monorepo_root_failure(tmp_path: Path) -> None:
    start = tmp_path / "no-root" / "app"
    start.mkdir(parents=True)
    try:
        guard_script._find_monorepo_root(start)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when libs dir not found")


def test_load_orchestrator_from_tmp(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    libs = tmp_path / "libs"
    pkg_dir = libs / "monorepo_guards" / "src" / "monorepo_guards"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "orchestrator.py").write_text(
        "def run_for_project(*, monorepo_root, project_root):\n    return 7\n"
    )
    # Ensure a clean import path
    sys.modules.pop("monorepo_guards.orchestrator", None)
    run_for_project = guard_script._load_orchestrator(tmp_path)
    code = run_for_project(monorepo_root=tmp_path, project_root=tmp_path / "services")
    assert code == 7


class _RunnerProto(Protocol):
    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int: ...


def test_main_parses_args_and_invokes_runner(monkeypatch: MonkeyPatch) -> None:
    # Stub runner to capture inputs
    called: dict[str, str] = {}

    def _runner(*, monorepo_root: Path, project_root: Path) -> int:
        called["root"] = str(monorepo_root)
        called["project"] = str(project_root)
        return 3

    def _load(_: Path) -> _RunnerProto:
        def _adapter(*, monorepo_root: Path, project_root: Path) -> int:
            return _runner(monorepo_root=monorepo_root, project_root=project_root)

        return _adapter

    monkeypatch.setattr(guard_script, "_load_orchestrator", _load)

    # Use explicit root override, verbose, and unknown token to hit all branches
    proj_root = Path(guard_script.__file__).resolve().parents[1]
    args = ["--root", str(proj_root), "--verbose", "--unknown"]
    rc = guard_script.main(args)
    assert rc == 3
    assert called["project"] == str(proj_root)

    # No verbose flag; should not print
    args2: list[str] = []
    rc2 = guard_script.main(args2)
    assert rc2 == 3
