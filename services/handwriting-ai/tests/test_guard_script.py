from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pytest
import scripts.guard as guard


class _RunCall(TypedDict):
    monorepo_root: Path
    project_root: Path


def test_find_monorepo_root_locates_libs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    libs = root / "libs"
    libs.mkdir(parents=True, exist_ok=True)
    start = libs / "some" / "deep"
    start.mkdir(parents=True, exist_ok=True)

    found = guard._find_monorepo_root(start)
    assert found == root


def test_find_monorepo_root_raises_when_missing(tmp_path: Path) -> None:
    start = tmp_path / "no_libs"
    start.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _ = guard._find_monorepo_root(start)


def test_load_orchestrator_imports_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    libs_dir = tmp_path / "libs"
    mg_src = libs_dir / "monorepo_guards" / "src" / "monorepo_guards"
    mg_src.mkdir(parents=True, exist_ok=True)
    (mg_src / "__init__.py").write_text("", encoding="utf-8")
    # Write orchestrator that stores call info in a file and returns int
    (mg_src / "orchestrator.py").write_text(
        "from pathlib import Path\n"
        "def run_for_project(monorepo_root, project_root):\n"
        "    (monorepo_root / 'call_info.txt').write_text("
        "f'{project_root}', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    run = guard._load_orchestrator(tmp_path)
    result = run(monorepo_root=tmp_path, project_root=tmp_path / "proj")
    assert result == 0
    call_info = (tmp_path / "call_info.txt").read_text(encoding="utf-8")
    assert call_info == str(tmp_path / "proj")


def test_main_invokes_run_and_supports_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 3

    def _fake_find_monorepo_root(start: Path) -> Path:
        return start

    def _fake_load_orchestrator(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find_monorepo_root)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load_orchestrator)

    project_root = Path(__file__).resolve().parents[1]

    rc = guard.main(["--root", str(tmp_path), "-v"])
    assert rc == 3
    assert len(calls) == 1
    assert calls[0]["monorepo_root"] == project_root
    assert calls[0]["project_root"] == tmp_path


def test_main_uses_default_args_when_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 0

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    rc = guard.main(None)
    assert rc == 0
    assert calls[0]["project_root"].name == "handwriting-ai"


def test_main_skips_unknown_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[_RunCall] = []

    def _fake_run_for_project(*, monorepo_root: Path, project_root: Path) -> int:
        calls.append({"monorepo_root": monorepo_root, "project_root": project_root})
        return 7

    def _fake_find(start: Path) -> Path:
        return start

    def _fake_load(_: Path) -> guard._RunForProject:
        return _fake_run_for_project

    monkeypatch.setattr(guard, "_find_monorepo_root", _fake_find)
    monkeypatch.setattr(guard, "_load_orchestrator", _fake_load)

    rc = guard.main(["--unknown", "--root", str(tmp_path)])
    assert rc == 7
    assert len(calls) == 1
    assert calls[0]["project_root"] == tmp_path
