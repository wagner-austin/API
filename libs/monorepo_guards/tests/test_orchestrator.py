from __future__ import annotations

from pathlib import Path

from monorepo_guards.config import GuardConfig
from monorepo_guards.orchestrator import _run_with_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_orchestrator_reports_violations_and_nonzero(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "server" / "model_trainer"
    bad = src / "bad.py"
    any_kw = "An" + "y"
    ti = "# " + "type" + ": " + "ignore"
    code = (
        f"from typing import {any_kw}\n"
        f"x: {any_kw} = 1  {ti}\n"
        "import logging\n"
        "pri" + "nt('x')\n"
        "logging.basic" + "Config(level=10)\n"
        "try:\n"
        "    1/0\n"
        "except Exception:\n"
        "    pass\n"
    )
    _write(bad, code)

    cfg = GuardConfig(
        root=root,
        directories=("server/model_trainer",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    rc = _run_with_config(cfg)
    assert rc != 0


def test_orchestrator_pass_no_files(tmp_path: Path) -> None:
    # No configured directories exist; should pass with zero violations
    cfg = GuardConfig(
        root=tmp_path,
        directories=("nonexistent",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        # include a segment to exercise dataclass-rule inclusion path
        dataclass_ban_segments=(("should", "trigger"),),
    )
    rc = _run_with_config(cfg)
    assert rc == 0


def test_orchestrator_truncates_long_line(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "src"
    bad = src / "bad.py"
    long_tail = "x" * 300
    code = "# TO" + "DO " + long_tail + "\n"
    _write(bad, code)

    cfg = GuardConfig(
        root=root,
        directories=("src",),
        exclude_parts=(".venv", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"),
        forbid_pyi=True,
        allow_print_in_tests=False,
        dataclass_ban_segments=(),
    )
    rc = _run_with_config(cfg)
    assert rc != 0
