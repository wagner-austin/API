from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from monorepo_guards.tests_rules import PolicyTestsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_tests_rule_flags_simplenamespace_in_tests(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = root / "pkg" / "tests"
    path = tests_dir / "mod.py"
    tok = "Simple" + "Namespace"
    paren = "("
    _write(path, f"from types import {tok}\nns = {tok}{paren})\n")

    rule = PolicyTestsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "tests-simplenamespace" in kinds


def test_tests_rule_flags_qualified_simplenamespace(tmp_path: Path) -> None:
    """Ensure qualified types module usage is also flagged."""
    root = tmp_path
    tests_dir = root / "pkg" / "tests"
    path = tests_dir / "mod.py"
    tok = "Simple" + "Namespace"
    paren = "("
    _write(path, f"import types\nns = types.{tok}{paren})\n")

    rule = PolicyTestsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "tests-simplenamespace" in kinds


def test_tests_rule_ignores_non_tests_paths(tmp_path: Path) -> None:
    root = tmp_path
    src_dir = root / "pkg" / "src"
    path = src_dir / "mod.py"
    tok = "Simple" + "Namespace"
    _write(path, f"from types import {tok}\n")

    rule = PolicyTestsRule()
    violations = rule.run([path])
    assert not violations


def test_tests_rule_raises_on_os_error(tmp_path: Path) -> None:
    root = tmp_path
    tests_dir = root / "pkg" / "tests"
    path = tests_dir / "bad.py"
    _write(path, "x = 1\n")

    rule = PolicyTestsRule()
    with (
        patch.object(Path, "read_text", side_effect=OSError("Read error")),
        pytest.raises(RuntimeError, match=r"failed to read.*bad\.py"),
    ):
        rule.run([path])
