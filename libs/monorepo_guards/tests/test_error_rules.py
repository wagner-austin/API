from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.error_rules import ErrorsRule


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_errors_rule_flags_local_errors_module(tmp_path: Path) -> None:
    path = tmp_path / "service" / "errors.py"
    _write(path, "# local errors module\n")

    rule = ErrorsRule()
    violations = rule.run([path])
    assert len(violations) == 1
    assert violations[0].kind == "local-errors-module"


def test_errors_rule_skips_platform_core_errors_module(tmp_path: Path) -> None:
    path = tmp_path / "platform_core" / "src" / "platform_core" / "errors.py"
    _write(path, "# platform core errors module\n")

    rule = ErrorsRule()
    violations = rule.run([path])
    assert len(violations) == 0


def test_errors_rule_flags_app_error_definition(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app" / "foo.py"
    _write(path, "class AppError(Exception):\n    ...\n")

    rule = ErrorsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "local-app-error" in kinds
    assert all(v.file == path for v in violations)


def test_errors_rule_flags_error_code_and_module(tmp_path: Path) -> None:
    path = tmp_path / "src" / "app" / "errors" / "base.py"
    _write(path, "from enum import Enum\nclass ErrorCode(Enum):\n    BAD = 'BAD'\n")

    rule = ErrorsRule()
    violations = rule.run([path])
    kinds = {v.kind for v in violations}
    assert "local-errors-module" in kinds
    assert "local-error-code" in kinds


def test_errors_rule_raises_on_invalid_syntax(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    _write(path, "class AppError(Exception\n")

    rule = ErrorsRule()
    with pytest.raises(RuntimeError, match=r"failed to parse"):
        rule.run([path])
