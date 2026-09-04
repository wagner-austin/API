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


def test_errors_rule_skips_the_split_out_code_vocabulary(tmp_path: Path) -> None:
    """`platform_core/error_codes.py` is central, not local.

    The vocabulary was split out of `errors.py` on 2026-09-04 when that file
    reached the 600-line ceiling. The predicate previously spelled "central"
    as the single filename `errors.py`, which made "one file" an unstated part
    of this rule and flagged the split half as a LOCAL error type.
    """
    path = tmp_path / "platform_core" / "src" / "platform_core" / "error_codes.py"
    _write(path, "from enum import StrEnum\nclass ErrorCode(StrEnum):\n    BAD = 'BAD'\n")

    rule = ErrorsRule()

    assert rule.run([path]) == []


def test_a_code_vocabulary_outside_platform_core_is_still_local(tmp_path: Path) -> None:
    """THE HALF THAT MUST NOT LOOSEN.

    Widening the filename set would be worthless if the name alone bought the
    exemption: a service could add its own `error_codes.py` and define
    whatever it liked. Membership requires BOTH the filename and platform_core
    in the path.
    """
    path = tmp_path / "services" / "some_service" / "error_codes.py"
    _write(path, "from enum import StrEnum\nclass ErrorCode(StrEnum):\n    BAD = 'BAD'\n")

    rule = ErrorsRule()
    kinds = {v.kind for v in rule.run([path])}

    assert "local-error-code" in kinds


def test_an_app_error_in_the_code_vocabulary_is_not_flagged(tmp_path: Path) -> None:
    """Both central modules are exempt from both class checks, not one each.

    A predicate that exempted `error_codes.py` for `ErrorCode` only would fire
    the moment the split moved anything else, which is the fragility that made
    the original spelling wrong.
    """
    path = tmp_path / "platform_core" / "src" / "platform_core" / "error_codes.py"
    _write(path, "class AppError(Exception):\n    ...\n")

    rule = ErrorsRule()

    assert rule.run([path]) == []
