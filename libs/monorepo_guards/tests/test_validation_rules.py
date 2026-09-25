from __future__ import annotations

from pathlib import Path

from monorepo_guards.validation_rules import ValidationRule


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_validation_rule_flags_duplicates_and_missing_imports(tmp_path: Path) -> None:
    rule = ValidationRule()
    svc_path = _write(
        tmp_path / "services/foo/src/foo/validators.py",
        "def _decode_int_range(x, y):\n    return 1\n",
    )
    violations = rule.run([svc_path])
    kinds = {v.kind for v in violations}
    assert "duplicate-validators" in kinds
    assert "missing-platform-validators-import" in kinds


def test_validation_rule_allows_platform_validators(tmp_path: Path) -> None:
    rule = ValidationRule()
    platform_path = _write(
        tmp_path / "libs/platform_core/src/platform_core/validators.py",
        "def _decode_int_range(x, y):\n    return 1\n",
    )
    violations = rule.run([platform_path, tmp_path / "README.md"])
    assert violations == []


def test_a_longer_name_that_merely_starts_the_same_is_not_a_duplicate(tmp_path: Path) -> None:
    """``def _decode_stream(`` is not ``def _decode_str(``.

    THE EXACT FALSE POSITIVE THIS PREDICATE HAD. The banned names were matched
    as substrings without their opening parenthesis, so ``def _decode_str``
    fired on ``def _decode_stream``. Measured 2026-09-20 20:38Z: tools/fleet
    make lint refused a helper legitimately named ``_decode_stream`` and it was
    renamed to ``_decode_captured`` to get past the guard, which is a correct
    symbol changed to suit a broken check.
    """
    rule = ValidationRule()
    svc_path = _write(
        tmp_path / "services/foo/src/foo/streams.py",
        "def _decode_stream(raw):\n    return raw\n",
    )
    assert [v.kind for v in rule.run([svc_path])] == []


def test_the_banned_validator_itself_is_still_refused(tmp_path: Path) -> None:
    """The narrowed predicate still fires on the real thing.

    Paired with the case above deliberately: a predicate can be narrowed until
    it catches nothing, and a false-positive fix that quietly disarms the rule
    is worse than the false positive.
    """
    rule = ValidationRule()
    svc_path = _write(
        tmp_path / "services/foo/src/foo/streams.py",
        "def _decode_str(raw):\n    return raw\n",
    )
    assert [v.kind for v in rule.run([svc_path])] == ["duplicate-validators"]


def test_validation_rule_requires_import_in_service_validators(tmp_path: Path) -> None:
    rule = ValidationRule()
    svc_path = _write(
        tmp_path / "services/bar/src/bar/validators.py",
        "from platform_core.validators import load_json_dict\n",
    )
    violations = rule.run([svc_path])
    assert violations == []
