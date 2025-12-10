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


def test_validation_rule_requires_import_in_service_validators(tmp_path: Path) -> None:
    rule = ValidationRule()
    svc_path = _write(
        tmp_path / "services/bar/src/bar/validators.py",
        "from platform_core.validators import load_json_dict\n",
    )
    violations = rule.run([svc_path])
    assert violations == []
