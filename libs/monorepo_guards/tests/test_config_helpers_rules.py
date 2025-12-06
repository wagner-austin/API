from __future__ import annotations

from pathlib import Path

from monorepo_guards.config_helpers_rules import ConfigHelpersRule


def test_config_helpers_rule_detects_duplicate(tmp_path: Path) -> None:
    file = tmp_path / "settings.py"
    file.write_text(
        """
def _parse_int(key: str, default: int) -> int:
    return default
""",
        encoding="utf-8",
    )

    rule = ConfigHelpersRule()
    violations = rule.run([file])
    assert len(violations) == 1
    assert violations[0].kind == "config-helper-duplicate"
    assert "_parse_int" in violations[0].line


def test_config_helpers_rule_allows_platform_core(tmp_path: Path) -> None:
    # Simulate platform_core/config.py path
    platform_core = tmp_path / "src" / "platform_core" / "config"
    platform_core.mkdir(parents=True)
    config_file = platform_core / "_utils.py"
    config_file.write_text(
        """
def _parse_int(key: str, default: int) -> int:
    return default
""",
        encoding="utf-8",
    )

    rule = ConfigHelpersRule()
    violations = rule.run([config_file])
    assert len(violations) == 0


def test_config_helpers_rule_detects_all_banned_names(tmp_path: Path) -> None:
    file = tmp_path / "config.py"
    file.write_text(
        """
def _parse_int(key: str, default: int) -> int:
    return default

def _parse_float(key: str, default: float) -> float:
    return default

def _parse_bool(key: str, default: bool) -> bool:
    return default

def _parse_str(key: str, default: str) -> str:
    return default

def _clean_env_value(env: object, key: str) -> str | None:
    return None
""",
        encoding="utf-8",
    )

    rule = ConfigHelpersRule()
    violations = rule.run([file])
    assert len(violations) == 5
    # Check that all banned function names are detected
    violation_lines = {v.line for v in violations}
    assert all(
        any(name in line for line in violation_lines)
        for name in ["_parse_int", "_parse_float", "_parse_bool", "_parse_str", "_clean_env_value"]
    )


def test_config_helpers_rule_ignores_non_banned_functions(tmp_path: Path) -> None:
    file = tmp_path / "helpers.py"
    file.write_text(
        """
def some_other_function() -> None:
    pass

def _other_helper() -> None:
    pass
""",
        encoding="utf-8",
    )

    rule = ConfigHelpersRule()
    violations = rule.run([file])
    assert len(violations) == 0


def test_config_helpers_rule_handles_read_error(tmp_path: Path) -> None:
    import pytest

    # Create a file path that doesn't exist
    nonexistent = tmp_path / "does_not_exist.py"

    rule = ConfigHelpersRule()
    with pytest.raises(RuntimeError, match="Failed to read"):
        rule.run([nonexistent])


def test_config_helpers_rule_handles_syntax_error(tmp_path: Path) -> None:
    import pytest

    file = tmp_path / "invalid.py"
    file.write_text("def invalid syntax here", encoding="utf-8")

    rule = ConfigHelpersRule()
    with pytest.raises(RuntimeError, match="Failed to parse"):
        rule.run([file])
