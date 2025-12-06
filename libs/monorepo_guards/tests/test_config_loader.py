from __future__ import annotations

from pathlib import Path

import pytest

from monorepo_guards.config_loader import _decode_monorepo_guard_config


def test_decode_monorepo_guard_config_with_all_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
directories = ["src", "lib", "tests"]
exclude_parts = [".venv", "node_modules"]
forbid_pyi = false
allow_print_in_tests = true
dataclass_ban_segments = [["src"], ["lib", "internal"]]
""",
        encoding="utf-8",
    )

    config = _decode_monorepo_guard_config(tmp_path)

    assert config.directories == ("src", "lib", "tests")
    assert config.exclude_parts == (".venv", "node_modules")
    assert not config.forbid_pyi
    assert config.allow_print_in_tests
    assert config.dataclass_ban_segments == (("src",), ("lib", "internal"))


def test_decode_monorepo_guard_config_with_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
""",
        encoding="utf-8",
    )

    config = _decode_monorepo_guard_config(tmp_path)

    assert config.directories == ("src", "scripts", "tests")
    assert config.exclude_parts == ()
    assert config.forbid_pyi
    assert not config.allow_print_in_tests
    assert config.dataclass_ban_segments == ()


def test_decode_monorepo_guard_config_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text("", encoding="utf-8")

    config = _decode_monorepo_guard_config(tmp_path)

    assert config.directories == ("src", "scripts", "tests")
    assert config.exclude_parts == ()
    assert config.forbid_pyi
    assert not config.allow_print_in_tests
    assert config.dataclass_ban_segments == ()


def test_decode_monorepo_guard_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Monorepo guard config not found"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_monorepo_guard_config_invalid_guards_section(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
guards = "not a mapping"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"'guards' section.*must be a mapping"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_string_tuple_invalid_type(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
directories = "not a list"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list or tuple of strings"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_string_tuple_invalid_item(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
directories = [123, "test"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"All items.*must be strings"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_boolean_invalid_type(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
forbid_pyi = "not a bool"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_dataclass_ban_segments_invalid_outer_type(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
dataclass_ban_segments = "not a list"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list or tuple of lists/tuples of strings"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_dataclass_ban_segments_invalid_inner_type(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
dataclass_ban_segments = ["not a list"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"All items.*must be lists or tuples of strings"):
        _decode_monorepo_guard_config(tmp_path)


def test_decode_dataclass_ban_segments_invalid_item(tmp_path: Path) -> None:
    config_path = tmp_path / "monorepo-guards.toml"
    config_path.write_text(
        """
[guards]
dataclass_ban_segments = [[123]]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"All sub-items.*must be strings"):
        _decode_monorepo_guard_config(tmp_path)
