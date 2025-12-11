from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from platform_core.config import (
    _decode_table,
    _decode_toml,
    _optional_env_str,
    _parse_bool,
    _parse_float,
    _parse_int,
    _parse_log_level,
    _parse_str,
    _require_env_csv,
    _require_env_str,
    load_data_bank_settings,
    load_turkic_api_settings,
)
from platform_core.json_utils import JSONValue
from platform_core.testing import make_fake_env


def test_require_env_str_missing() -> None:
    make_fake_env()
    with pytest.raises(RuntimeError):
        _require_env_str("MISSING_KEY")


def test_require_env_str_blank() -> None:
    env = make_fake_env()
    env.set("BLANK_KEY", "   ")
    with pytest.raises(RuntimeError):
        _require_env_str("BLANK_KEY")


def test_optional_env_str() -> None:
    env = make_fake_env()
    assert _optional_env_str("OPT_KEY") is None
    env.set("OPT_KEY", " value ")
    assert _optional_env_str("OPT_KEY") == "value"
    env.set("OPT_KEY", "   ")
    assert _optional_env_str("OPT_KEY") is None


def test_require_env_csv() -> None:
    env = make_fake_env()
    env.set("CSV_KEY", " alpha , beta,,gamma ")
    out = _require_env_csv("CSV_KEY")
    assert out == frozenset({"alpha", "beta", "gamma"})
    env.set("CSV_KEY", "   ")
    with pytest.raises(RuntimeError):
        _require_env_csv("CSV_KEY")
    env.set("CSV_KEY", " , , ")
    with pytest.raises(RuntimeError):
        _require_env_csv("CSV_KEY")


def test_load_toml_success(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[section]\nkey = "value"\n', encoding="utf-8")
    out = _decode_toml(path)
    assert out == {"section": {"key": "value"}}


def test_load_toml_invalid_syntax_propagates(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("123\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        _decode_toml(path)


def test_load_toml_uses_hook(tmp_path: Path) -> None:
    """Test that _decode_toml uses the tomllib_loads hook."""
    from platform_core.config import _test_hooks

    path = tmp_path / "config.toml"
    path.write_text("value = 1\n", encoding="utf-8")

    def _fake_tomllib_loads(_: str) -> dict[str, JSONValue]:
        return {"hooked": True}

    _test_hooks.tomllib_loads = _fake_tomllib_loads
    result = _decode_toml(path)
    assert result == {"hooked": True}


def test_decode_table_success() -> None:
    result = _decode_table({"section": {"key": "value"}}, "section")
    assert result == {"key": "value"}


def test_decode_table_wrong_type() -> None:
    with pytest.raises(RuntimeError):
        _decode_table({"section": "not-a-table"}, "section")


def test_decode_table_missing() -> None:
    assert _decode_table({}, "missing") == {}


def test_load_data_bank_settings() -> None:
    env = make_fake_env()
    env.set("REDIS_URL", "redis://ignored")
    env.set("API_UPLOAD_KEYS", "one,two")
    cfg = load_data_bank_settings()
    assert cfg["api_upload_keys"] == frozenset({"one", "two"})
    assert cfg["api_read_keys"] == cfg["api_upload_keys"]
    assert cfg["api_delete_keys"] == cfg["api_upload_keys"]
    assert cfg["data_root"] == "/data/files"
    assert cfg["min_free_gb"] == 1
    assert cfg["delete_strict_404"] is False
    assert cfg["max_file_bytes"] == 0


def test_load_turkic_api_settings_defaults() -> None:
    env = make_fake_env()
    env.set("TURKIC_DATA_BANK_API_KEY", "secret")
    cfg = load_turkic_api_settings()
    assert cfg["data_bank_api_key"] == "secret"
    assert cfg["redis_url"] == "redis://redis:6379/0"
    assert cfg["data_dir"] == "/data"
    assert cfg["environment"] == "local"
    assert cfg["data_bank_api_url"] == ""


def test_load_turkic_api_settings_override() -> None:
    env = make_fake_env()
    env.set("TURKIC_DATA_BANK_API_KEY", "secret")
    env.set("TURKIC_REDIS_URL", "redis://override")
    cfg = load_turkic_api_settings()
    assert cfg["redis_url"] == "redis://override"


def test_parse_str() -> None:
    env = make_fake_env()
    assert _parse_str("STR_KEY", "default") == "default"
    env.set("STR_KEY", " value ")
    assert _parse_str("STR_KEY", "default") == "value"
    env.set("STR_KEY", "   ")
    assert _parse_str("STR_KEY", "default") == "default"


def test_parse_int() -> None:
    env = make_fake_env()
    assert _parse_int("INT_KEY", 42) == 42
    env.set("INT_KEY", " 123 ")
    assert _parse_int("INT_KEY", 42) == 123
    env.set("INT_KEY", "not-a-number")
    with pytest.raises(ValueError):
        _parse_int("INT_KEY", 42)


def test_parse_float() -> None:
    env = make_fake_env()
    assert _parse_float("FLOAT_KEY", 3.14) == 3.14
    env.set("FLOAT_KEY", " 2.5 ")
    assert _parse_float("FLOAT_KEY", 3.14) == 2.5
    env.set("FLOAT_KEY", "not-a-float")
    with pytest.raises(ValueError):
        _parse_float("FLOAT_KEY", 3.14)


def test_parse_bool() -> None:
    env = make_fake_env()
    assert _parse_bool("BOOL_KEY", False) is False
    assert _parse_bool("BOOL_KEY", True) is True

    for true_val in ["1", "true", "TRUE", "yes", "YES", "y", "Y", "on", "ON"]:
        env.set("BOOL_KEY", true_val)
        assert _parse_bool("BOOL_KEY", False) is True

    for false_val in ["0", "false", "FALSE", "no", "NO", "n", "N", "off", "OFF"]:
        env.set("BOOL_KEY", false_val)
        assert _parse_bool("BOOL_KEY", True) is False

    env.set("BOOL_KEY", "invalid")
    with pytest.raises(ValueError):
        _parse_bool("BOOL_KEY", False)


def test_parse_log_level() -> None:
    env = make_fake_env()
    assert _parse_log_level("LOG_LEVEL", "INFO") == "INFO"

    for level in ["DEBUG", "debug", "Debug"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "DEBUG"

    env.clear()
    for level in ["INFO", "info", "Info"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "DEBUG") == "INFO"

    env.clear()
    for level in ["WARNING", "warning", "Warning"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "WARNING"

    env.clear()
    for level in ["ERROR", "error", "Error"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "ERROR"

    env.clear()
    for level in ["CRITICAL", "critical", "Critical"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "CRITICAL"

    # Invalid value falls back to default
    env.clear()
    env.set("LOG_LEVEL", "invalid")
    assert _parse_log_level("LOG_LEVEL", "WARNING") == "WARNING"


def test_default_get_env_returns_real_environ() -> None:
    """Test _default_get_env reads from real os.environ."""
    import os

    from platform_core.config._test_hooks import _default_get_env

    # Test getting a key that exists in real environment
    result = _default_get_env("PATH")
    assert result == os.getenv("PATH")

    # Test getting a key that doesn't exist
    result = _default_get_env("__DEFINITELY_NOT_A_REAL_ENV_VAR__")
    assert result is None


def test_fake_env_delete() -> None:
    """Test FakeEnv.delete removes a key."""
    env = make_fake_env({"KEY": "value"})
    assert env.get("KEY") == "value"
    env.delete("KEY")
    assert env.get("KEY") is None
    # Deleting non-existent key should not raise
    env.delete("NONEXISTENT")
