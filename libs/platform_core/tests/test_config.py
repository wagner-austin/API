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


def test_require_env_str_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _require_env_str("MISSING_KEY")


def test_require_env_str_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLANK_KEY", "   ")
    with pytest.raises(RuntimeError):
        _require_env_str("BLANK_KEY")


def test_optional_env_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPT_KEY", raising=False)
    assert _optional_env_str("OPT_KEY") is None
    monkeypatch.setenv("OPT_KEY", " value ")
    assert _optional_env_str("OPT_KEY") == "value"
    monkeypatch.setenv("OPT_KEY", "   ")
    assert _optional_env_str("OPT_KEY") is None


def test_require_env_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CSV_KEY", " alpha , beta,,gamma ")
    out = _require_env_csv("CSV_KEY")
    assert out == frozenset({"alpha", "beta", "gamma"})
    monkeypatch.setenv("CSV_KEY", "   ")
    with pytest.raises(RuntimeError):
        _require_env_csv("CSV_KEY")
    monkeypatch.setenv("CSV_KEY", " , , ")
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


def test_load_toml_non_table_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text("value = 1\n", encoding="utf-8")

    def _decode_fake_loads(_: str) -> JSONValue:
        return "not-a-table"

    monkeypatch.setattr(tomllib, "loads", _decode_fake_loads)

    with pytest.raises(RuntimeError):
        _decode_toml(path)


def test_decode_table_success() -> None:
    result = _decode_table({"section": {"key": "value"}}, "section")
    assert result == {"key": "value"}


def test_decode_table_wrong_type() -> None:
    with pytest.raises(RuntimeError):
        _decode_table({"section": "not-a-table"}, "section")


def test_decode_table_missing() -> None:
    assert _decode_table({}, "missing") == {}


def test_load_data_bank_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://ignored")
    monkeypatch.setenv("API_UPLOAD_KEYS", "one,two")
    monkeypatch.delenv("API_READ_KEYS", raising=False)
    monkeypatch.delenv("API_DELETE_KEYS", raising=False)
    cfg = load_data_bank_settings()
    assert cfg["api_upload_keys"] == frozenset({"one", "two"})
    assert cfg["api_read_keys"] == cfg["api_upload_keys"]
    assert cfg["api_delete_keys"] == cfg["api_upload_keys"]
    assert cfg["data_root"] == "/data/files"
    assert cfg["min_free_gb"] == 1
    assert cfg["delete_strict_404"] is False
    assert cfg["max_file_bytes"] == 0


def test_load_turkic_api_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "secret")
    monkeypatch.delenv("TURKIC_REDIS_URL", raising=False)
    cfg = load_turkic_api_settings()
    assert cfg["data_bank_api_key"] == "secret"
    assert cfg["redis_url"] == "redis://redis:6379/0"
    assert cfg["data_dir"] == "/data"
    assert cfg["environment"] == "local"
    assert cfg["data_bank_api_url"] == ""


def test_load_turkic_api_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURKIC_DATA_BANK_API_KEY", "secret")
    monkeypatch.setenv("TURKIC_REDIS_URL", "redis://override")
    cfg = load_turkic_api_settings()
    assert cfg["redis_url"] == "redis://override"


def test_parse_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STR_KEY", raising=False)
    assert _parse_str("STR_KEY", "default") == "default"
    monkeypatch.setenv("STR_KEY", " value ")
    assert _parse_str("STR_KEY", "default") == "value"
    monkeypatch.setenv("STR_KEY", "   ")
    assert _parse_str("STR_KEY", "default") == "default"


def test_parse_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INT_KEY", raising=False)
    assert _parse_int("INT_KEY", 42) == 42
    monkeypatch.setenv("INT_KEY", " 123 ")
    assert _parse_int("INT_KEY", 42) == 123
    monkeypatch.setenv("INT_KEY", "not-a-number")
    with pytest.raises(ValueError):
        _parse_int("INT_KEY", 42)


def test_parse_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLOAT_KEY", raising=False)
    assert _parse_float("FLOAT_KEY", 3.14) == 3.14
    monkeypatch.setenv("FLOAT_KEY", " 2.5 ")
    assert _parse_float("FLOAT_KEY", 3.14) == 2.5
    monkeypatch.setenv("FLOAT_KEY", "not-a-float")
    with pytest.raises(ValueError):
        _parse_float("FLOAT_KEY", 3.14)


def test_parse_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOL_KEY", raising=False)
    assert _parse_bool("BOOL_KEY", False) is False
    assert _parse_bool("BOOL_KEY", True) is True

    for true_val in ["1", "true", "TRUE", "yes", "YES", "y", "Y", "on", "ON"]:
        monkeypatch.setenv("BOOL_KEY", true_val)
        assert _parse_bool("BOOL_KEY", False) is True

    for false_val in ["0", "false", "FALSE", "no", "NO", "n", "N", "off", "OFF"]:
        monkeypatch.setenv("BOOL_KEY", false_val)
        assert _parse_bool("BOOL_KEY", True) is False

    monkeypatch.setenv("BOOL_KEY", "invalid")
    with pytest.raises(ValueError):
        _parse_bool("BOOL_KEY", False)


def test_parse_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert _parse_log_level("LOG_LEVEL", "INFO") == "INFO"

    for level in ["DEBUG", "debug", "Debug"]:
        monkeypatch.setenv("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "DEBUG"

    for level in ["INFO", "info", "Info"]:
        monkeypatch.setenv("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "DEBUG") == "INFO"

    for level in ["WARNING", "warning", "Warning"]:
        monkeypatch.setenv("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "WARNING"

    for level in ["ERROR", "error", "Error"]:
        monkeypatch.setenv("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "ERROR"

    for level in ["CRITICAL", "critical", "Critical"]:
        monkeypatch.setenv("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", "INFO") == "CRITICAL"

    # Invalid value falls back to default
    monkeypatch.setenv("LOG_LEVEL", "invalid")
    assert _parse_log_level("LOG_LEVEL", "WARNING") == "WARNING"
