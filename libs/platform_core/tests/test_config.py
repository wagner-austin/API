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
    _parse_log_format,
    _parse_log_level,
    _parse_str,
    _require_env_csv,
    _require_env_str,
)
from platform_core.config._utils import _validate_log_format, _validate_log_level
from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import LogFormat, LogLevel
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
    assert _parse_log_level("LOG_LEVEL", LogLevel.INFO) is LogLevel.INFO

    for level in ["DEBUG", "debug", "Debug"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", LogLevel.INFO) is LogLevel.DEBUG

    env.clear()
    for level in ["INFO", "info", "Info"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", LogLevel.DEBUG) is LogLevel.INFO

    env.clear()
    for level in ["WARNING", "warning", "Warning"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", LogLevel.INFO) is LogLevel.WARNING

    env.clear()
    for level in ["ERROR", "error", "Error"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", LogLevel.INFO) is LogLevel.ERROR

    env.clear()
    for level in ["CRITICAL", "critical", "Critical"]:
        env.set("LOG_LEVEL", level)
        assert _parse_log_level("LOG_LEVEL", LogLevel.INFO) is LogLevel.CRITICAL

    env.clear()
    env.set("LOG_LEVEL", "invalid")
    with pytest.raises(JSONTypeError, match="Invalid log level: invalid"):
        _parse_log_level("LOG_LEVEL", LogLevel.WARNING)


def test_parse_log_format() -> None:
    env = make_fake_env()
    assert _parse_log_format("LOG_FORMAT", LogFormat.JSON) is LogFormat.JSON

    for spelling in ["TEXT", "text", "Text"]:
        env.set("LOG_FORMAT", spelling)
        assert _parse_log_format("LOG_FORMAT", LogFormat.JSON) is LogFormat.TEXT

    env.clear()
    env.set("LOG_FORMAT", "json")
    assert _parse_log_format("LOG_FORMAT", LogFormat.TEXT) is LogFormat.JSON

    env.clear()
    env.set("LOG_FORMAT", "yaml")
    with pytest.raises(JSONTypeError, match="Invalid log format: yaml"):
        _parse_log_format("LOG_FORMAT", LogFormat.JSON)


class TestValidatingALogLevel:
    """An unrecognised level is refused rather than quietly downgraded.

    `_parse_log_level` used to return the default, so `LOG_LEVEL=TRACE`
    started the service at INFO and logged nothing about it -- and the levels
    an operator is most likely to mistype are the verbose ones they reached
    for while debugging, so the failure hid exactly the output that was asked
    for. `grandma-api` and `opportunity-radar-api` had each written a strict
    copy beside it rather than fix it; those copies are gone and this is the
    one that remains.
    """

    def test_every_level_is_accepted_in_any_case(self) -> None:
        assert _validate_log_level("debug") is LogLevel.DEBUG
        assert _validate_log_level("INFO") is LogLevel.INFO
        assert _validate_log_level("Warning") is LogLevel.WARNING
        assert _validate_log_level("ERROR") is LogLevel.ERROR
        assert _validate_log_level("critical") is LogLevel.CRITICAL

    def test_an_unrecognised_level_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="Invalid log level: TRACE"):
            _validate_log_level("TRACE")


class TestValidatingALogFormat:
    """The set is `json` and `text`, and it is now spelled once.

    It had been written three times -- `platform_core.logging` plus a copy in
    each of the two services -- which is the shape a fourth copy spelling
    `console` would have joined without anything noticing.
    """

    def test_both_formats_are_accepted_in_any_case(self) -> None:
        assert _validate_log_format("JSON") is LogFormat.JSON
        assert _validate_log_format("Text") is LogFormat.TEXT

    def test_an_unrecognised_format_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="Invalid log format: console"):
            _validate_log_format("console")


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


def test_default_get_environment_returns_the_whole_real_environ() -> None:
    """Test _default_get_environment answers the process question, not the
    configuration one: what a CHILD should inherit.

    ``subprocess`` replaces the environment wholesale when given one, so a
    caller adding a single variable has to start from the parent's -- and this
    module is the only one permitted to read it.
    """
    import os

    from platform_core.config._test_hooks import _default_get_environment

    assert _default_get_environment() == dict(os.environ)


def test_default_get_environment_returns_a_copy() -> None:
    """A caller's overlay must not reach back into this process's own
    environment: it is building a child's, not editing its own."""
    from platform_core.config._test_hooks import _default_get_environment

    first = _default_get_environment()
    first["__PLATFORM_CORE_TEST_ONLY__"] = "1"
    assert "__PLATFORM_CORE_TEST_ONLY__" not in _default_get_environment()


def test_fake_env_delete() -> None:
    """Test FakeEnv.delete removes a key."""
    env = make_fake_env({"KEY": "value"})
    assert env.get("KEY") == "value"
    env.delete("KEY")
    assert env.get("KEY") is None
    # Deleting non-existent key should not raise
    env.delete("NONEXISTENT")
