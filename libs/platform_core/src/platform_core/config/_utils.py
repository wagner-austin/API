from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.logging import LogFormat, LogLevel

from . import _test_hooks

_VALID_LOG_LEVELS: frozenset[LogLevel] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class _EnvError(RuntimeError):
    pass


def _require_env_str(key: str) -> str:
    value = _test_hooks.get_env(key)
    if value is None:
        raise _EnvError(f"Missing required env var: {key}")
    trimmed = value.strip()
    if trimmed == "":
        raise _EnvError(f"Empty env var: {key}")
    return trimmed


def _optional_env_str(key: str) -> str | None:
    value = _test_hooks.get_env(key)
    if value is None:
        return None
    trimmed = value.strip()
    if trimmed == "":
        return None
    return trimmed


def _require_env_csv(key: str) -> frozenset[str]:
    raw = _require_env_str(key)
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise _EnvError(f"Env var {key} must contain at least one entry")
    return frozenset(parts)


def _parse_str(key: str, default: str) -> str:
    val = _optional_env_str(key)
    return val if val is not None else default


def _parse_int(key: str, default: int) -> int:
    val = _optional_env_str(key)
    if val is None:
        return default
    return int(val)


def _parse_float(key: str, default: float) -> float:
    val = _optional_env_str(key)
    if val is None:
        return default
    return float(val)


def _parse_bool(key: str, default: bool) -> bool:
    val = _optional_env_str(key)
    if val is None:
        return default
    normalized = val.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {key}: {val!r}")


def _validate_log_level(value: str) -> LogLevel:
    """Narrow a string to a log level.

    Args:
        value: The value to narrow, in any case.

    Returns:
        The matching level.

    Raises:
        JSONTypeError: If the value names no level. It does NOT fall back to
            a default: `_parse_log_level` used to, so `LOG_LEVEL=TRACE`
            started the service at INFO and logged nothing about it. An
            operator who set a level and did not get it has no way to tell
            from the running process, and the levels they are most likely to
            mistype are the verbose ones they reached for while debugging.
    """
    upper = value.upper()
    if upper == "DEBUG":
        return "DEBUG"
    if upper == "INFO":
        return "INFO"
    if upper == "WARNING":
        return "WARNING"
    if upper == "ERROR":
        return "ERROR"
    if upper == "CRITICAL":
        return "CRITICAL"
    raise JSONTypeError(f"Invalid log level: {value}")


def _validate_log_format(value: str) -> LogFormat:
    """Narrow a string to a log format.

    Args:
        value: The value to narrow, in any case.

    Returns:
        The matching format.

    Raises:
        JSONTypeError: If the value names no format.
    """
    lower = value.lower()
    if lower == "json":
        return "json"
    if lower == "text":
        return "text"
    raise JSONTypeError(f"Invalid log format: {value}")


def _parse_log_level(key: str, default: LogLevel) -> LogLevel:
    val = _optional_env_str(key)
    if val is None:
        return default
    return _validate_log_level(val)


def _parse_log_format(key: str, default: LogFormat) -> LogFormat:
    val = _optional_env_str(key)
    if val is None:
        return default
    return _validate_log_format(val)


def _decode_toml(path: Path) -> dict[str, JSONValue]:
    text = path.read_text(encoding="utf-8")
    return _test_hooks.tomllib_loads(text)


def _decode_table(data: dict[str, JSONValue], key: str) -> dict[str, JSONValue]:
    raw = data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"TOML key {key} must be a table")
    return {str(k): v for k, v in raw.items()}


__all__ = [
    "JSONValue",
    "LogFormat",
    "LogLevel",
    "_decode_table",
    "_decode_toml",
    "_optional_env_str",
    "_parse_bool",
    "_parse_float",
    "_parse_int",
    "_parse_log_format",
    "_parse_log_level",
    "_parse_str",
    "_require_env_csv",
    "_require_env_str",
    "_validate_log_format",
    "_validate_log_level",
]
