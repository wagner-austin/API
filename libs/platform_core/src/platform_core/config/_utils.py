from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from platform_core.json_utils import JSONValue

# Log level type - must match platform_core.logging.LogLevel
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_VALID_LOG_LEVELS: frozenset[LogLevel] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class _EnvError(RuntimeError):
    pass


def _require_env_str(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise _EnvError(f"Missing required env var: {key}")
    trimmed = value.strip()
    if trimmed == "":
        raise _EnvError(f"Empty env var: {key}")
    return trimmed


def _optional_env_str(key: str) -> str | None:
    value = os.getenv(key)
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


def _parse_log_level(key: str, default: LogLevel) -> LogLevel:
    val = _optional_env_str(key)
    if val is None:
        return default
    upper_val = val.upper()
    if upper_val == "DEBUG":
        return "DEBUG"
    if upper_val == "INFO":
        return "INFO"
    if upper_val == "WARNING":
        return "WARNING"
    if upper_val == "ERROR":
        return "ERROR"
    if upper_val == "CRITICAL":
        return "CRITICAL"
    return default


def _decode_toml(path: Path) -> dict[str, JSONValue]:
    text = path.read_text(encoding="utf-8")
    parsed: JSONValue = tomllib.loads(text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"TOML root must be a table: {path}")
    return parsed


def _decode_table(data: dict[str, JSONValue], key: str) -> dict[str, JSONValue]:
    raw = data.get(key)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"TOML key {key} must be a table")
    return {str(k): v for k, v in raw.items()}


__all__ = [
    "JSONValue",
    "LogLevel",
    "_decode_table",
    "_decode_toml",
    "_optional_env_str",
    "_parse_bool",
    "_parse_float",
    "_parse_int",
    "_parse_log_level",
    "_parse_str",
    "_require_env_csv",
    "_require_env_str",
]
