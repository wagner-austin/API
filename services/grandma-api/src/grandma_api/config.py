"""Configuration for grandma-api service.

Provides TypedDict settings with encode/decode/require_* validation.
"""

from __future__ import annotations

from platform_core.config import (
    _parse_int,
    _parse_log_format,
    _parse_log_level,
    _require_env_str,
    _validate_log_format,
    _validate_log_level,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_str,
)
from platform_core.logging import LogFormat, LogLevel
from typing_extensions import TypedDict

# =============================================================================
# Settings TypedDict
# =============================================================================


class GrandmaApiSettings(TypedDict):
    """Configuration settings for grandma-api service."""

    openai_api_key: str
    api_token: str
    port: int
    log_level: LogLevel
    log_format: LogFormat


def encode_grandma_api_settings(settings: GrandmaApiSettings) -> JSONObject:
    """Encode GrandmaApiSettings to JSON-compatible dict.

    Args:
        settings: The settings to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "openai_api_key": settings["openai_api_key"],
        "api_token": settings["api_token"],
        "port": settings["port"],
        "log_level": settings["log_level"],
        "log_format": settings["log_format"],
    }


def decode_grandma_api_settings(obj: JSONObject) -> GrandmaApiSettings:
    """Decode JSON object to GrandmaApiSettings with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated GrandmaApiSettings.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    openai_api_key = require_str(obj, "openai_api_key")
    api_token = require_str(obj, "api_token")
    port = require_int(obj, "port")
    log_level_raw = require_str(obj, "log_level")
    log_format_raw = require_str(obj, "log_format")

    return GrandmaApiSettings(
        openai_api_key=openai_api_key,
        api_token=api_token,
        port=port,
        log_level=_validate_log_level(log_level_raw),
        log_format=_validate_log_format(log_format_raw),
    )


def require_grandma_api_settings(obj: JSONValue) -> GrandmaApiSettings:
    """Validate and convert JSONValue to GrandmaApiSettings.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated GrandmaApiSettings.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_grandma_api_settings(obj)


# =============================================================================
# Environment Loading
# =============================================================================


def load_settings() -> GrandmaApiSettings:
    """Load settings from environment variables.

    Returns:
        Validated GrandmaApiSettings.

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    openai_api_key = _require_env_str("OPENAI_API_KEY")
    api_token = _require_env_str("API_TOKEN")
    port = _parse_int("PORT", 8080)
    log_level = _parse_log_level("LOG_LEVEL", "INFO")
    log_format = _parse_log_format("LOG_FORMAT", "json")

    return GrandmaApiSettings(
        openai_api_key=openai_api_key,
        api_token=api_token,
        port=port,
        log_level=log_level,
        log_format=log_format,
    )


__all__ = [
    "GrandmaApiSettings",
    "LogFormat",
    "LogLevel",
    "decode_grandma_api_settings",
    "encode_grandma_api_settings",
    "load_settings",
    "require_grandma_api_settings",
]
