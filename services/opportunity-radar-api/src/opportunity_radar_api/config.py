"""Configuration for opportunity-radar-api service.

Provides TypedDict settings with encode/decode/require_* validation.
"""

from __future__ import annotations

from typing import Literal

from platform_core.config import (
    _optional_env_str,
    _parse_int,
    _parse_log_level,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    optional_str,
    require_int,
    require_str,
)
from platform_core.logging import LogLevel
from typing_extensions import TypedDict

# Log format type for this service
LogFormat = Literal["json", "text"]


# =============================================================================
# Settings TypedDict
# =============================================================================


class OpportunityRadarSettings(TypedDict):
    """Configuration settings for opportunity-radar-api service."""

    kaggle_api_token: str
    port: int
    log_level: LogLevel
    log_format: LogFormat
    github_token: str | None
    github_repo: str | None


def encode_opportunity_radar_settings(settings: OpportunityRadarSettings) -> JSONObject:
    """Encode OpportunityRadarSettings to JSON-compatible dict.

    Args:
        settings: The settings to encode.

    Returns:
        JSON-compatible dictionary.
    """
    return {
        "kaggle_api_token": settings["kaggle_api_token"],
        "port": settings["port"],
        "log_level": settings["log_level"],
        "log_format": settings["log_format"],
        "github_token": settings["github_token"],
        "github_repo": settings["github_repo"],
    }


def _validate_log_level(value: str) -> LogLevel:
    """Validate and convert string to LogLevel.

    Args:
        value: String value to validate.

    Returns:
        Validated LogLevel.

    Raises:
        JSONTypeError: If value is not a valid log level.
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
    """Validate and convert string to LogFormat.

    Args:
        value: String value to validate.

    Returns:
        Validated LogFormat.

    Raises:
        JSONTypeError: If value is not a valid log format.
    """
    lower = value.lower()
    if lower == "json":
        return "json"
    if lower == "text":
        return "text"
    raise JSONTypeError(f"Invalid log format: {value}")


def decode_opportunity_radar_settings(obj: JSONObject) -> OpportunityRadarSettings:
    """Decode JSON object to OpportunityRadarSettings with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated OpportunityRadarSettings.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    kaggle_api_token = require_str(obj, "kaggle_api_token")
    port = require_int(obj, "port")
    log_level_raw = require_str(obj, "log_level")
    log_format_raw = require_str(obj, "log_format")
    github_token = optional_str(obj, "github_token")
    github_repo = optional_str(obj, "github_repo")

    return OpportunityRadarSettings(
        kaggle_api_token=kaggle_api_token,
        port=port,
        log_level=_validate_log_level(log_level_raw),
        log_format=_validate_log_format(log_format_raw),
        github_token=github_token,
        github_repo=github_repo,
    )


def require_opportunity_radar_settings(obj: JSONValue) -> OpportunityRadarSettings:
    """Validate and convert JSONValue to OpportunityRadarSettings.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated OpportunityRadarSettings.

    Raises:
        JSONTypeError: If validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_opportunity_radar_settings(obj)


# =============================================================================
# Environment Loading
# =============================================================================


def _parse_log_format(key: str, default: LogFormat) -> LogFormat:
    """Parse log format from environment with default.

    Args:
        key: Environment variable name.
        default: Default value if not set.

    Returns:
        Parsed LogFormat.
    """
    val = _optional_env_str(key)
    if val is None:
        return default
    lower = val.lower()
    if lower == "json":
        return "json"
    if lower == "text":
        return "text"
    return default


def load_settings() -> OpportunityRadarSettings:
    """Load settings from environment variables.

    Returns:
        Validated OpportunityRadarSettings.

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    # Kaggle token is optional - if not set, Kaggle features won't work
    kaggle_api_token = _optional_env_str("KAGGLE_API_TOKEN")
    if kaggle_api_token is None:
        kaggle_api_token = ""

    port = _parse_int("PORT", 8010)
    log_level = _parse_log_level("LOG_LEVEL", "INFO")
    log_format = _parse_log_format("LOG_FORMAT", "json")

    # GitHub settings for codebase scanning from GitHub API
    github_token = _optional_env_str("GITHUB_TOKEN")
    github_repo = _optional_env_str("GITHUB_REPO")

    return OpportunityRadarSettings(
        kaggle_api_token=kaggle_api_token,
        port=port,
        log_level=log_level,
        log_format=log_format,
        github_token=github_token,
        github_repo=github_repo,
    )


__all__ = [
    "LogFormat",
    "LogLevel",
    "OpportunityRadarSettings",
    "decode_opportunity_radar_settings",
    "encode_opportunity_radar_settings",
    "load_settings",
    "require_opportunity_radar_settings",
]
