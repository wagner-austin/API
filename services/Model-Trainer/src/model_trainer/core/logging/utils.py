from __future__ import annotations

from platform_core.logging import LogLevel


def narrow_log_level(level_str: str) -> LogLevel:
    """Narrow a string to a valid LogLevel Literal type.

    Args:
        level_str: Log level string from configuration

    Returns:
        Validated LogLevel literal, defaulting to INFO if invalid
    """
    if level_str == "DEBUG":
        return "DEBUG"
    if level_str == "INFO":
        return "INFO"
    if level_str == "WARNING":
        return "WARNING"
    if level_str == "ERROR":
        return "ERROR"
    if level_str == "CRITICAL":
        return "CRITICAL"
    return "INFO"


__all__ = ["narrow_log_level"]
