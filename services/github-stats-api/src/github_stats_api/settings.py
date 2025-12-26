from __future__ import annotations

from platform_core.config import _parse_int, _parse_str
from typing_extensions import TypedDict


class Settings(TypedDict, total=True):
    """Service settings.

    Attributes:
        github_token: GitHub personal access token for API requests.
        cache_ttl_seconds: How long to cache GitHub API responses.
        port: HTTP server port.
    """

    github_token: str
    cache_ttl_seconds: int
    port: int


def load_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        Settings TypedDict with validated values.

    Raises:
        ValueError: If integer environment variables have invalid values.
    """
    return {
        "github_token": _parse_str("GITHUB_TOKEN", ""),
        "cache_ttl_seconds": _parse_int("CACHE_TTL_SECONDS", 1800),
        "port": _parse_int("PORT", 8000),
    }


__all__ = ["Settings", "load_settings"]
