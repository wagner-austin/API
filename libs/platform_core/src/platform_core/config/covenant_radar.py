from __future__ import annotations

from typing import TypedDict

from ._utils import _require_env_str


class CovenantRadarSettings(TypedDict):
    """Configuration for covenant-radar-api service."""

    redis_url: str
    database_url: str


def load_covenant_radar_settings() -> CovenantRadarSettings:
    """Load covenant-radar settings from environment variables.

    Required environment variables:
        REDIS_URL: Redis connection URL for job queue
        DATABASE_URL: PostgreSQL connection URL
    """
    return {
        "redis_url": _require_env_str("REDIS_URL"),
        "database_url": _require_env_str("DATABASE_URL"),
    }


__all__ = ["CovenantRadarSettings", "load_covenant_radar_settings"]
