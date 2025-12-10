"""Test hooks for QR API - allows injecting test dependencies."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.config import _optional_env_str
from platform_workers.redis import RedisStrProto, redis_for_kv


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


# Module-level injectable factory for testing.
# Default is the production implementation.
redis_factory: Callable[[str], RedisStrProto] = redis_for_kv

# Config hook for reading environment variables.
# Tests can replace this to return test values.
get_env: Callable[[str], str | None] = _default_get_env

__all__ = ["get_env", "redis_factory"]
