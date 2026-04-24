"""Environment configuration for doc-extract-api."""

from __future__ import annotations

from platform_core.config import _require_env_str


def get_redis_url() -> str:
    """Get the Redis connection URL.

    Returns:
        Redis URL string.

    Raises:
        RuntimeError: If REDIS_URL is not set.
    """
    return _require_env_str("REDIS_URL")


def get_database_url() -> str:
    """Get the corvis Postgres connection string.

    Returns:
        Postgres connection string.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    return _require_env_str("DATABASE_URL")


def get_tenant_email() -> str:
    """Get the tenant email for resolving tenant context.

    Returns:
        Email address string.

    Raises:
        RuntimeError: If DOC_EXTRACT_TENANT_EMAIL is not set.
    """
    return _require_env_str("DOC_EXTRACT_TENANT_EMAIL")


__all__ = [
    "get_database_url",
    "get_redis_url",
    "get_tenant_email",
]
