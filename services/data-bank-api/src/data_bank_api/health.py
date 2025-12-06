"""Standardized health check utilities for data-bank-api.

Uses platform_core.health for liveness and platform_workers.health for Redis,
with additional storage checks specific to this service.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from platform_core.health import HealthResponse, ReadyResponse, healthz
from platform_core.logging import get_logger
from platform_workers.health import readyz_redis_with_workers
from platform_workers.redis import RedisStrProto

_logger = get_logger(__name__)


def _is_writable(path: Path) -> bool:
    """Check if a path is writable by attempting to create a temp file.

    Returns False if the directory is not writable or a filesystem error occurs.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(prefix="probe_", dir=str(path))
    except OSError:
        return False
    os.close(fd)
    Path(tmp).unlink(missing_ok=True)
    return True


def _free_gb(path: Path) -> float:
    """Return free disk space in GB for the given path."""
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def healthz_endpoint() -> HealthResponse:
    """Liveness probe - always returns ok."""
    return healthz()


def readyz_endpoint(
    redis: RedisStrProto,
    data_root: str,
    min_free_gb: int,
) -> ReadyResponse:
    """Readiness probe - checks Redis and storage.

    Args:
        redis: Redis client for connectivity check
        data_root: Path to data directory that must be writable
        min_free_gb: Minimum free disk space in GB

    Returns:
        Ready response with status and optional reason
    """
    # Check Redis connectivity and worker presence first
    redis_result = readyz_redis_with_workers(redis)
    if redis_result["status"] == "degraded":
        return redis_result

    # Check storage writable (create if missing)
    root = Path(data_root)
    writable: bool
    writable = _is_writable(root)
    if not writable:
        _logger.warning("readyz storage not writable: %s", data_root)
        return {"status": "degraded", "reason": "storage not writable"}

    # Check disk space
    free = _free_gb(root)
    if free < float(min_free_gb):
        _logger.warning("readyz low disk: %.2f GB < %d GB", free, min_free_gb)
        return {"status": "degraded", "reason": "low disk"}

    return {"status": "ready", "reason": None}


__all__ = [
    "healthz_endpoint",
    "readyz_endpoint",
]
