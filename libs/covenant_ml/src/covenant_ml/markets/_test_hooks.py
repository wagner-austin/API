"""Real I/O implementations, directly replaceable through typed dependency seams."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from os import cpu_count
from time import sleep

from platform_core.environment_record import (
    HostProbe,
    VersionReader,
    stdlib_host_probe,
)
from platform_core.environment_record import (
    installed_version as _installed_version,
)
from platform_core.http_client import HttpxClient, SyncTransport, build_client


def utc_now() -> str:
    """Read the wall clock as an aware UTC ISO timestamp.

    Returns:
        Current UTC timestamp.
    """
    return datetime.now(UTC).isoformat()


clock: Callable[[], str] = utc_now
pause: Callable[[float], None] = sleep
client_factory: Callable[[float, SyncTransport | None], HttpxClient] = build_client
host_probe: HostProbe = stdlib_host_probe(cpu_count)
installed_version: VersionReader = _installed_version


__all__ = ["client_factory", "clock", "host_probe", "installed_version", "pause", "utc_now"]
