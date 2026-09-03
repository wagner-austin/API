"""Shared numeric contract for the bot-service package.

Owns every fixed number that has to agree across the service's own
sub-modules (server bind, probe URL, tests). One module, one home —
no literal drift between :mod:`service_main`, :mod:`probe`, and the
callers that key off the same port.

The one out-of-tree consumer that MUST also match this port is
``MCPs/fiesta/nginx.conf``'s ``proxy_pass`` line. nginx cannot import
Python; if this port ever changes, nginx.conf changes with it. A
comment there flags this coupling.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks

SERVICE_HOST = "0.0.0.0"
"""Bind interface for the aiohttp server.

Every interface, not loopback: the fiesta docker container's nginx
reaches this service via the host's Tailscale IPv4 (100.77.206.124),
which is a non-loopback interface. See
:mod:`tankpit_bot.service.service_main` for the trust-boundary
rationale.
"""

SERVICE_PORT = 27100
"""TCP port the aiohttp server binds to.

Chosen 2026-07-13 after 47100 became unreliable — Windows dynamic
port reservations (Hyper-V / WSL / Docker Desktop) cover chunks of
the 40000-60000 range at boot, visible via ``PermissionError`` on
bind but NOT via ``netsh interface ipv4 show excludedportrange``.
27100 sits well below that range so it stays bindable across boots.
"""


FLEET_CHILD_PORT_BASE = 27101
"""First port the manager may hand to a fleet child's service.

One above :data:`SERVICE_PORT` so a standalone ``tankpit-bot-service``
and a fleet child can coexist on one host without contending for a
port. Children bind inside the manager's own container and are reached
through it, so nothing in this range is ever published.
"""

FLEET_CHILD_PORT_COUNT = 32
"""How many child ports the manager may hand out.

A ceiling, not a target: the fleet is bounded by accounts long before
it is bounded by ports. Exhausting the range is an error rather than a
wrap-around, because two children sharing a port would silently serve
each other's video.
"""


def resolve_service_port() -> int:
    """Resolve this process's service port from the environment.

    ``TANKPIT_BOT_SERVICE_PORT`` lets a second bot instance run its
    own service beside the first (2026-08-06, the two-bots-one-map
    lift). Unset or empty means :data:`SERVICE_PORT` — the port the
    fiesta nginx proxy targets; a second instance is curl-only unless
    nginx grows a matching location block.

    Returns:
        The validated TCP port.

    Raises:
        ValueError: If the value is not an integer in [1024, 65535].
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_SERVICE_PORT")
    if raw is None or raw == "":
        return SERVICE_PORT
    port = int(raw)
    if not 1024 <= port <= 65535:
        raise ValueError(f"TANKPIT_BOT_SERVICE_PORT {port} outside [1024, 65535]")
    return port


def health_url(port: int) -> str:
    """Return the loopback health-probe URL for one service port.

    Args:
        port: The service's resolved TCP port.

    Returns:
        The ``/health`` URL the existence probe targets. Tests target
        a fixture-owned aiohttp test server on a random port via
        :func:`tankpit_bot.service.probe.probe_health_url` — never
        this URL — so they stay stable regardless of whether a real
        service is running on the developer's machine.
    """
    return f"http://127.0.0.1:{port}/health"


def child_video_url(port: int) -> str:
    """Return the loopback MJPEG URL for one fleet child's service.

    Loopback because a child binds inside the manager's own container
    and is reached only by the manager relaying to it. The address is
    not configurable for that reason: a child on another host is not a
    child this manager spawned.

    Args:
        port: The child's allocated service port.

    Returns:
        The child's ``/video`` URL.
    """
    return f"http://127.0.0.1:{port}/video"


SERVICE_IDLE_EXIT_SECONDS = 1800.0
"""Idle self-exit threshold (2026-07-18 lifecycle pass).

The service exits on its own after this many seconds with no active
session AND no SSE subscriber — the phone's START SERVER button
relaunches it in ~2 s (idempotent probe), so an idle server is pure
waste. Before this existed the server ran forever once started; the
only stop was closing its cmd window on the PC.
"""

SERVICE_IDLE_POLL_SECONDS = 60.0
"""Cadence of the idle-exit monitor's liveness checks."""


__all__ = [
    "FLEET_CHILD_PORT_BASE",
    "FLEET_CHILD_PORT_COUNT",
    "SERVICE_HOST",
    "SERVICE_IDLE_EXIT_SECONDS",
    "SERVICE_IDLE_POLL_SECONDS",
    "SERVICE_PORT",
    "child_video_url",
    "health_url",
    "resolve_service_port",
]
