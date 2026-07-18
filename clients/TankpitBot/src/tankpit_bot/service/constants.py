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

HEALTH_URL = f"http://127.0.0.1:{SERVICE_PORT}/health"
"""Fixed loopback + port + path the existence-probe targets.

Used by :func:`tankpit_bot.service.probe.default_probe_existing_instance`.
Tests target a fixture-owned aiohttp test server on a random port
via :func:`tankpit_bot.service.probe.probe_health_url` — never this
URL — so tests stay stable regardless of whether a real service
happens to be running on the developer's machine.
"""

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
    "HEALTH_URL",
    "SERVICE_HOST",
    "SERVICE_IDLE_EXIT_SECONDS",
    "SERVICE_IDLE_POLL_SECONDS",
    "SERVICE_PORT",
]
