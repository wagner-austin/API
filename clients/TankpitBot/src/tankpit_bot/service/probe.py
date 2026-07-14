"""HTTP existence probe for the bot service.

Separated from :mod:`tankpit_bot.service._test_hooks` because the probe
carries a testable core (:func:`probe_health_url`) parameterised on
URL — the hooks module only cares about the nullary
``ProbeExistingInstanceProtocol`` binding. Keeping the utility in its
own module gives it a single-concern home and drops the hook module
back to pure DI plumbing.

Production callers use :func:`default_probe_existing_instance`, which
targets :data:`HEALTH_URL`. Tests use :func:`probe_health_url` directly
with a fixture-owned test server URL — see
:mod:`tests.service.test_probe`.
"""

from __future__ import annotations

from http.client import HTTPConnection, HTTPException
from urllib.parse import urlparse

from platform_core.logging import get_logger

from tankpit_bot.service.constants import HEALTH_URL

log = get_logger(__name__)


# ``HEALTH_URL`` re-exported for callers that only import from
# :mod:`tankpit_bot.service.probe`. Every value lives in
# :mod:`tankpit_bot.service.constants` — single source of truth.


def probe_health_url(url: str) -> bool:
    """Return True when ``url`` responds with ``200 ok``.

    Uses stdlib :mod:`http.client` (not :func:`urllib.request.urlopen`)
    because ``urlopen``'s return type collapses to ``Any`` under
    strict mypy — its context-manager surface is polymorphic across
    every URL scheme it supports. :class:`http.client.HTTPConnection`
    exposes an :class:`~http.client.HTTPResponse` with typed
    ``status`` / ``read`` so every attribute resolves to a concrete
    type.

    The 1-second timeout keeps double-tap latency low: a live service
    answers within milliseconds; nothing on the port fails fast with
    :class:`ConnectionRefusedError`.

    Args:
        url: Health-endpoint URL to probe. Production wires this to
            :data:`HEALTH_URL`; tests wire it to a fixture-owned
            aiohttp test server on a random port.

    Returns:
        True when the endpoint returns ``200`` with the exact body
        ``"ok"`` — the marker we control end-to-end so a random HTTP
        server on the port does not fool the probe. False on any
        connection error, timeout, non-200 status, or a body that
        does not match.

    Raises:
        ValueError: When the URL has no host component (a caller
            bug, not an environmental failure — bubbles up so tests
            catch it and the operator sees the misconfiguration).
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        raise ValueError(f"probe URL missing host: {url!r}")
    port = parsed.port if parsed.port is not None else 80
    path = parsed.path if parsed.path != "" else "/"
    conn = HTTPConnection(host, port, timeout=1.0)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        if response.status != 200:
            return False
        body = response.read().decode("utf-8", errors="replace").strip()
        return body == "ok"
    except (OSError, HTTPException) as exc:
        log.debug("existence probe unreachable (%s): %s", url, exc)
        return False
    finally:
        conn.close()


def default_probe_existing_instance() -> bool:
    """Production probe fixed to :data:`HEALTH_URL`.

    Trivial delegate — split out so
    :data:`tankpit_bot.service._test_hooks.probe_existing_instance`
    can bind to it as a nullary callable matching
    :class:`~tankpit_bot.service._test_hooks.ProbeExistingInstanceProtocol`.

    Returns:
        See :func:`probe_health_url`.
    """
    return probe_health_url(HEALTH_URL)


__all__ = [
    "HEALTH_URL",
    "default_probe_existing_instance",
    "probe_health_url",
]
