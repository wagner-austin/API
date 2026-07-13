"""Long-running entry point for the bot service.

Runs an aiohttp server on the loopback interface (nginx proxies
``/api/tankbot/*`` from the fiesta docker container). Owns exactly
one :class:`SessionRunner`, one shared :class:`ModeBridge`, and one
shared :class:`StatusBus` — the same three primitives the tick loop
publishes into whenever a session is active.

Every non-pure operation (aiohttp construction, ``.env`` loading,
``asyncio.run`` entry, bot construction) is routed through
:mod:`tankpit_bot.service._test_hooks` so tests can substitute fakes
without conditionals in the service code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot.bot.config import resolve_prefer_account, resolve_target_url
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import SiteRunnerProtocol
from tankpit_bot.service.http_server import make_app
from tankpit_bot.service.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.service.session_runner import SessionRunner
from tankpit_bot.service.status_bus import StatusBus, StatusBusProtocol
from tankpit_bot.service.types import idle_session_status

log = get_logger(__name__)

# Bind on every interface so the fiesta docker container's nginx
# can proxy ``/api/tankbot/*`` to the host — the container reaches
# the host via the host's Tailscale IPv4 (100.77.206.124), not
# loopback, because docker networking on Windows/WSL2 cannot
# forward-to-loopback reliably (see MCPs/fiesta/nginx.conf for the
# 2026-07-01/02 history that led to the Tailscale-IP proxy_pass).
# Trust boundary: the machine's LAN + the operator's Tailnet — the
# same boundary Vibeshine already accepts on 47990.
_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 47100
_DEFAULT_STOP_FILE = Path("runs/state/STOP")


async def run_service_forever(
    site: SiteRunnerProtocol,
    stop_event: asyncio.Event,
) -> None:
    """Serve until ``stop_event`` is set, then tear the site down.

    ``site.cleanup`` runs even when ``site.start`` raises — the site's
    ``AppRunner`` may have partially set up (opened the socket, wired
    handlers) before start failed, and cleanup is idempotent. The
    ``finally`` wraps both stages for that reason.

    Args:
        site: The aiohttp site backing the HTTP surface.
        stop_event: Signal the caller sets to request shutdown —
            typically wired to SIGINT / SIGTERM handlers or a test
            harness that just calls ``stop_event.set()``.
    """
    try:
        await site.start()
        log.info("Bot service ready")
        await stop_event.wait()
    finally:
        log.info("Bot service shutting down")
        await site.cleanup()


async def _async_main(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> None:
    """Wire the primitives, publish an initial idle frame, and serve forever.

    Site construction goes through :data:`service_hooks.build_site` so
    tests can inject a fake site without opening a real port.

    Args:
        host: Loopback host the HTTP server binds to.
        port: TCP port the HTTP server binds to.
    """
    mode_bridge: ModeBridgeProtocol = ModeBridge()
    status_bus: StatusBusProtocol = StatusBus()
    runner = SessionRunner(
        bot_factory=service_hooks.build_bot_factory(
            resolve_target_url(),
            headless=False,
            prefer_account=resolve_prefer_account(),
        ),
        mode_bridge=mode_bridge,
        status_bus=status_bus,
        stop_file_path=_DEFAULT_STOP_FILE,
    )
    status_bus.publish(idle_session_status(get_current_time_ms()))
    app = make_app(runner, mode_bridge, status_bus)
    site = await service_hooks.build_site(app, host, port)
    stop_event = asyncio.Event()
    await run_service_forever(site, stop_event)


def main() -> None:
    """Console entry point for ``tankpit-bot-service``.

    Loads the ``.env`` file via :data:`service_hooks.load_dotenv`,
    probes for an already-running instance via
    :data:`service_hooks.probe_existing_instance`, and — only if no
    other instance is answering — runs the service under
    :data:`service_hooks.serve` (defaults to :func:`asyncio.run` around
    :func:`_async_main`).

    The probe short-circuit makes the entry-point idempotent: a
    second ``make service`` (or a double-tap of the phone's SERVER
    button, once Phase C.2 lands) exits cleanly with a "already
    running" log line instead of crash-looping on the port-bind
    conflict that the previous respawn loop retried every 5 s.

    A ``KeyboardInterrupt`` unwinds cleanly — the aiohttp
    ``AppRunner``'s cleanup handles socket teardown.
    """
    service_hooks.load_dotenv()
    if service_hooks.probe_existing_instance():
        log.info(
            "tankpit-bot-service already responding on %s:%d; exiting idempotently.",
            _DEFAULT_HOST,
            _DEFAULT_PORT,
        )
        return
    try:
        service_hooks.serve()
    except KeyboardInterrupt:
        log.info("Bot service interrupted")


__all__ = [
    "main",
    "run_service_forever",
]
