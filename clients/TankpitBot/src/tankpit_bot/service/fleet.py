"""AI-operated bot fleet: the ``tankpit-fleet`` entry point.

Built 2026-08-06 for the operating AI rather than the SPA: it runs in a
terminal the operator owns, so an orchestration harness dying can never
kill a live tank, and the AI drives it with plain HTTP.

This module is the composition root and holds nothing else -- the
registry is :mod:`tankpit_bot.service.fleet_manager`, the routes are
:mod:`tankpit_bot.service.fleet_routes`, and the serve loop is
:mod:`tankpit_bot.service.serving`.

**Lifecycle (2026-09-01).** The manager used to block in
``aiohttp.web.run_app``, which cannot be asked to stop, so its only
exit was to abandon whatever it had spawned. Every restart therefore
left orphans: bots still playing, still burning fuel, with nothing
able to see or stop them. Now the manager owns a drain.

* An interrupt, or ``POST /shutdown``, asks every live bot to stop and
  the manager KEEPS SERVING while they tear down. It exits after the
  last one is gone, which is the only moment at which exiting orphans
  nothing.
* Draining never kills. Each bot ends through the same stop sentinel a
  bounded session ends on, so it writes its scorecard and quits to the
  lobby rather than being cut down mid-game with its rank on the line.
* If the manager dies anyway -- a crash, a killed terminal -- the bots
  it spawned survive by design, and the NEXT manager adopts them from
  their spawn records (:mod:`tankpit_bot.service.fleet_adoption`).
  Between the drain and adoption there is no state in which a running
  bot is unreachable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.fleet_config import resolve_fleet_port
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_routes import make_fleet_app
from tankpit_bot.service.serving import cancel_and_await, run_until_stopped

log = get_logger(__name__)

#: Loopback only: the fleet spawns processes and hands out control of
#: live accounts, so it is never exposed off the machine.
FLEET_HOST_DEFAULT = "127.0.0.1"


def resolve_fleet_host() -> str:
    """Resolve the fleet manager's bind address from the environment.

    ``TANKPIT_FLEET_HOST`` exists for the fleet CONTAINER, where the
    process must bind ``0.0.0.0`` to be reachable through the
    published port — the loopback-only security property moves to the
    docker port mapping (``127.0.0.1:27300:27300``), which is the
    container boundary's equivalent of the default here. On a host
    run, leave it unset.

    Returns:
        ``TANKPIT_FLEET_HOST`` when set, else
        :data:`FLEET_HOST_DEFAULT`.
    """
    raw = core_hooks.get_env("TANKPIT_FLEET_HOST")
    if raw is None or raw == "":
        return FLEET_HOST_DEFAULT
    return raw


#: How often the drain monitor re-checks whether the last bot is gone.
FLEET_DRAIN_POLL_SECONDS = 1.0


async def exit_when_drained(
    manager: FleetManager,
    stop_event: asyncio.Event,
    *,
    poll_seconds: float = FLEET_DRAIN_POLL_SECONDS,
) -> None:
    """Stop the manager once a requested drain has finished.

    Waits without a deadline, deliberately. A bot's teardown ends with
    quitting to the lobby, and cutting that short to meet a timeout is
    how a tank gets left exposed in a live game. So the manager stays
    up for as long as its last child needs, reporting who it is still
    waiting on.

    Args:
        manager: The registry whose bots are draining.
        stop_event: The serve loop's shutdown signal.
        poll_seconds: Cadence of the "is anyone left" checks.

    Returns:
        None. Returns once the drain is complete, or never, if it was
        never requested.
    """
    while not stop_event.is_set():
        await asyncio.sleep(poll_seconds)
        if not manager.draining():
            continue
        live = manager.live_instances()
        if live:
            log.info(
                "Fleet: draining — waiting on %d bot(s): %s",
                len(live),
                ", ".join(live),
            )
            continue
        log.info("Fleet: drain complete; every bot has torn down")
        stop_event.set()
        return


def drain_on_interrupt(manager: FleetManager) -> Callable[[], None]:
    """Build the interrupt handler that drains instead of exiting.

    Ctrl+C on the fleet is a request to shut the FLEET down, and the
    fleet is its bots. Exiting immediately would leave them running
    with no supervisor, so the interrupt starts the drain and the
    manager stays up until it finishes.

    Args:
        manager: The registry to drain.

    Returns:
        A zero-argument handler for SIGINT and SIGTERM.
    """

    def handle() -> None:
        """Start the drain, or report what it is still waiting on."""
        live = manager.live_instances()
        if manager.draining():
            log.info(
                "Fleet: already draining — still waiting on %d bot(s): %s. "
                "They are quitting to the lobby; the manager exits when they finish.",
                len(live),
                ", ".join(live) or "none",
            )
            return
        log.info("Fleet: interrupt received — draining %d bot(s) before exit", len(live))
        manager.request_drain()

    return handle


async def _async_main() -> None:
    """Adopt, serve, and drain: the manager's whole life.

    Returns:
        None. Returns once the site has been torn down.
    """
    port = resolve_fleet_port()
    manager = FleetManager()
    manager.adopt()
    stop_event = asyncio.Event()
    host = resolve_fleet_host()
    site = await service_hooks.build_site(make_fleet_app(manager), host, port)
    drain_monitor = asyncio.create_task(exit_when_drained(manager, stop_event))
    core_hooks.install_signal_handlers(drain_on_interrupt(manager))
    log.info("tankpit-fleet listening on %s:%d", host, port)
    try:
        await run_until_stopped(site, stop_event, name="Fleet manager")
    finally:
        await cancel_and_await(drain_monitor)


def main() -> None:
    """Run the ``tankpit-fleet`` manager until its bots are done.

    Returns:
        None.
    """
    core_hooks.load_dotenv()
    try:
        service_hooks.serve_fleet()
    except KeyboardInterrupt:
        # Only reachable in the sliver before the drain handler is
        # installed, when this manager has spawned nothing. Anything
        # it adopted keeps its spawn record, so the next manager finds
        # it again — the tanks are never the thing that is lost here.
        log.info("fleet manager stopped before it began serving (Ctrl+C)")


__all__ = [
    "FLEET_DRAIN_POLL_SECONDS",
    "FLEET_HOST_DEFAULT",
    "drain_on_interrupt",
    "exit_when_drained",
    "log",
    "main",
    "resolve_fleet_host",
]
