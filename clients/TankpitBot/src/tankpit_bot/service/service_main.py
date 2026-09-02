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

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.bot.config import (
    resolve_prefer_account,
    resolve_target_url,
)
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.bus.frame_bus import FrameBus, FrameBusProtocol
from tankpit_bot.bus.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.bus.session_status import idle_session_status
from tankpit_bot.bus.status_bus import StatusBus, StatusBusProtocol
from tankpit_bot.runtime_artifacts import resolve_bot_instance
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.config import resolve_idle_exit_seconds
from tankpit_bot.service.constants import (
    SERVICE_HOST,
    SERVICE_IDLE_EXIT_SECONDS,
    SERVICE_IDLE_POLL_SECONDS,
    resolve_service_port,
)
from tankpit_bot.service.http_server import SessionRunnerHTTPProtocol, make_app
from tankpit_bot.service.serving import run_until_stopped
from tankpit_bot.service.session_runner import SessionRunner

log = get_logger(__name__)


def resolve_service_stop_file() -> Path:
    """Return this instance's stop-file sentinel path.

    Instance-scoped (2026-08-06, the two-bots-one-map lift): two
    services must not share one sentinel, or stopping one bot stops
    both. The sole-bot namespace keeps ``runs/state/STOP``.

    Returns:
        The sentinel path for this process's instance.
    """
    instance = resolve_bot_instance()
    state_dir = Path("runs/state") / instance if instance else Path("runs/state")
    return state_dir / "STOP"


async def exit_when_idle(
    runner: SessionRunnerHTTPProtocol,
    status_bus: StatusBusProtocol,
    frame_bus: FrameBusProtocol,
    stop_event: asyncio.Event,
    *,
    idle_exit_seconds: float = SERVICE_IDLE_EXIT_SECONDS,
    poll_seconds: float = SERVICE_IDLE_POLL_SECONDS,
) -> None:
    """Set ``stop_event`` after a sustained stretch of total idleness.

    "Idle" means no session running AND no SSE subscriber AND no video
    viewer — nobody is using the service and nobody is even watching
    it. The idle clock resets whenever any condition breaks, so an
    operator staring at the stats strip (SSE open) or the live video
    (``/video`` open) keeps the service alive indefinitely. Part of
    the 2026-07-18 lifecycle pass: the phone's START SERVER button
    relaunches in ~2 s, so an abandoned server has no reason to
    outlive its last viewer by more than this window.

    Args:
        runner: Session runner whose ``is_running`` gates the clock.
        status_bus: Bus whose ``subscriber_count`` gates the clock.
        frame_bus: Video-frame bus whose ``subscriber_count`` also
            gates the clock (2026-07-28 watch page).
        stop_event: The service main's shutdown signal.
        idle_exit_seconds: Sustained idle seconds before exit. A
            non-positive value DISABLES the idle self-exit — the
            always-on deployment (2026-07-29): the SPA's tankpit
            video is served by this process, so the startup launcher
            runs it with the exit off and the monitor returns
            immediately.
        poll_seconds: Cadence of the idleness checks.
    """
    if idle_exit_seconds <= 0:
        log.info(
            "Idle self-exit disabled (threshold %.0f); service runs until stopped.",
            idle_exit_seconds,
        )
        return
    idle_elapsed = 0.0
    while not stop_event.is_set():
        await asyncio.sleep(poll_seconds)
        if (
            runner.is_running()
            or status_bus.subscriber_count() > 0
            or frame_bus.subscriber_count() > 0
        ):
            idle_elapsed = 0.0
            continue
        idle_elapsed += poll_seconds
        if idle_elapsed >= idle_exit_seconds:
            log.info(
                "No session and no SSE subscriber for %.0f s; exiting idle service.",
                idle_elapsed,
            )
            stop_event.set()
            return


async def _async_main(host: str = SERVICE_HOST, port: int | None = None) -> None:
    """Wire the primitives, publish an initial idle frame, and serve forever.

    Site construction goes through :data:`service_hooks.build_site` so
    tests can inject a fake site without opening a real port.

    Args:
        host: Bind host for the HTTP server.
        port: TCP port to bind; ``None`` resolves this instance's
            port from the environment (``resolve_service_port``).
    """
    bound_port = resolve_service_port() if port is None else port
    mode_bridge: ModeBridgeProtocol = ModeBridge()
    status_bus: StatusBusProtocol = StatusBus()
    frame_bus: FrameBusProtocol = FrameBus()
    runner = SessionRunner(
        bot_factory=service_hooks.build_bot_factory(
            resolve_target_url(),
            headless=False,
            prefer_account=resolve_prefer_account(),
        ),
        mode_bridge=mode_bridge,
        status_bus=status_bus,
        frame_bus=frame_bus,
        stop_file_path=resolve_service_stop_file(),
    )
    status_bus.publish(idle_session_status(get_current_time_ms()))
    stop_event = asyncio.Event()
    app = make_app(runner, mode_bridge, status_bus, frame_bus, stop_event.set)
    site = await service_hooks.build_site(app, host, bound_port)
    idle_monitor = asyncio.create_task(
        exit_when_idle(
            runner,
            status_bus,
            frame_bus,
            stop_event,
            idle_exit_seconds=resolve_idle_exit_seconds(),
        )
    )
    try:
        await run_until_stopped(site, stop_event, name="Bot service")
    finally:
        idle_monitor.cancel()


def main() -> None:
    """Console entry point for ``tankpit-bot-service``.

    Loads the ``.env`` file via :data:`core_hooks.load_dotenv`,
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
    core_hooks.load_dotenv()
    if service_hooks.probe_existing_instance():
        log.info(
            "tankpit-bot-service already responding on %s:%d; exiting idempotently.",
            SERVICE_HOST,
            resolve_service_port(),
        )
        return
    try:
        service_hooks.serve()
    except KeyboardInterrupt:
        log.info("Bot service interrupted")


__all__ = [
    "exit_when_idle",
    "main",
]
