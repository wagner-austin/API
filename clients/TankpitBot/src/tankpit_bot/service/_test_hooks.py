"""Dependency-injection hooks internal to the service package.

Every non-pure operation :mod:`tankpit_bot.service.service_main` depends
on is exposed here as a module-level symbol assigned to a real
implementation. Production code (imported once at boot) uses the
default assignments; tests reassign the symbol to a fake for the
duration of a test and restore it in teardown.

The pattern is unconditional — the service code always calls the hook
directly, never a real function guarded by ``if TESTING``.

Kept inside the service package (rather than the top-level
:mod:`tankpit_bot._test_hooks` tree) because the hook Protocols
reference :mod:`tankpit_bot.service.types`, which transitively pulls
:mod:`tankpit_bot.types.modes`. Loading that during the top-level
``_test_hooks`` init would cycle through ``bot.ai.combat_landing`` →
``_test_hooks.TerrainMapProtocol``. Locating the service hooks inside
the service tree keeps the import graph acyclic.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from typing import Protocol

from aiohttp import web

from tankpit_bot.runtime_artifacts import bot_run_dir
from tankpit_bot.service.probe import default_probe_existing_instance
from tankpit_bot.service.session_runner import BotFactoryProtocol


class BotFactoryBuilderProtocol(Protocol):
    """Builds a :class:`BotFactoryProtocol` from session-level configuration."""

    def __call__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
    ) -> BotFactoryProtocol:
        """Return a bot factory bound to the requested session config.

        Args:
            target_url: URL the bot navigates to on session start.
            headless: Whether the launched Chromium runs headless.
            prefer_account: Whether the bot uses account credentials
                instead of guest login.

        Returns:
            A callable that :class:`SessionRunner` invokes once per
            session with a shared bridge + bus.
        """
        ...


class ProbeExistingInstanceProtocol(Protocol):
    """Detects whether another :func:`main` invocation is already serving.

    :func:`main` calls this before :data:`serve` so a double-invocation
    (double-tap of the phone's SERVER button, a second ``make service``
    while the first is still up) is idempotent — the second instance
    exits cleanly instead of crash-looping while the outer respawn loop
    fights the port-bind conflict every 5 s.
    """

    def __call__(self) -> bool:
        """Return True when a tankpit-bot-service instance already responds.

        Returns:
            True when ``GET http://127.0.0.1:27100/health`` returns
            ``200 ok``; False when the port is idle OR another
            (non-us) process holds the port and does not answer
            ``/health`` with the expected body.
        """
        ...


class ServeProtocol(Protocol):
    """Runs the bot service until it exits (defaults to :func:`asyncio.run`).

    Nullary by design: the hook does not accept a coroutine argument
    because a strict-typed :class:`Awaitable` / :class:`Coroutine`
    signature cannot be passed to :func:`asyncio.run` without leaking
    ``Any`` through mypy's generic constraints. Production wires the
    hook to a factory that builds + drives the coroutine internally.
    """

    def __call__(self) -> None:
        """Run the bot service to completion or fail loudly."""
        ...


class RunWebAppProtocol(Protocol):
    """Blocking web-app runner the fleet entry point delegates to."""

    def __call__(self, app: web.Application, *, host: str, port: int) -> None:
        """Serve the application until the process is interrupted.

        Args:
            app: Routed application to serve.
            host: Bind host.
            port: TCP port to bind.
        """
        ...


class SiteRunnerProtocol(Protocol):
    """Aiohttp lifecycle handle :func:`run_service_forever` consumes."""

    async def start(self) -> None:
        """Begin serving. Non-blocking; the event loop keeps running."""
        ...

    async def cleanup(self) -> None:
        """Tear the site + underlying application runner down cleanly."""
        ...


class SiteFactoryProtocol(Protocol):
    """Async factory that produces a started-ready site for an app."""

    async def __call__(
        self,
        app: web.Application,
        host: str,
        port: int,
    ) -> SiteRunnerProtocol:
        """Set up aiohttp and return the merged ``SiteRunnerProtocol``.

        Args:
            app: Routed :class:`web.Application` to serve.
            host: Loopback host to bind to.
            port: TCP port to bind to.

        Returns:
            A :class:`SiteRunnerProtocol` whose ``start`` / ``cleanup``
            :func:`run_service_forever` drives.
        """
        ...


class _AppRunnerCleanupProtocol(Protocol):
    """Minimum surface :class:`_AiohttpSite` needs from :class:`web.AppRunner`."""

    async def cleanup(self) -> None:
        """Tear the application runner down."""
        ...


class _TCPSiteStartProtocol(Protocol):
    """Minimum surface :class:`_AiohttpSite` needs from :class:`web.TCPSite`."""

    async def start(self) -> None:
        """Begin serving on the site's host + port."""
        ...


class _AiohttpSite:
    """Adapter matching :class:`SiteRunnerProtocol` for aiohttp's split API.

    aiohttp splits the lifecycle across :class:`web.AppRunner` (setup +
    cleanup) and :class:`web.TCPSite` (start + stop). The service main's
    :class:`SiteRunnerProtocol` merges the two so
    :func:`run_service_forever` sees a single start / cleanup surface.
    """

    def __init__(
        self,
        aiohttp_runner: _AppRunnerCleanupProtocol,
        site: _TCPSiteStartProtocol,
    ) -> None:
        """Store both aiohttp handles for the merged lifecycle.

        Args:
            aiohttp_runner: The aiohttp application runner, already
                ``setup``-called by the caller.
            site: The TCP site pinning the app to a host + port.
        """
        self._aiohttp_runner = aiohttp_runner
        self._site = site

    async def start(self) -> None:
        """Begin serving on the site's host + port."""
        await self._site.start()

    async def cleanup(self) -> None:
        """Tear down the site and its underlying application runner."""
        await self._aiohttp_runner.cleanup()


async def _real_build_site(
    app: web.Application,
    host: str,
    port: int,
) -> SiteRunnerProtocol:
    """Production site factory — wires aiohttp's split AppRunner + TCPSite.

    ``reuse_address=True`` is passed to :class:`web.TCPSite` so a
    restart within Windows' ~120 s ``TIME_WAIT`` window succeeds
    instead of failing with ``WinError 10048`` (EADDRINUSE). Every
    Ctrl+C exit + immediate ``make service`` restart pattern hits
    this if the flag is missing; the flag maps to the underlying
    ``SO_REUSEADDR`` socket option, which is the standard fix.

    Args:
        app: Routed :class:`web.Application` to serve.
        host: Loopback host to bind to.
        port: TCP port to bind to.

    Returns:
        The aiohttp AppRunner + TCPSite pair merged into a
        :class:`SiteRunnerProtocol` via :class:`_AiohttpSite`.
    """
    aiohttp_runner = web.AppRunner(app)
    await aiohttp_runner.setup()
    return _AiohttpSite(
        aiohttp_runner,
        web.TCPSite(aiohttp_runner, host, port, reuse_address=True),
    )


def _real_serve() -> None:
    """Production entry point — drives ``_async_main`` under :func:`asyncio.run`.

    Kept inside the hook so the coroutine object never crosses a
    Protocol boundary and mypy sees a plain nullary callable.
    """
    from tankpit_bot.service.service_main import _async_main

    asyncio.run(_async_main())


def _real_run_web_app(app: web.Application, *, host: str, port: int) -> None:
    """Production runner — :func:`aiohttp.web.run_app` until interrupted.

    Args:
        app: Routed application to serve.
        host: Bind host.
        port: TCP port to bind.
    """
    web.run_app(app, host=host, port=port)


def _real_build_bot_factory(
    target_url: str, *, headless: bool, prefer_account: bool
) -> BotFactoryProtocol:
    """Production bot factory — constructs a real :class:`Bot` per session.

    Args:
        target_url: URL the bot navigates to on session start.
        headless: Whether the launched Chromium runs headless.
        prefer_account: Whether the bot uses account credentials
            instead of guest login.

    Returns:
        A :class:`BotFactoryProtocol` callable that
        :class:`SessionRunner` invokes once per session.
    """
    from tankpit_bot.bot.base import Bot
    from tankpit_bot.bus.frame_bus import FrameBusProtocol
    from tankpit_bot.bus.mode_bridge import ModeBridgeProtocol
    from tankpit_bot.bus.status_bus import StatusBusProtocol
    from tankpit_bot.service.session_runner import RunnableBotProtocol

    def factory(
        *,
        mode_bridge: ModeBridgeProtocol,
        status_bus: StatusBusProtocol,
        frame_bus: FrameBusProtocol,
    ) -> RunnableBotProtocol:
        return Bot(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            mode_bridge=mode_bridge,
            status_bus=status_bus,
            frame_bus=frame_bus,
        )

    return factory


#: Site-construction hook — production wires aiohttp; tests inject a
#: fake that returns a recording :class:`SiteRunnerProtocol` without
#: opening a socket.
build_site: SiteFactoryProtocol = _real_build_site

#: ``.env`` loader hook — production reads the real ``.env`` file;
#: tests replace with a no-op so the process env stays clean.
#: Existence-probe hook — production sends a live HTTP GET to the
#: expected health endpoint (see :mod:`tankpit_bot.service.probe`);
#: tests inject a scriptable double that returns True or False to
#: exercise the short-circuit + normal paths of :func:`main` without
#: touching the network.
probe_existing_instance: ProbeExistingInstanceProtocol = default_probe_existing_instance

#: Service-run hook — production drives ``_async_main`` under
#: :func:`asyncio.run`; tests replace with a fake that either returns
#: normally or raises :class:`KeyboardInterrupt` to exercise
#: :func:`main`'s interrupt branch without a real event loop.
serve: ServeProtocol = _real_serve

#: Bot-factory builder hook — production constructs a real
#: :class:`Bot`; tests replace with a fake that produces a
#: :class:`RunnableBotProtocol` stub.
build_bot_factory: BotFactoryBuilderProtocol = _real_build_bot_factory


class SpawnedProcessProtocol(Protocol):
    """The child-process surface the fleet manager consumes."""

    @property
    def pid(self) -> int:
        """The child's process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code, or None while the child runs."""
        ...


class SpawnBotProcessProtocol(Protocol):
    """Spawns one bot child process for the fleet manager."""

    def __call__(self, env_overrides: dict[str, str]) -> SpawnedProcessProtocol:
        """Start a bot child with the given environment overrides.

        Args:
            env_overrides: Variables set in the child's environment.

        Returns:
            The spawned process handle.
        """
        ...


#: Bootstrap the fleet child runs: apply ``KEY=VALUE`` argv pairs to
#: its OWN environment, then hand off to the bot entry point. The
#: manager never reads the parent environment — the child inherits it
#: whole (``env=None``) and the per-instance overrides ride in as
#: arguments, applied on the far side of the process boundary where
#: the ``get_env`` seam does not exist yet.
_CHILD_BOOTSTRAP = (
    "import os, sys\n"
    "for pair in sys.argv[1:]:\n"
    "    key, _, value = pair.partition('=')\n"
    "    os.environ[key] = value\n"
    "del sys.argv[1:]\n"
    "from tankpit_bot.bot.entry import main\n"
    "main()\n"
)
# The argv wipe matters: the entry point parses sys.argv, and the
# KEY=VALUE pairs are bootstrap freight, not bot arguments — the
# first live fleet spawn died on "unrecognized arguments" without it.


def _real_spawn_bot_process(env_overrides: dict[str, str]) -> subprocess.Popen[bytes]:
    """Spawn one ``tankpit-bot`` child with instance environment.

    The child runs the existing bot entry point in its own process;
    the per-instance isolation (artifact namespace, stop sentinel,
    account selection) all lands through the environment.

    The child's stdout and stderr go to its OWN
    ``runs/bot/<instance>/console.log``, never to the manager's
    terminal. Inheriting the console (the behavior until 2026-08-28)
    put every tick line and viewport dump of every bot into the
    ``make fleet`` window — N interleaved streams with no instance
    prefix — duplicating what the bot already writes to
    ``latest.log``, and contradicting this service's own rule that
    the manager owns lifecycle while telemetry stays on disk.

    Redirecting to a FILE rather than discarding matters: the
    interpreter prints an uncaught exception's traceback to stderr as
    the process dies, AFTER the bot's file logging is gone. The
    2026-08-28 bad-password run is the case in point — its
    ``latest.log`` ends at "Login errors: Invalid username or
    password." and the ``GameNotJoinedError`` traceback existed only
    on the console. ``DEVNULL`` would have destroyed the one artifact
    that explained the exit.

    Args:
        env_overrides: Variables set in the child's environment
            (``TANKPIT_BOT_INSTANCE`` and friends), layered over the
            inherited parent environment by the child's bootstrap.

    Returns:
        The spawned process handle.
    """
    pairs = [f"{key}={value}" for key, value in env_overrides.items()]
    console = bot_run_dir(env_overrides.get("TANKPIT_BOT_INSTANCE", "")) / "console.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    # Append, not truncate: a restart must not erase the traceback
    # that explains why the previous run of this instance died.
    with console.open("a", encoding="utf-8") as stream:
        return subprocess.Popen(
            [sys.executable, "-c", _CHILD_BOOTSTRAP, *pairs],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


#: Fleet-manager spawn seam. Tests inject a fake that records env and
#: returns a controllable process double; production spawns the real
#: ``tankpit-bot`` child.
spawn_bot_process: SpawnBotProcessProtocol = _real_spawn_bot_process

#: Fleet-manager serve seam — production blocks in
#: :func:`aiohttp.web.run_app`; tests inject a recorder so
#: ``fleet.main`` is exercised without opening a socket.
run_web_app: RunWebAppProtocol = _real_run_web_app


__all__ = [
    "RunWebAppProtocol",
    "SiteFactoryProtocol",
    "SiteRunnerProtocol",
    "SpawnBotProcessProtocol",
    "SpawnedProcessProtocol",
    "_real_run_web_app",
    "_real_spawn_bot_process",
    "build_bot_factory",
    "build_site",
    "probe_existing_instance",
    "run_web_app",
    "serve",
    "spawn_bot_process",
]
