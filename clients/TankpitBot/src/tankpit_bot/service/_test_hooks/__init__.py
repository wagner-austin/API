"""Dependency-injection hooks internal to the service package.

Every non-pure operation the bot service and the fleet manager depend
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

Split into a package 2026-09-01 at the 600-line ceiling, by role
rather than by size, mirroring the top-level ``_test_hooks`` tree:

* :mod:`~tankpit_bot.service._test_hooks.serving` — getting an aiohttp
  site and running it.
* :mod:`~tankpit_bot.service._test_hooks.processes` — spawning bots,
  and re-attaching to the ones a previous manager spawned.
* :mod:`~tankpit_bot.service._test_hooks.bots` — constructing the bot
  a session runs.

Every name stays importable from ``tankpit_bot.service._test_hooks``
exactly as before, and a test that swaps one here swaps what
production reads: production calls ``service_hooks.<name>``, which
resolves through this module.
"""

from __future__ import annotations

from tankpit_bot.service._test_hooks.bots import (
    BotFactoryBuilderProtocol,
    _real_build_bot_factory,
    build_bot_factory,
)
from tankpit_bot.service._test_hooks.processes import (
    OpenAdoptedProcessProtocol,
    ProcessIdentityProtocol,
    SleepSecondsProtocol,
    SpawnBotProcessProtocol,
    SpawnedProcessProtocol,
    _AdoptedProcess,
    _real_open_adopted_process,
    _real_process_identity,
    _real_sleep_seconds,
    _real_spawn_bot_process,
    open_adopted_process,
    process_identity,
    sleep_seconds,
    spawn_bot_process,
)
from tankpit_bot.service._test_hooks.serving import (
    ProbeExistingInstanceProtocol,
    ServeProtocol,
    SiteFactoryProtocol,
    SiteRunnerProtocol,
    _AiohttpSite,
    _real_build_site,
    _real_serve,
    _real_serve_fleet,
    build_site,
    probe_existing_instance,
    serve,
    serve_fleet,
)

__all__ = [
    "BotFactoryBuilderProtocol",
    "OpenAdoptedProcessProtocol",
    "ProbeExistingInstanceProtocol",
    "ProcessIdentityProtocol",
    "ServeProtocol",
    "SiteFactoryProtocol",
    "SiteRunnerProtocol",
    "SleepSecondsProtocol",
    "SpawnBotProcessProtocol",
    "SpawnedProcessProtocol",
    "_AdoptedProcess",
    "_AiohttpSite",
    "_real_build_bot_factory",
    "_real_build_site",
    "_real_open_adopted_process",
    "_real_process_identity",
    "_real_serve",
    "_real_serve_fleet",
    "_real_sleep_seconds",
    "_real_spawn_bot_process",
    "build_bot_factory",
    "build_site",
    "open_adopted_process",
    "probe_existing_instance",
    "process_identity",
    "serve",
    "serve_fleet",
    "sleep_seconds",
    "spawn_bot_process",
]
