from __future__ import annotations

import pytest
from discord.app_commands import AppCommand
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot._test_hooks import BotTreeProto, SnowflakeLike
from clubbot.config import DiscordbotSettings
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator, GuildLike
from clubbot.services.qr.client import QRService


class _GuildImpl:
    """Fake guild for testing that matches GuildLike protocol."""

    __slots__ = ("id", "name")

    def __init__(self, *, id: int, name: str) -> None:
        self.id = id
        self.name = name


class _SyncCall:
    __slots__ = ("guild",)

    def __init__(self, guild: SnowflakeLike | None) -> None:
        self.guild = guild


def _make_cfg(guild_ids: list[int] | None = None) -> DiscordbotSettings:
    ids: list[int] = list(guild_ids) if guild_ids is not None else []
    return build_settings(
        guild_ids=ids,
        qr_default_border=2,
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
    )


def _make_sync_recorder(
    calls: list[_SyncCall],
) -> _test_hooks.TreeSyncFactoryProtocol:
    """Create a tree sync hook that records calls for verification."""

    async def _recording_sync(
        tree: BotTreeProto, guild: SnowflakeLike | None = None
    ) -> list[AppCommand]:
        calls.append(_SyncCall(guild=guild))
        return []

    return _recording_sync


@pytest.mark.asyncio
async def test_on_ready_triggers_sync_commands() -> None:
    cfg = build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
        commands_sync_global=True,
    )

    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()

    calls: list[_SyncCall] = []
    _test_hooks.tree_sync = _make_sync_recorder(calls)

    orch.register_listeners()
    ready = orch._on_ready_listener
    if ready is None:
        raise AssertionError("expected on_ready_listener")
    await ready()

    assert len(calls) == 1
    call_guild = calls[0].guild
    assert call_guild is None


@pytest.mark.asyncio
async def test_sync_commands_global_only() -> None:
    cfg = build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
        commands_sync_global=True,
    )

    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()

    calls: list[_SyncCall] = []
    _test_hooks.tree_sync = _make_sync_recorder(calls)

    await orch.sync_commands()

    assert len(calls) == 1
    assert calls[0].guild is None


@pytest.mark.asyncio
async def test_on_guild_join_no_per_guild_sync() -> None:
    # No per-guild sync should occur on join; global-only model
    cfg = _make_cfg([555])

    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()
    orch.register_listeners()

    calls: list[_SyncCall] = []
    _test_hooks.tree_sync = _make_sync_recorder(calls)

    # Invoke listener directly
    join_listener = orch._on_guild_join_listener
    if join_listener is None:
        raise AssertionError("expected on_guild_join_listener")
    guild: GuildLike = _GuildImpl(id=555, name="Target")
    await join_listener(guild)
    assert calls == []


@pytest.mark.asyncio
async def test_global_sync_runs_once_on_boot() -> None:
    # Create a config with global sync enabled and no target guilds
    cfg = build_settings(
        qr_default_border=2,
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
        commands_sync_global=True,
    )

    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()

    calls: list[_SyncCall] = []
    _test_hooks.tree_sync = _make_sync_recorder(calls)
    orch.register_listeners()

    # First ready should perform a global sync
    ready_listener = orch._on_ready_listener
    if ready_listener is None:
        raise AssertionError("expected on_ready_listener")
    await ready_listener()
    assert len(calls) == 1
    assert calls[0].guild is None

    # Subsequent ready should not sync again
    await ready_listener()
    assert len(calls) == 1
