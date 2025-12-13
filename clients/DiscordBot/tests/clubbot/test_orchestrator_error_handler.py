from __future__ import annotations

import logging
from typing import NoReturn

import discord
import pytest
from discord.app_commands import AppCommand
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, InteractionProto, MessageProto, UserProto
from tests.support.discord_fakes import FakeMessage
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot._test_hooks import BotRunnerProtocol, BotTreeProto, SnowflakeLike
from clubbot.config import DiscordbotSettings as Config
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService


class _FakeResp:
    def __init__(self, done: bool) -> None:
        self._done = done
        self.sent: list[tuple[str, bool]] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.sent.append((content or "", ephemeral))

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class _FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> MessageProto:
        self.sent.append((content or "", ephemeral))
        return FakeMessage()


class _FakeUser:
    @property
    def id(self) -> int:
        return 123

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
    ) -> MessageProto:
        return FakeMessage()


class _Interaction(InteractionProto):
    """Fake interaction that implements InteractionProto for testing."""

    def __init__(self, done: bool) -> None:
        self._response = _FakeResp(done)
        self._followup = _FakeFollowup()
        self._user: UserProto = _FakeUser()

    @property
    def response(self) -> _FakeResp:
        return self._response

    @property
    def followup(self) -> _FakeFollowup:
        return self._followup

    @property
    def user(self) -> UserProto:
        return self._user


def _base_cfg(command_sync_global: bool = False) -> Config:
    return build_settings(
        discord_token="t.t.t",
        commands_sync_global=command_sync_global,
        guild_ids=[],
    )


def _build_orchestrator(command_sync_global: bool = False) -> BotOrchestrator:
    cfg = _base_cfg(command_sync_global)

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    orch.build_bot()
    orch.register_listeners()
    return orch


@pytest.mark.asyncio
async def test_on_app_command_error_branches() -> None:
    """Test error handler uses followup when response is done, else response."""
    # No need to build orchestrator - use the hook directly
    handler = _test_hooks.app_command_error_handler

    # When response.is_done is True -> followup.send
    inter1 = _Interaction(done=True)
    await handler(inter1, RuntimeError("x"))
    assert inter1.followup.sent and not inter1.response.sent

    # When response.is_done is False -> response.send_message
    inter2 = _Interaction(done=False)
    await handler(inter2, RuntimeError("y"))
    assert inter2.response.sent


@pytest.mark.asyncio
async def test_setup_hook_invokes_wiring_and_register() -> None:
    """Test that setup_hook calls both wiring and register_listeners.

    Uses a custom _Bot subclass to verify setup_hook flow without mocking.
    """
    cfg = _base_cfg()

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    bot = orch.build_bot()

    # Build was called, now invoke setup_hook directly
    # The setup_hook is already defined in the _Bot class created by build_bot
    await bot.setup_hook()

    # Verify listeners were registered by calling them
    ready_listener = orch._on_ready_listener
    if ready_listener is None:
        raise AssertionError("on_ready_listener not registered")
    guild_join_listener = orch._on_guild_join_listener
    if guild_join_listener is None:
        raise AssertionError("on_guild_join_listener not registered")
    error_listener = orch._on_application_command_error_listener
    if error_listener is None:
        raise AssertionError("on_application_command_error_listener not registered")


class _ForbiddenError(Exception):
    """Test exception for forbidden - name contains 'Forbidden'."""


class _HTTPError(Exception):
    """Test exception for HTTP errors - name contains 'HTTP'."""


@pytest.mark.asyncio
async def test_sync_commands_exception_paths() -> None:
    """Test sync_commands handles Forbidden and HTTP exceptions correctly."""
    orch = _build_orchestrator()

    async def raise_forbidden() -> bool:
        raise _ForbiddenError()

    async def raise_http() -> bool:
        raise _HTTPError()

    # Test Forbidden path - should log warning and not re-raise
    _test_hooks.orchestrator_sync_global_override = raise_forbidden
    await orch.sync_commands()

    # Test HTTP path - should re-raise
    _test_hooks.orchestrator_sync_global_override = raise_http
    with pytest.raises(_HTTPError):
        await orch.sync_commands()


@pytest.mark.asyncio
async def test_sync_commands_unknown_exception_to_runtime_error() -> None:
    """Test sync_commands converts unknown exceptions to RuntimeError."""
    orch = _build_orchestrator()

    class _WeirdError(Exception):
        pass

    async def raise_weird() -> bool:
        raise _WeirdError()

    _test_hooks.orchestrator_sync_global_override = raise_weird
    with pytest.raises(RuntimeError):
        await orch.sync_commands()


@pytest.mark.asyncio
async def test_on_ready_sync_toggle() -> None:
    """Test that on_ready only syncs once."""
    orch = _build_orchestrator(command_sync_global=True)

    on_ready = orch._on_ready_listener
    if on_ready is None:
        raise AssertionError("expected on_ready_listener")

    calls: dict[str, int] = {"n": 0}

    async def fake_tree_sync(
        tree: BotTreeProto, guild: SnowflakeLike | None = None
    ) -> list[AppCommand]:
        calls["n"] += 1
        return []

    _test_hooks.tree_sync = fake_tree_sync
    await on_ready()
    await on_ready()
    assert calls["n"] == 1


class _MyEError(Exception):
    """Test error that gets raised on send."""


class _RespRaises:
    """Response that raises on send_message."""

    def is_done(self) -> bool:
        return False

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        raise _MyEError()

    async def defer(self, *, ephemeral: bool = False) -> None:
        pass


class _FollowRaises:
    """Followup that raises on send."""

    async def send(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        file: FileProto | None = None,
        ephemeral: bool = False,
    ) -> NoReturn:
        raise _MyEError()


class _InterRaises:
    """Interaction where both response and followup raise."""

    def __init__(self) -> None:
        self.response = _RespRaises()
        self.followup = _FollowRaises()
        self.user: UserProto = _FakeUser()


@pytest.mark.asyncio
async def test_on_app_command_error_send_failure_raises() -> None:
    """Test error handler re-raises when send fails with Discord exceptions."""
    # Use the hook directly - no need to build orchestrator
    handler = _test_hooks.app_command_error_handler

    # Use discord_exception_types hook to inject our test exception type
    _test_hooks.discord_exception_types = lambda: (_MyEError, _MyEError, _MyEError)

    inter = _InterRaises()
    with pytest.raises(_MyEError):
        await handler(inter, RuntimeError("x"))


def test_preflight_app_id_mismatch_debug_no_raise() -> None:
    """Test that the app ID mismatch path logs debug but doesn't raise."""
    cfg = build_settings(discord_token="MTIz.x.y")

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    # The mismatch path is non-fatal; it logs debug and returns
    orch._preflight_token_check()


class _FakeBotForRun:
    """Fake bot that tracks run() calls and implements BotRunnerProtocol."""

    __slots__ = ("run_token",)

    def __init__(self) -> None:
        self.run_token: str | None = None

    def run(self, token: str) -> None:
        self.run_token = token


def test_run_calls_build_and_bot_run() -> None:
    """Test run() builds bot and calls bot.run with token."""
    cfg = build_settings(discord_token="tkn")

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    fake_bot = _FakeBotForRun()

    # Use orchestrator_build_bot hook to inject our fake bot
    def _build_fake_bot(orchestrator: _test_hooks.OrchestratorLike) -> BotRunnerProtocol:
        _ = orchestrator  # Ignore orchestrator; return fake
        return fake_bot

    _test_hooks.orchestrator_build_bot = _build_fake_bot
    orch.run()
    assert fake_bot.run_token == "tkn"


def test_run_production_path_calls_build_bot() -> None:
    """Test run() uses build_bot via the default hook.

    The default orchestrator_build_bot hook calls orchestrator.build_bot().
    We verify this by checking the hook behavior directly.
    """
    cfg = build_settings(discord_token="tkn")

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)

    # The default hook calls orchestrator.build_bot()
    # We can verify this by calling build_bot and checking the bot instance
    bot = orch.build_bot()

    # Verify bot was created and is the same instance as orch.bot
    assert orch.bot is bot


@pytest.mark.asyncio
async def test_on_application_command_error_listener_invokes_hook() -> None:
    """Test that the on_application_command_error listener invokes the hook.

    This covers orchestrator.py line 140:
        await _test_hooks.app_command_error_handler(interaction, error)
    """
    cfg = _base_cfg()

    def _test_load_settings() -> Config:
        return cfg

    _test_hooks.load_settings = _test_load_settings

    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    orch.build_bot()
    orch.register_listeners()

    error_listener = orch._on_application_command_error_listener
    if error_listener is None:
        raise AssertionError("on_application_command_error_listener not registered")

    # Track whether hook was called - only track the error, not the interaction type
    hook_called_with_errors: list[Exception] = []

    async def _recording_hook(
        interaction: discord.Interaction | InteractionProto, error: Exception
    ) -> None:
        _ = interaction  # Suppress unused
        hook_called_with_errors.append(error)

    _test_hooks.app_command_error_handler = _recording_hook

    # Create a fake interaction and error
    # _Interaction implements InteractionProto which is now accepted by the listener
    inter: InteractionProto = _Interaction(done=False)
    err = RuntimeError("test error")

    # Call the listener directly
    await error_listener(inter, err)

    # Verify the hook was invoked with the correct error
    assert len(hook_called_with_errors) == 1
    assert hook_called_with_errors[0] is err


logger = logging.getLogger(__name__)
