from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import NoReturn, Protocol

import discord
import pytest
from discord.ext import commands
from platform_discord.discord_types import Embed
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import FileProto, InteractionProto, MessageProto, UserProto
from tests.support.discord_fakes import FakeMessage
from tests.support.settings import build_settings

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
        self, content: str | None = None, *, embed: Embed | None = None, ephemeral: bool = False
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


class ErrorHandler(Protocol):
    __name__: str

    def __call__(
        self, interaction: InteractionProto, error: BaseException, /
    ) -> Coroutine[None, None, None]: ...


class OnReadyHandler(Protocol):
    __name__: str

    def __call__(self) -> Coroutine[None, None, None]: ...


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


class _Interaction:
    def __init__(self, done: bool) -> None:
        self.response = _FakeResp(done)
        self.followup = _FakeFollowup()
        self.user: UserProto = _FakeUser()


def _base_cfg(command_sync_global: bool = False) -> Config:
    return build_settings(
        discord_token="t.t.t",
        commands_sync_global=command_sync_global,
        guild_ids=[],
    )


def _build_orchestrator(command_sync_global: bool = False) -> BotOrchestrator:
    cfg = _base_cfg(command_sync_global)
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    orch.build_bot()
    return orch


def test_on_app_command_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _build_orchestrator()
    bot = orch.bot
    if bot is None:
        raise AssertionError("expected bot")

    captured: dict[str, ErrorHandler] = {}

    def capture_add_listener(func: ErrorHandler, name: str | None = None) -> None:
        key = name if name is not None else func.__name__
        captured[key] = func

    monkeypatch.setattr(bot, "add_listener", capture_add_listener, raising=True)
    orch.register_listeners()
    handler = captured["on_application_command_error"]

    # When response.is_done is True -> followup.send
    inter1 = _Interaction(done=True)
    asyncio.get_event_loop().run_until_complete(handler(inter1, RuntimeError("x")))
    assert inter1.followup.sent and not inter1.response.sent

    # When response.is_done is False -> response.send_message
    inter2 = _Interaction(done=False)
    asyncio.get_event_loop().run_until_complete(handler(inter2, RuntimeError("y")))
    assert inter2.response.sent


@pytest.mark.asyncio
async def test_setup_hook_invokes_wiring_and_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    calls: dict[str, bool] = {"wired": False, "registered": False}
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    # Patch wire_bot_async to track wiring call
    original_wire = cont.wire_bot_async

    async def tracking_wire(bot: commands.Bot) -> None:
        calls["wired"] = True
        await original_wire(bot)

    object.__setattr__(cont, "wire_bot_async", tracking_wire)
    orch = BotOrchestrator(cont)

    def reg() -> None:
        calls["registered"] = True

    monkeypatch.setattr(orch, "register_listeners", reg, raising=True)
    bot = orch.build_bot()
    await bot.setup_hook()
    assert calls["wired"] and calls["registered"]


class _ForbiddenError(Exception):
    """Test exception for forbidden."""


class _HTTPError(Exception):
    """Test exception for HTTP errors."""


def test_sync_commands_exception_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _build_orchestrator()

    async def raise_forbidden() -> bool:
        raise _ForbiddenError()

    async def raise_http() -> bool:
        raise _HTTPError()

    monkeypatch.setattr(orch, "_sync_global", raise_forbidden, raising=True)
    asyncio.get_event_loop().run_until_complete(orch.sync_commands())
    monkeypatch.setattr(orch, "_sync_global", raise_http, raising=True)
    with pytest.raises(_HTTPError):
        asyncio.get_event_loop().run_until_complete(orch.sync_commands())


def test_sync_commands_unknown_exception_to_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _build_orchestrator()

    class _WeirdError(Exception):
        pass

    async def raise_weird() -> bool:
        raise _WeirdError()

    monkeypatch.setattr(orch, "_sync_global", raise_weird, raising=True)
    with pytest.raises(RuntimeError):
        asyncio.get_event_loop().run_until_complete(orch.sync_commands())


@pytest.mark.asyncio
async def test_on_ready_sync_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _build_orchestrator(command_sync_global=True)
    bot = orch.bot
    if bot is None:
        raise AssertionError("expected bot")
    captured: dict[str, OnReadyHandler] = {}

    def capture(func: OnReadyHandler, name: str | None = None) -> None:
        key = name if name is not None else func.__name__
        captured[key] = func

    monkeypatch.setattr(bot, "add_listener", capture, raising=True)
    orch.register_listeners()
    on_ready = captured["on_ready"]

    calls: dict[str, int] = {"n": 0}

    async def fake_tree_sync(
        *, guild: discord.Guild | None = None
    ) -> list[discord.app_commands.AppCommand]:
        calls["n"] += 1
        return []

    object.__setattr__(bot.tree, "sync", fake_tree_sync)
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
        self, content: str | None = None, *, embed: Embed | None = None, ephemeral: bool = False
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


def test_on_app_command_error_send_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = _build_orchestrator()
    bot = orch.bot
    if bot is None:
        raise AssertionError("expected bot")
    captured: dict[str, ErrorHandler] = {}

    def capture(func: ErrorHandler, name: str | None = None) -> None:
        key = name if name is not None else func.__name__
        captured[key] = func

    monkeypatch.setattr(bot, "add_listener", capture, raising=True)
    orch.register_listeners()
    handler = captured["on_application_command_error"]

    # Patch orchestrator's local exception classes used in except clause
    import clubbot.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod, "_DHTTPExceptionError", _MyEError, raising=True)
    monkeypatch.setattr(orch_mod, "_DForbiddenError", _MyEError, raising=True)
    monkeypatch.setattr(orch_mod, "_DNotFoundError", _MyEError, raising=True)

    inter = _InterRaises()
    with pytest.raises(_MyEError):
        asyncio.get_event_loop().run_until_complete(handler(inter, RuntimeError("x")))


def test_preflight_app_id_mismatch_debug_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_settings(discord_token="MTIz.x.y")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "999")
    # Mismatch path is non-fatal; logs debug and returns
    orch._preflight_token_check()


class _FakeBotForRun:
    """Fake bot that tracks run() calls."""

    def __init__(self) -> None:
        self.run_token: str | None = None

    def run(self, token: str) -> None:
        self.run_token = token


def test_run_calls_build_and_bot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = build_settings(discord_token="tkn")
    cont = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(cont)
    fake_bot = _FakeBotForRun()

    def build_fake() -> _FakeBotForRun:
        return fake_bot

    monkeypatch.setattr(orch, "build_bot", build_fake, raising=True)
    orch.run()
    assert fake_bot.run_token == "tkn"


logger = logging.getLogger(__name__)
