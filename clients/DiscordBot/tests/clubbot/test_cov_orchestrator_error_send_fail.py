from __future__ import annotations

import asyncio
import logging

import pytest
from platform_discord.embed_helpers import EmbedProto
from platform_discord.protocols import (
    FileProto,
    FollowupProto,
    InteractionProto,
    MessageProto,
    ResponseProto,
    UserProto,
)
from tests.support.discord_fakes import RaisingFollowup
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.container import ServiceContainer
from clubbot.orchestrator import BotOrchestrator
from clubbot.services.qr.client import QRService

logger = logging.getLogger(__name__)


class _HTTPError(Exception):
    pass


class _Msg(MessageProto):
    @property
    def id(self) -> int:
        return 1

    async def edit(
        self, *, content: str | None = None, embed: EmbedProto | None = None
    ) -> MessageProto:
        _ = (content, embed)
        return self


class _Resp(ResponseProto):
    def __init__(self) -> None:
        self._done = True

    def is_done(self) -> bool:
        return True

    async def send_message(
        self,
        content: str | None = None,
        *,
        embed: EmbedProto | None = None,
        ephemeral: bool = False,
    ) -> None:
        raise AssertionError("should not be called")

    async def defer(self, *, ephemeral: bool = False) -> None:
        self._done = True


class _User(UserProto):
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
        _ = (content, embed, file)
        return _Msg()


class _Interaction(InteractionProto):
    def __init__(self) -> None:
        self._response = _Resp()
        self._followup: FollowupProto = RaisingFollowup(_HTTPError)
        self._user: UserProto = _User()

    @property
    def response(self) -> ResponseProto:
        return self._response

    @property
    def followup(self) -> FollowupProto:
        return self._followup

    @property
    def user(self) -> UserProto:
        return self._user


def test_on_app_command_error_followup_send_failure() -> None:
    cfg = build_settings(
        commands_sync_global=False,
        qr_default_border=2,
        qr_public_responses=True,
        qr_api_url="http://localhost:8080",
    )
    container = ServiceContainer(cfg=cfg, qr_service=QRService(cfg))
    orch = BotOrchestrator(container)
    orch.build_bot()
    orch.register_listeners()

    # Use the hook directly - no need to build orchestrator
    handler = _test_hooks.app_command_error_handler

    # Use the discord_exception_types hook to inject our custom exception types
    def _custom_exception_types() -> tuple[type[Exception], type[Exception], type[Exception]]:
        return (_HTTPError, _HTTPError, _HTTPError)

    original = _test_hooks.discord_exception_types
    _test_hooks.discord_exception_types = _custom_exception_types
    try:
        interaction: InteractionProto = _Interaction()
        with pytest.raises(_HTTPError):
            asyncio.run(handler(interaction, RuntimeError("x")))
    finally:
        _test_hooks.discord_exception_types = original
