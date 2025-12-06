from __future__ import annotations

import logging

import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import (
    FakeBot,
    RecordedSend,
    RecordingInteraction,
)
from tests.support.settings import build_settings

from clubbot.cogs.invite import InviteCog, _resolve_app_id
from clubbot.config import DiscordbotSettings


def _cfg() -> DiscordbotSettings:
    return build_settings(qr_public_responses=True)


def _bot_with_app_id(app_id: int | None) -> BotProto:
    return FakeBot(application_id=app_id)


def _last_send(sent: list[RecordedSend]) -> RecordedSend:
    assert sent, "Expected at least one send"
    return sent[-1]


def test_resolve_app_id_requires_application_id() -> None:
    bot: BotProto = _bot_with_app_id(1234)
    assert _resolve_app_id(bot) == 1234


def test_resolve_app_id_raises_when_missing() -> None:
    bot: BotProto = _bot_with_app_id(None)
    with pytest.raises(RuntimeError):
        _resolve_app_id(bot)


@pytest.mark.asyncio
async def test_invite_sends_embed_via_response() -> None:
    bot: BotProto = _bot_with_app_id(13579)
    cfg = _cfg()
    cog = InviteCog(bot, cfg)
    inter = RecordingInteraction()

    await cog._invite_impl(inter)

    last = _last_send(inter.sent)
    assert last["where"] == "response"
    if last["embed"] is None:
        raise AssertionError("expected embed")
    assert last["ephemeral"] is True


@pytest.mark.asyncio
async def test_invite_sends_embed_via_followup_when_response_done() -> None:
    bot: BotProto = _bot_with_app_id(24680)
    cfg = _cfg()
    cog = InviteCog(bot, cfg)
    inter = RecordingInteraction(response_done=True)

    await cog._invite_impl(inter)

    last = _last_send(inter.sent)
    assert last["where"] == "followup"
    if last["embed"] is None:
        raise AssertionError("expected embed")
    assert last["ephemeral"] is True


@pytest.mark.asyncio
async def test_invite_reports_missing_app_id() -> None:
    bot: BotProto = _bot_with_app_id(None)
    cfg = _cfg()
    cog = InviteCog(bot, cfg)
    inter = RecordingInteraction()

    with pytest.raises(RuntimeError):
        await cog._invite_impl(inter)


logger = logging.getLogger(__name__)
