from __future__ import annotations

import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import FakeBot, StrAppIdBot

from clubbot.cogs.invite import _resolve_app_id


def _bot_with_int_app_id() -> BotProto:
    return FakeBot(application_id=1234567890)


def test_resolve_app_id_accepts_int_application_id() -> None:
    bot: BotProto = _bot_with_int_app_id()
    assert _resolve_app_id(bot) == 1234567890


def test_resolve_app_id_rejects_string_application_id() -> None:
    bot: BotProto = StrAppIdBot()
    with pytest.raises(RuntimeError):
        _resolve_app_id(bot)
