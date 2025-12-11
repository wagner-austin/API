from __future__ import annotations

import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import NoneAppIdBot

from clubbot.cogs.invite import _resolve_app_id


def test_resolve_app_id_raises_when_no_application_id() -> None:
    """Test that _resolve_app_id raises RuntimeError when bot has no application_id."""
    bot: BotProto = NoneAppIdBot()
    with pytest.raises(RuntimeError):
        _resolve_app_id(bot)
