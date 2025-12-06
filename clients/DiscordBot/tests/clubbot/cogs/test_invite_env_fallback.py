from __future__ import annotations

import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import NoneAppIdBot

from clubbot.cogs.invite import _resolve_app_id


def test_resolve_app_id_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    bot: BotProto = NoneAppIdBot()
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "555")
    with pytest.raises(RuntimeError):
        _resolve_app_id(bot)
