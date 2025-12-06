from __future__ import annotations

import logging

import pytest
from platform_discord.protocols import BotProto
from tests.support.discord_fakes import FakeBot, FakeDigitService

import clubbot.cogs.digits as digits_mod
from clubbot.cogs.digits import DigitsCog

_Bot = FakeBot


@pytest.mark.asyncio
async def test_digits_cog_initializes_event_subscriber(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support.settings import build_settings

    created: dict[str, str | bool] = {}

    class _FakeSub:
        def __init__(self, bot: BotProto, *, redis_url: str, events_channel: str) -> None:
            _ = bot
            created["redis_url"] = redis_url
            created["events_channel"] = events_channel
            self._started = False

        def start(self) -> None:
            self._started = True
            created["started"] = True

        async def stop(self) -> None:
            created["stopped"] = True

    monkeypatch.setattr("clubbot.cogs.digits.DigitsEventSubscriber", _FakeSub, raising=True)

    bot = _Bot()
    cfg = build_settings(
        qr_default_border=2,
        redis_url="redis://fake",
        digits_public_responses=False,
        digits_max_image_mb=2,
    )
    svc = FakeDigitService()
    _ = DigitsCog(bot, cfg, svc)
    assert created.get("redis_url") == "redis://fake"
    assert created.get("events_channel") == digits_mod.DEFAULT_DIGITS_EVENTS_CHANNEL
    assert created.get("started") is True


logger = logging.getLogger(__name__)
