from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.cogs.digits import DigitsCog

_Bot = FakeBot


class _FakeSub:
    """Fake subscriber that tracks calls."""

    def __init__(self, *, redis_url: str) -> None:
        self.redis_url = redis_url
        self._started = False
        self._stopped = False

    def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._stopped = True


@pytest.mark.asyncio
async def test_digits_cog_initializes_event_subscriber() -> None:
    captured: list[_FakeSub] = []

    def _factory(
        *,
        bot: _test_hooks.BotProto,
        redis_url: str,
    ) -> _FakeSub:
        _ = bot
        sub = _FakeSub(redis_url=redis_url)
        captured.append(sub)
        return sub

    _test_hooks.digits_event_subscriber_factory = _factory

    bot = _Bot()
    cfg = build_settings(
        qr_default_border=2,
        redis_url="redis://fake",
        digits_public_responses=False,
        digits_max_image_mb=2,
    )
    svc = FakeDigitService()
    _ = DigitsCog(bot, cfg, svc)

    assert len(captured) == 1
    assert captured[0].redis_url == "redis://fake"
    assert captured[0]._started is True


logger = logging.getLogger(__name__)
