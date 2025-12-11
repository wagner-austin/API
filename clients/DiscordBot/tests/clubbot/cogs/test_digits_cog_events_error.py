from __future__ import annotations

import logging
from typing import NoReturn

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.cogs.digits import DigitsCog
from clubbot.config import DiscordbotSettings


def _cfg_no_redis() -> DiscordbotSettings:
    return build_settings(
        qr_default_border=2,
        redis_url=None,
        digits_public_responses=False,
        digits_max_image_mb=2,
    )


@pytest.mark.asyncio
async def test_digits_cog_no_redis_skips_subscriber() -> None:
    """Test that when redis_url is not set, subscriber is not created."""
    factory_called: list[bool] = []

    def _tracking_factory(
        *,
        bot: _test_hooks.BotProto,
        redis_url: str,
    ) -> NoReturn:
        _ = (bot, redis_url)
        factory_called.append(True)
        raise AssertionError("Should not be called when redis_url is None")

    _test_hooks.digits_event_subscriber_factory = _tracking_factory

    bot = FakeBot()
    cfg = _cfg_no_redis()
    svc = FakeDigitService()
    cog = DigitsCog(bot, cfg, svc)
    assert not factory_called, "Factory should not be called when redis_url is None"
    assert cog._subscriber is None


logger = logging.getLogger(__name__)
