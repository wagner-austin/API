from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.cogs.digits import DigitsCog


class _MockSubscriber:
    """A fake subscriber for testing."""

    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    async def stop(self) -> None:
        self.stop_count += 1


@pytest.mark.asyncio
async def test_digits_cog_ensure_start_and_unload_calls() -> None:
    cfg = build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        digits_public_responses=False,
        digits_rate_limit=1,
        digits_rate_window_seconds=1,
        digits_max_image_mb=1,
        redis_url="redis://example",
    )

    fake_sub = _MockSubscriber()

    def _factory(
        *, bot: _test_hooks.BotProto, redis_url: str
    ) -> _test_hooks.DigitsEventSubscriberLike:
        _ = (bot, redis_url)
        return fake_sub

    original = _test_hooks.digits_event_subscriber_factory
    _test_hooks.digits_event_subscriber_factory = _factory
    try:
        cog = DigitsCog(
            bot=FakeBot(),
            config=cfg,
            service=FakeDigitService(),
            autostart_subscriber=False,
        )

        cog.ensure_subscriber_started()
        await cog.cog_unload()

        assert fake_sub.start_count == 1 and fake_sub.stop_count == 1
    finally:
        _test_hooks.digits_event_subscriber_factory = original


logger = logging.getLogger(__name__)
