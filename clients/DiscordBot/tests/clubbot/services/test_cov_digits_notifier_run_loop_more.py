"""Test digits_notifier run loop event handling."""

from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import TrackingBot, TrackingUser

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber

_CONFIG_EVENT = (
    '{"type":"digits.metrics.config.v1","user_id":1,"job_id":"r",'
    '"model_id":"m","total_epochs":2,"queue":"digits"}'
)


class _EventSource(MessageSource):
    """Source that returns a valid event."""

    def __init__(self) -> None:
        self.n = 0
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        _ = channel

    async def get(self) -> str | None:
        self.n += 1
        if self.n == 1:
            return _CONFIG_EVENT
        await asyncio.sleep(0.01)
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_digits_notifier_run_loop_handles_event() -> None:
    """Test that run loop processes events and triggers DM notifications."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = DigitsEventSubscriber(
        bot,
        redis_url="redis://",
        events_channel="ch",
        source_factory=lambda _: _EventSource(),
    )
    sub.start()
    await asyncio.sleep(0.05)
    await sub.stop()
    # Should have sent an embed for the config event
    assert user.embeds
