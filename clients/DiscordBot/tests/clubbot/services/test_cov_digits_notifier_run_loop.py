"""Test digits_notifier run loop branches."""

from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import TrackingBot, TrackingUser

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


class _BranchSource(MessageSource):
    """Source that returns different message types to cover branches."""

    def __init__(self) -> None:
        self.n = 0
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        _ = channel

    async def get(self) -> str | None:
        self.n += 1
        if self.n == 1:
            return "not-json"  # Will fail decode
        if self.n == 2:
            return "{}"  # Will decode to None (unknown event)
        await asyncio.sleep(0.01)
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_digits_notifier_run_loop_covers_branches() -> None:
    """Test run loop handles decode failures and unknown events."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = DigitsEventSubscriber(
        bot,
        redis_url="redis://",
        events_channel="ch",
        source_factory=lambda _: _BranchSource(),
    )
    sub.start()
    await asyncio.sleep(0.05)
    await sub.stop()
    # No embeds should be sent for invalid/unknown events
    assert len(user.embeds) == 0
