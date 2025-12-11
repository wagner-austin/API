from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot, FakeMessageSource

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


@pytest.mark.asyncio
async def test_trainer_notifier_start_then_stop_is_safe() -> None:
    """Test that start followed by stop properly cleans up."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://", source_factory=_factory)

    # Start and stop
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()

    assert len(captured) == 1
    assert captured[0].closed is True
