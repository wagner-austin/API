from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot, FakeMessageSource

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


@pytest.mark.asyncio
async def test_trainer_notifier_stop_after_start() -> None:
    """Test that stop() properly stops after start()."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = TrainerEventSubscriber(
        bot=FakeBot(), redis_url="redis://example", source_factory=_factory
    )
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()

    assert len(captured) == 1
    assert captured[0].closed is True
