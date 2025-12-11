from __future__ import annotations

import asyncio
import logging

import pytest
from platform_discord.subscriber import MessageSource

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber
from tests.support.discord_fakes import FakeBot, FakeMessageSource


def _source_factory(captured: list[FakeMessageSource]) -> MessageSource:
    """Factory that captures created sources for verification."""
    src = FakeMessageSource()
    captured.append(src)
    return src


@pytest.mark.asyncio
async def test_trainer_notifier_start_stop() -> None:
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        return _source_factory(captured)

    sub = TrainerEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://example",
        source_factory=_factory,
    )

    sub.start()
    # idempotent start
    sub.start()
    # Allow the task to run briefly
    await asyncio.sleep(0)
    await sub.stop()

    # Verify source was created and closed
    assert len(captured) == 1
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_trainer_notifier_stop_without_start() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    # Should return early when no task is running
    await sub.stop()


logger = logging.getLogger(__name__)
