from __future__ import annotations

import asyncio
import logging

import pytest
from tests.support.discord_fakes import FakeBot

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_trainer_notifier_start_then_stop_idempotent() -> None:
    from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber

    bot = FakeBot()
    sub = TrainerEventSubscriber(bot=bot, redis_url="redis://", events_channel="ch")
    sub.start()
    await asyncio.sleep(0)
    # stop is safe to call and idempotent
    await sub.stop()
    await sub.stop()
