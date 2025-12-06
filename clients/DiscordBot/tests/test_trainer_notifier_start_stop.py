from __future__ import annotations

import asyncio
import logging

import pytest
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


@pytest.mark.asyncio
async def test_trainer_notifier_start_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")

    async def _noop_run(self: TrainerEventSubscriber) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TrainerEventSubscriber, "_run", _noop_run)

    sub.start()
    # idempotent start
    sub.start()
    await sub.stop()


@pytest.mark.asyncio
async def test_trainer_notifier_stop_without_start() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    # Should return early when no task is running
    await sub.stop()


logger = logging.getLogger(__name__)
