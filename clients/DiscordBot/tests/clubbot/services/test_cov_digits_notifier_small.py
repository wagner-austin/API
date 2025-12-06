from __future__ import annotations

import logging

import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1
from platform_discord.handwriting.runtime import RequestAction
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_digits_notifier_maybe_notify_and_on_completed_early() -> None:
    bot = FakeBot()
    sub = DigitsEventSubscriber(bot, redis_url="redis://", events_channel="ch")

    act: RequestAction = {"user_id": 1, "request_id": "r", "embed": None}
    await sub._maybe_notify(act)

    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "val_acc": 0.9,
    }
    await sub._handle_event(completed)
