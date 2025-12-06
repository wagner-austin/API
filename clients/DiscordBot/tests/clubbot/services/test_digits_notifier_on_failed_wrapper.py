from __future__ import annotations

import pytest
from platform_core.job_events import JobFailedV1
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_on_failed_wrapper_updates_message_cache() -> None:
    bot = FakeBot()
    sub = DigitsEventSubscriber(bot, redis_url="redis://fake")
    ev: JobFailedV1 = {
        "type": "digits.job.failed.v1",
        "job_id": "r",
        "user_id": 1,
        "error_kind": "system",
        "message": "boom",
        "domain": "digits",
    }
    await sub._on_failed(ev)  # Should DM and cache message without raising
    assert "r" in sub._messages
