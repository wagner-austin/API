from __future__ import annotations

import pytest
from platform_core.job_events import (
    JobCompletedV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
)

from clubbot.services.jobs.turkic_notifier import TurkicEventSubscriber
from tests.support.discord_fakes import TrackingBot, TrackingUser


@pytest.mark.asyncio
async def test_turkic_notifier_handles_events_from_user_id() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = TurkicEventSubscriber(
        bot=bot,
        redis_url="redis://x",
    )

    await sub._handle_event(
        JobStartedV1(
            type="turkic.job.started.v1",
            domain="turkic",
            job_id="j",
            user_id=42,
            queue="turkic",
        )
    )
    await sub._handle_event(
        JobProgressV1(
            type="turkic.job.progress.v1",
            domain="turkic",
            job_id="j",
            user_id=42,
            progress=10,
        )
    )
    await sub._handle_event(
        JobCompletedV1(
            type="turkic.job.completed.v1",
            domain="turkic",
            job_id="j",
            user_id=42,
            result_id="fid",
            result_bytes=128,
        )
    )
    assert any(e is not None for e in user.embeds)


@pytest.mark.asyncio
async def test_turkic_notifier_handles_failed_event() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = TurkicEventSubscriber(bot=bot, redis_url="redis://x")
    await sub._handle_event(
        JobFailedV1(
            type="turkic.job.failed.v1",
            domain="turkic",
            job_id="j2",
            user_id=42,
            error_kind="system",
            message="x",
        )
    )
    # Embed should be sent for failed event
    assert any(e is not None for e in user.embeds)
