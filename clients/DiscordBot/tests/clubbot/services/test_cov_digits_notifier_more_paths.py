"""Additional coverage paths for digits_notifier."""

from __future__ import annotations

import asyncio

import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1, DigitsConfigV1
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import TrackingBot, TrackingUser

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


class _EmptySource(MessageSource):
    """Source that returns None messages."""

    async def subscribe(self, channel: str) -> None:
        pass

    async def get(self) -> str | None:
        await asyncio.sleep(0)
        return None

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_digits_notifier_on_completed_invokes_notify() -> None:
    """Test that handling a completed event sends a DM notification."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = DigitsEventSubscriber(bot, redis_url="redis://")
    # First send a config event so runtime has config
    config: DigitsConfigV1 = {
        "type": "digits.metrics.config.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "total_epochs": 2,
        "queue": "digits",
    }
    await sub._handle_event(config)
    # Now send completed event
    completed: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "user_id": 1,
        "job_id": "r",
        "model_id": "m",
        "val_acc": 0.9,
    }
    await sub._handle_event(completed)
    # Should have sent embeds for both events
    assert len(user.embeds) >= 2


@pytest.mark.asyncio
async def test_digits_notifier_run_loop_not_msg_branch() -> None:
    """Test run loop with empty messages via source_factory."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = DigitsEventSubscriber(
        bot,
        redis_url="redis://",
        events_channel="ch",
        source_factory=lambda _: _EmptySource(),
    )
    sub.start()
    await asyncio.sleep(0.06)
    await sub.stop()


@pytest.mark.asyncio
async def test_digits_notifier_start_then_stop_idempotent() -> None:
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = DigitsEventSubscriber(bot, redis_url="redis://")
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()
    await sub.stop()
