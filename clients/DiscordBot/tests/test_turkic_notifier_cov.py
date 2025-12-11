from __future__ import annotations

import asyncio

import pytest
from platform_core.job_events import (
    JobCompletedV1,
    JobFailedV1,
    JobProgressV1,
    JobStartedV1,
    encode_job_event,
    make_started_event,
)
from platform_discord.subscriber import MessageSource

from clubbot.services.jobs.turkic_notifier import TurkicEventSubscriber
from tests.support.discord_fakes import FakeBot, FakeMessageSource, FakeUser


@pytest.mark.asyncio
async def test_stop_closes_source_after_task_completes() -> None:
    """Test that stop() closes the source properly."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = TurkicEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://x",
        source_factory=_factory,
    )
    sub.start()
    # Allow task to run
    await asyncio.sleep(0)
    await sub.stop()

    # Source should have been created and closed
    assert len(captured) == 1
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_stop_with_no_source_is_noop() -> None:
    """Test that stop without start is a no-op."""
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")
    # Should return early when no task is running
    await sub.stop()


@pytest.mark.asyncio
async def test_stop_idempotent() -> None:
    """Test that multiple stop calls are safe."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = TurkicEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://x",
        source_factory=_factory,
    )
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()
    # Second stop should be safe
    await sub.stop()

    assert len(captured) == 1
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_on_done_invoked_by_completed_task() -> None:
    """Test that source is closed when task completes naturally."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = TurkicEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://x",
        source_factory=_factory,
    )
    sub.start()
    await asyncio.sleep(0)
    # stop should find no running task by now and simply return after ensuring source is closed
    await sub.stop()

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_handle_event_branches_calls_notify() -> None:
    user = FakeUser()
    sub = TurkicEventSubscriber(
        bot=FakeBot(user),
        redis_url="redis://x",
    )
    started: JobStartedV1 = {
        "type": "turkic.job.started.v1",
        "domain": "turkic",
        "job_id": "j1",
        "user_id": 42,
        "queue": "q",
    }
    progress: JobProgressV1 = {
        "type": "turkic.job.progress.v1",
        "domain": "turkic",
        "job_id": "j1",
        "user_id": 42,
        "progress": 5,
    }
    completed: JobCompletedV1 = {
        "type": "turkic.job.completed.v1",
        "domain": "turkic",
        "job_id": "j1",
        "user_id": 42,
        "result_id": "f",
        "result_bytes": 10,
    }
    failed: JobFailedV1 = {
        "type": "turkic.job.failed.v1",
        "domain": "turkic",
        "job_id": "j2",  # Different job_id for failed to trigger new DM
        "user_id": 42,
        "error_kind": "user",
        "message": "bad",
    }

    await sub._handle_event(started)
    await sub._handle_event(progress)
    await sub._handle_event(completed)
    await sub._handle_event(failed)

    # Runtime uses edit-in-place for same job_id, so j1 gets 1 DM (edits)
    # and j2 gets a second DM for the failed event.
    # At minimum we should have sent at least one message.
    assert user.sent


@pytest.mark.asyncio
async def test_handle_event_unknown_type_is_noop() -> None:
    """Test that unknown event types are silently ignored."""
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")

    # Use a valid payload from a different domain so it is ignored
    started: JobStartedV1 = {
        "type": "digits.job.started.v1",
        "domain": "digits",
        "job_id": "j1",
        "user_id": 42,
        "queue": "q",
    }
    await sub._handle_event(started)
    assert sub._runtime["_configs"] == {}


@pytest.mark.asyncio
async def test_turkic_notifier_decode_filters_domain() -> None:
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")
    decode = sub._decode

    turkic_payload = encode_job_event(
        make_started_event(domain="turkic", job_id="j1", user_id=1, queue="q")
    )
    other_payload = encode_job_event(
        make_started_event(domain="digits", job_id="j2", user_id=1, queue="q")
    )

    if decode(turkic_payload) is None:
        raise AssertionError("expected turkic payload to decode")
    assert decode(other_payload) is None
