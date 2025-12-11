from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot, FakeEmbed, TrackingBot, TrackingUser

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


class _CloseOnlySource(MessageSource):
    def __init__(self) -> None:
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        _ = channel

    async def get(self) -> str | None:
        return None

    async def close(self) -> None:
        self.closed = True


class _ImmediateCloseSource(MessageSource):
    """A source that returns None immediately, causing run to finish."""

    def __init__(self) -> None:
        self.closed = False
        self.subscribed = False

    async def subscribe(self, channel: str) -> None:
        _ = channel
        self.subscribed = True

    async def get(self) -> str | None:
        # Return None to signal end of messages
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_trainer_stop_without_task_closes_source() -> None:
    src = _CloseOnlySource()

    def _factory(url: str) -> MessageSource:
        _ = url
        return src

    sub = TrainerEventSubscriber(
        bot=FakeBot(), redis_url="redis://example", source_factory=_factory
    )
    # Start to initialize source, then stop immediately
    sub.start()
    await asyncio.sleep(0)  # Let task start
    await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_trainer_stop_closes_source_normally() -> None:
    src = _ImmediateCloseSource()

    def _factory(url: str) -> MessageSource:
        _ = url
        return src

    sub = TrainerEventSubscriber(
        bot=FakeBot(), redis_url="redis://example", source_factory=_factory
    )
    sub.start()
    await asyncio.sleep(0.01)  # Let task run briefly
    await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_trainer_on_done_logs_for_exception() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")

    async def boom() -> None:
        raise RuntimeError("x")

    t = asyncio.create_task(boom())
    await asyncio.sleep(0)
    # _on_done is on the runner now
    sub._runner._on_done(t)


@pytest.mark.asyncio
async def test_trainer_notify_sends_dm() -> None:
    """Test that notify sends DM and caches message."""
    user = TrackingUser()
    bot = TrackingBot(user)
    sub = TrainerEventSubscriber(bot=bot, redis_url="redis://example")
    await sub.notify(1, "r", FakeEmbed(title="t"))
    assert len(user.embeds) == 1
    # Message should be cached
    if sub.get_cached_message("r") is None:
        raise AssertionError("expected cached message")


@pytest.mark.asyncio
async def test_trainer_run_delegates_to_runner() -> None:
    called = {"n": 0}

    class _CountingSource(MessageSource):
        async def subscribe(self, channel: str) -> None:
            _ = channel

        async def get(self) -> str | None:
            called["n"] += 1
            return None  # Return None to finish immediately

        async def close(self) -> None:
            pass

    def _factory(url: str) -> MessageSource:
        _ = url
        return _CountingSource()

    sub = TrainerEventSubscriber(
        bot=FakeBot(), redis_url="redis://example", source_factory=_factory
    )
    await sub._run()
    assert called["n"] >= 1


@pytest.mark.asyncio
async def test_trainer_stop_finished_no_exception_closes() -> None:
    src = _ImmediateCloseSource()

    def _factory(url: str) -> MessageSource:
        _ = url
        return src

    sub = TrainerEventSubscriber(
        bot=FakeBot(), redis_url="redis://example", source_factory=_factory
    )
    sub.start()
    await asyncio.sleep(0.01)  # Let task run briefly
    await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_trainer_on_done_no_exception_noop() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    done: asyncio.Task[None] = asyncio.create_task(asyncio.sleep(0))
    await done
    # _on_done is on the runner now
    sub._runner._on_done(done)


@pytest.mark.asyncio
async def test_trainer_stop_with_no_source() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    # Never started, so no source
    await sub.stop()
    # Should complete without error
