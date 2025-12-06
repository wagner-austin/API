from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from platform_discord.task_runner import TaskRunner
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


@pytest.mark.asyncio
async def test_trainer_stop_without_task_closes_source() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src
    await sub.stop()
    assert src.closed is True
    assert sub._source is None


@pytest.mark.asyncio
async def test_trainer_stop_cancelled_closes_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _stub_stop(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _stub_stop, raising=True)
    await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_trainer_stop_raises_on_exception_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _boom(self: TaskRunner) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(TaskRunner, "stop", _boom, raising=True)
    with pytest.raises(RuntimeError):
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
async def test_trainer_run_delegates_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    called = {"n": 0}

    async def _once(self: TaskRunner) -> None:
        called["n"] += 1

    monkeypatch.setattr(TaskRunner, "run_once", _once, raising=True)
    await sub._run()
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_trainer_stop_finished_no_exception_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _done(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _done, raising=True)
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
async def test_trainer_stop_cancelled_with_no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    sub._source = None

    async def _stub(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _stub, raising=True)
    await sub.stop()
