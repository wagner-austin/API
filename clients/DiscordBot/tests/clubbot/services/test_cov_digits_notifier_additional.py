from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from platform_discord.task_runner import TaskRunner
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


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
async def test_stop_without_task_or_subscriber_closes_source() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src
    await sub.stop()
    assert src.closed is True
    assert sub._source is None


@pytest.mark.asyncio
async def test_stop_cancelled_task_closes_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _stub(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _stub, raising=True)
    await sub.stop()
    assert src.closed is True
    assert sub._source is None


@pytest.mark.asyncio
async def test_stop_raises_on_exception_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _boom(self: TaskRunner) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(TaskRunner, "stop", _boom, raising=True)
    with pytest.raises(RuntimeError):
        await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_on_done_logs_for_exception() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")

    async def boom() -> None:
        raise RuntimeError("x")

    t = asyncio.create_task(boom())
    await asyncio.sleep(0)
    # Should not raise; error is logged via runner's _on_done
    sub._runner._on_done(t)


@pytest.mark.asyncio
async def test_digits_stop_finished_no_exception_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    src = _CloseOnlySource()
    sub._source = src

    async def _done(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _done, raising=True)
    await sub.stop()
    assert src.closed is True


@pytest.mark.asyncio
async def test_digits_stop_cancelled_with_no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    sub._source = None

    async def _stub(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _stub, raising=True)
    await sub.stop()


@pytest.mark.asyncio
async def test_digits_stop_finished_with_no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    sub._source = None

    async def _done(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(TaskRunner, "stop", _done, raising=True)
    await sub.stop()
