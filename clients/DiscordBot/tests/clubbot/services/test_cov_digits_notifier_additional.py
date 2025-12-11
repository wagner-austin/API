from __future__ import annotations

import asyncio

import pytest
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot, FakeMessageSource

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_stop_closes_source_properly() -> None:
    """Test that stop() closes the source after starting."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example", source_factory=_factory)
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()

    assert len(captured) == 1
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_stop_without_start_is_safe() -> None:
    """Test that stopping without starting is safe."""
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    # Should not raise
    await sub.stop()


@pytest.mark.asyncio
async def test_stop_idempotent() -> None:
    """Test that multiple stops are safe."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example", source_factory=_factory)
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()
    # Second stop should be safe
    await sub.stop()

    assert len(captured) == 1
    assert captured[0].closed is True


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
async def test_start_idempotent() -> None:
    """Test that multiple starts only create one source."""
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example", source_factory=_factory)
    sub.start()
    sub.start()  # Second start should be no-op
    await asyncio.sleep(0)
    await sub.stop()

    assert len(captured) == 1
