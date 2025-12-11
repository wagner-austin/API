from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from platform_core.job_events import JobEventV1
from platform_discord.discord_types import Embed as EmbedType
from platform_discord.embed_helpers import EmbedData, EmbedFieldData
from platform_discord.subscriber import MessageSource

from clubbot.services.jobs.turkic_notifier import TurkicEventSubscriber
from tests.support.discord_fakes import FakeBot, FakeMessageSource


class _FakeSource(MessageSource):
    def __init__(self) -> None:
        self.closed = False
        self.subscribed: list[str] = []

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    async def get(self) -> str | None:
        await asyncio.sleep(0)
        return None

    async def close(self) -> None:
        self.closed = True


class _StubSubscriber:
    def __init__(
        self,
        *,
        channel: str,
        source: MessageSource,
        decode: Callable[[str], JobEventV1 | None],
        handle: Callable[[JobEventV1], Awaitable[None]],
    ) -> None:
        _ = (channel, source, decode, handle)
        self.ran = False

    async def run(self, *, limit: int | None = None) -> None:
        _ = limit
        await asyncio.sleep(0)
        self.ran = True


class _NeverMatch:
    pass


class _FakeEmbed:
    @property
    def title(self) -> str | None:
        return None

    @property
    def description(self) -> str | None:
        return None

    @property
    def color_value(self) -> int | None:
        return None

    @property
    def footer_text(self) -> str | None:
        return None

    @property
    def field_count(self) -> int:
        return 0

    def add_field(self, *, name: str, value: str, inline: bool = True) -> None:
        _ = (name, value, inline)

    def set_footer(self, *, text: str) -> None:
        _ = text

    def get_field(self, name: str) -> EmbedFieldData | None:
        _ = name
        return None

    def has_field(self, name: str) -> bool:
        _ = name
        return False

    def get_all_fields(self) -> list[EmbedFieldData]:
        return []

    def get_field_value(self, name: str) -> str | None:
        _ = name
        return None

    def to_dict(self) -> EmbedData:
        return {}


@pytest.mark.asyncio
async def test_turkic_notifier_run_and_start_stop() -> None:
    def _src_factory(url: str) -> _FakeSource:
        _ = url
        return _FakeSource()

    sub = TurkicEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://x",
        source_factory=_src_factory,
    )
    sub.start()
    # early return branch
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()


@pytest.mark.asyncio
async def test_turkic_notifier_runner_on_done_handles_exception() -> None:
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")

    async def _boom() -> None:
        raise RuntimeError("x")

    t = asyncio.create_task(_boom())
    await asyncio.sleep(0)  # let it finish with exception
    # TaskRunner._on_done is on the internal _runner attribute.
    sub._runner._on_done(t)


@pytest.mark.asyncio
async def test_turkic_notifier_stop_without_task_closes_source() -> None:
    """Test that stop closes source when task is not running."""
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
    # Start to create source, then stop immediately
    sub.start()
    await asyncio.sleep(0)
    await sub.stop()

    # Source should be closed
    assert len(captured) == 1
    assert captured[0].closed is True


@pytest.mark.asyncio
async def test_turkic_notifier_maybe_notify_noop_when_no_embed() -> None:
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")
    await sub._maybe_notify({"job_id": "j", "user_id": 1, "embed": None})


@pytest.mark.asyncio
async def test_turkic_notifier_run_with_existing_subscriber() -> None:
    """Test that _run() invokes run_once on the runner."""
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
    # Call _run() which should call run_once internally
    await sub._run()

    # Source should have been created during run_once
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_turkic_notifier_notify_basic() -> None:
    # Test basic notify functionality with a valid user
    sub = TurkicEventSubscriber(bot=FakeBot(), redis_url="redis://x")
    embed: EmbedType = _FakeEmbed()
    # Should complete without error
    await sub.notify(1, "j", embed)
