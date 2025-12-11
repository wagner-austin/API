from __future__ import annotations

import pytest
from platform_discord.protocols import MessageProto
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot, FakeEmbed, FakeMessage, FakeMessageSource

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_run_calls_runner_once() -> None:
    captured: list[FakeMessageSource] = []

    def _factory(url: str) -> MessageSource:
        _ = url
        src = FakeMessageSource()
        captured.append(src)
        return src

    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example", source_factory=_factory)
    await sub._run()
    # run_once should have been called which creates source
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_digits_notifier_notify_edits_existing_message() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    msgs: dict[str, MessageProto] = {"r": FakeMessage()}
    sub._messages = msgs
    await sub.notify(1, "r", FakeEmbed(title="t"))
