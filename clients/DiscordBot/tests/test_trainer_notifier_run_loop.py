from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest
from platform_core.job_events import default_events_channel
from platform_discord.subscriber import MessageSource
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


class _FakeSource(MessageSource):
    instances: ClassVar[list[_FakeSource]] = []

    def __init__(self, redis_url: str) -> None:
        self.url = redis_url
        self.subscribed: list[str] = []
        self.closed: bool = False
        _FakeSource.instances.append(self)

    async def subscribe(self, channel: str) -> None:
        # Record channel; no external IO
        self.subscribed.append(channel)

    async def get(self) -> str | None:
        await asyncio.sleep(0)
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_constructs_and_runs_subscriber() -> None:
    # Clear class variable for test isolation
    _FakeSource.instances.clear()

    # Use source_factory parameter to inject fake source
    sub = TrainerEventSubscriber(
        bot=FakeBot(),
        redis_url="redis://unit-test",
        source_factory=lambda url: _FakeSource(url),
    )

    # Execute the run path once; it should subscribe and then close without hanging
    await sub._run()

    # Validate our fake source saw the expected channel and was closed
    assert _FakeSource.instances, "expected FakeSource to be constructed"
    inst = _FakeSource.instances[-1]
    assert inst.url == "redis://unit-test"
    assert inst.subscribed == [default_events_channel("trainer")]
    assert inst.closed is True
