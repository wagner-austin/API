from __future__ import annotations

import pytest
from platform_discord.protocols import MessageProto
from platform_discord.task_runner import TaskRunner
from tests.support.discord_fakes import FakeBot, FakeEmbed, FakeMessage

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_run_calls_runner_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    called = {"n": 0}

    async def _once(self: TaskRunner) -> None:
        called["n"] += 1

    monkeypatch.setattr(TaskRunner, "run_once", _once, raising=True)
    await sub._run()
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_digits_notifier_notify_edits_existing_message() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://example")
    msgs: dict[str, MessageProto] = {"r": FakeMessage()}
    sub._messages = msgs
    await sub.notify(1, "r", FakeEmbed(title="t"))
