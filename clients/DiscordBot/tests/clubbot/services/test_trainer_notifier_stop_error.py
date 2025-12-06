from __future__ import annotations

import pytest
from platform_discord.task_runner import TaskRunner
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


@pytest.mark.asyncio
async def test_trainer_notifier_stop_raises_on_task_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")

    async def boom(self: TaskRunner) -> None:
        raise RuntimeError("x")

    monkeypatch.setattr(TaskRunner, "stop", boom, raising=True)
    with pytest.raises(RuntimeError):
        await sub.stop()
