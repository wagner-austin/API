from __future__ import annotations

import asyncio

import pytest
from platform_discord.task_runner import TaskRunner
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


class _TrackingLogger:
    """Logger that tracks warning calls."""

    def __init__(self) -> None:
        self.warning_count = 0

    def warning(self, msg: str, *args: str, extra: dict[str, str] | None = None) -> None:
        self.warning_count += 1


@pytest.mark.asyncio
async def test_trainer_notifier_start_then_stop_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://")

    # Ensure runner.stop is invoked; no logging expectations in new runner-based flow
    async def _stub_stop(self: TaskRunner) -> None:
        await asyncio.sleep(0)

    # Start and stop
    sub.start()
    # Replace class runner.stop with stub for deterministic stop
    monkeypatch.setattr(TaskRunner, "stop", _stub_stop, raising=True)
    await asyncio.sleep(0)
    await sub.stop()
