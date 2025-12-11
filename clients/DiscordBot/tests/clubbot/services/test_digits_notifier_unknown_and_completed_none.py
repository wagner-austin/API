from __future__ import annotations

import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1
from platform_core.digits_metrics_events import DigitsEventV1 as DEvent
from tests.support.discord_fakes import FakeBot

from clubbot import _test_hooks
from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_handles_unknown_event_branch() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://fake")
    ev: DEvent = __import__("builtins").dict(type="digits.metrics.unknown.v1")
    await sub._handle_event(ev)


@pytest.mark.asyncio
async def test_digits_notifier_completed_returns_none_skips_notify() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://fake")

    def _none_action(
        runtime: _test_hooks.DigitsRuntime,
        *,
        user_id: int,
        request_id: str,
        model_id: str,
        run_id: str | None,
        val_acc: float,
    ) -> _test_hooks.RequestAction | None:
        _ = (runtime, user_id, request_id, model_id, run_id, val_acc)
        return None

    original = _test_hooks.on_completed
    _test_hooks.on_completed = _none_action
    try:
        ev: DigitsCompletedMetricsV1 = {
            "type": "digits.metrics.completed.v1",
            "job_id": "r",
            "user_id": 1,
            "model_id": "m",
            "val_acc": 0.5,
        }
        await sub._handle_completed_event(ev)
    finally:
        _test_hooks.on_completed = original
