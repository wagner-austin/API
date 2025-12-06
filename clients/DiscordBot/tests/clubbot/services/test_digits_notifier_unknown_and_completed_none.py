from __future__ import annotations

import pytest
from platform_core.digits_metrics_events import DigitsCompletedMetricsV1
from platform_core.digits_metrics_events import DigitsEventV1 as DEvent
from platform_discord.handwriting.runtime import DigitsRuntime, RequestAction
from tests.support.discord_fakes import FakeBot

import clubbot.services.jobs.digits_notifier as dn
from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


@pytest.mark.asyncio
async def test_digits_notifier_handles_unknown_event_branch() -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://fake")
    ev: DEvent = __import__("builtins").dict(type="digits.metrics.unknown.v1")
    await sub._handle_event(ev)


@pytest.mark.asyncio
async def test_digits_notifier_completed_returns_none_skips_notify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sub = DigitsEventSubscriber(FakeBot(), redis_url="redis://fake")

    def _none_action(
        runtime: DigitsRuntime,
        *,
        user_id: int,
        request_id: str,
        model_id: str,
        run_id: str | None,
        val_acc: float,
    ) -> RequestAction | None:
        _ = (runtime, user_id, request_id, model_id, run_id, val_acc)
        return None

    monkeypatch.setattr(dn, "on_completed", _none_action, raising=True)
    ev: DigitsCompletedMetricsV1 = {
        "type": "digits.metrics.completed.v1",
        "job_id": "r",
        "user_id": 1,
        "model_id": "m",
        "val_acc": 0.5,
    }
    await sub._handle_completed_event(ev)
