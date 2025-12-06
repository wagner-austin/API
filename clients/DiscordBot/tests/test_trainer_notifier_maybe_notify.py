from __future__ import annotations

import pytest
from platform_discord.trainer.runtime import RequestAction
from tests.support.discord_fakes import FakeBot

from clubbot.services.jobs.trainer_notifier import TrainerEventSubscriber


@pytest.mark.asyncio
async def test_trainer_maybe_notify_skips_when_no_embed() -> None:
    sub = TrainerEventSubscriber(bot=FakeBot(), redis_url="redis://example")
    act: RequestAction = {"user_id": 1, "request_id": "r", "embed": None}
    await sub._maybe_notify(act)  # Should return early without raising
