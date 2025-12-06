from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog


@pytest.mark.asyncio
async def test_digits_cog_ensure_start_and_unload_calls() -> None:
    cfg = build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        digits_public_responses=False,
        digits_rate_limit=1,
        digits_rate_window_seconds=1,
        digits_max_image_mb=1,
    )

    cog = DigitsCog(
        bot=FakeBot(),
        config=cfg,
        service=FakeDigitService(),
        autostart_subscriber=False,
    )

    calls: dict[str, int] = {"start": 0, "stop": 0}

    class _Sub:
        def start(self) -> None:
            calls["start"] += 1

        async def stop(self) -> None:
            calls["stop"] += 1

    object.__setattr__(cog, "_subscriber", _Sub())

    cog.ensure_subscriber_started()
    await cog.cog_unload()

    assert calls["start"] == 1 and calls["stop"] == 1


logger = logging.getLogger(__name__)
