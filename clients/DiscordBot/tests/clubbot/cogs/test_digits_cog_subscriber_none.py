from __future__ import annotations

import logging

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_digits_cog_subscriber_none_paths() -> None:
    cfg = build_settings(
        qr_default_border=1,
        qr_public_responses=False,
        digits_public_responses=False,
        digits_rate_limit=1,
        digits_rate_window_seconds=1,
        digits_max_image_mb=1,
        redis_url=None,
    )

    cog = DigitsCog(bot=FakeBot(), config=cfg, service=FakeDigitService())
    cog.ensure_subscriber_started()
    await cog.cog_unload()
