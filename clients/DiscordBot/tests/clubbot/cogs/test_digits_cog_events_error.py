from __future__ import annotations

import logging
import sys
import types

import pytest
from tests.support.discord_fakes import FakeBot, FakeDigitService
from tests.support.settings import build_settings

from clubbot.cogs.digits import DigitsCog
from clubbot.config import DiscordbotSettings


def _cfg() -> DiscordbotSettings:
    return build_settings(
        qr_default_border=2,
        redis_url="redis://fake",
        digits_public_responses=False,
        digits_max_image_mb=2,
    )


@pytest.mark.asyncio
async def test_digits_cog_events_subscriber_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_mod = types.ModuleType("clubbot.services.jobs.digits_notifier")
    modules: dict[str, types.ModuleType] = sys.modules
    monkeypatch.setitem(modules, "clubbot.services.jobs.digits_notifier", empty_mod)
    bot = FakeBot()
    cfg = _cfg()
    svc = FakeDigitService()
    _ = DigitsCog(bot, cfg, svc)


logger = logging.getLogger(__name__)
