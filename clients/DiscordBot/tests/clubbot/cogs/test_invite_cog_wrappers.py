from __future__ import annotations

import pytest
from tests.support.discord_fakes import FakeBot, RecordingInteraction
from tests.support.settings import build_settings

from clubbot.cogs.invite import InviteCog


@pytest.mark.asyncio
async def test_invite_impl_raises_when_bot_missing() -> None:
    cfg = build_settings()
    cog = InviteCog(FakeBot(application_id=123456789), cfg)
    # Simulate missing bot by clearing attribute
    cog.bot = None
    with pytest.raises(RuntimeError):
        await cog._invite_impl(RecordingInteraction())
