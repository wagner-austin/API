"""Tests for digits_notifier error handling paths."""

from __future__ import annotations

import pytest
from platform_discord.protocols import UserProto
from tests.support.discord_fakes import FakeEmbed, FakeUser

from clubbot.services.jobs.digits_notifier import DigitsEventSubscriber


class _RaisingBot:
    """Bot that raises on fetch_user."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        raise RuntimeError("fetch_user failed")


class _WorkingBot:
    """Bot with working fetch_user."""

    async def fetch_user(self, user_id: int, /) -> UserProto:
        return FakeUser(user_id=user_id)


@pytest.mark.asyncio
async def test_notify_propagates_fetch_user_error() -> None:
    """Test that errors from fetch_user propagate correctly."""
    sub = DigitsEventSubscriber(_RaisingBot(), redis_url="redis://fake")
    with pytest.raises(RuntimeError, match="fetch_user failed"):
        await sub.notify(1, "r", FakeEmbed(title="t"))


@pytest.mark.asyncio
async def test_notify_sends_dm_successfully() -> None:
    """Test that notify sends DM when bot works correctly."""
    sub = DigitsEventSubscriber(_WorkingBot(), redis_url="redis://fake")
    # Should not raise
    await sub.notify(1, "r", FakeEmbed(title="t"))
    # Message should be cached
    cached = sub.get_cached_message("r")
    if cached is None:
        raise AssertionError("expected cached message")
