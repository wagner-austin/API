from __future__ import annotations

import asyncio
import logging

import pytest
from tests.support.discord_fakes import (
    FakeFollowupRaises,
    FakeInteraction,
    FakeResponse,
    FakeResponseRaises,
)


def test_base_cog_handle_user_error_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from clubbot.cogs.base import BaseCog

    bc = BaseCog()
    log = bc.request_logger("r1")

    # response not done -> uses response.send_message
    response1 = FakeResponse(done=False)
    inter1 = FakeInteraction(response=response1, followup=FakeFollowupRaises())
    asyncio.run(bc.handle_user_error(inter1, log, "m"))
    assert response1.sent

    # response done -> followup first, then fallback warning path when followup fails
    inter2 = FakeInteraction(response=FakeResponse(done=True), followup=FakeFollowupRaises())
    asyncio.run(bc.handle_user_error(inter2, log, "m2"))


def test_base_cog_handle_exception_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from clubbot.cogs.base import BaseCog

    bc = BaseCog()
    log = bc.request_logger("r2")

    # First: response not done
    response1 = FakeResponse(done=False)
    inter1 = FakeInteraction(response=response1, followup=FakeFollowupRaises())
    asyncio.run(bc.handle_exception(inter1, log, RuntimeError("x")))

    # Next: both followup and response raise -> warning path
    inter2 = FakeInteraction(response=FakeResponseRaises(done=False), followup=FakeFollowupRaises())
    asyncio.run(bc.handle_exception(inter2, log, RuntimeError("y")))


logger = logging.getLogger(__name__)
