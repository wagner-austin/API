"""Tests for the respawn notice the tick body emits once per death.

``_note_respawn`` runs on EVERY tick that has a self state -- it is the
second thing the tick body does after reading the world -- so its guard
is what keeps "Respawned at ..." to one line per actual respawn instead
of one line per tick.
"""

from __future__ import annotations

import logging

import pytest

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import SelfStateDict, make_self_state
from tests.conftest import FakeEnv

_RESPAWNED = "Respawned at"


def _self_at(x: int, y: int, fuel: int) -> SelfStateDict:
    """Return a self state for the respawn notice.

    Args:
        x: Tank X coordinate.
        y: Tank Y coordinate.
        fuel: Current fuel.

    Returns:
        A self-state record the notice can format.
    """
    return make_self_state(
        tank_id=1,
        x=x,
        y=y,
        team=2,
        rank=0,
        fuel=fuel,
        leaderboard_position=1,
    )


class TestRespawnNotice:
    """The respawn line is emitted once, when a respawn was actually pending."""

    def test_no_pending_respawn_logs_nothing(
        self,
        fake_env: FakeEnv,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An ordinary tick does not announce a respawn.

        The deadline is 0 whenever the tank has not died, which is the
        overwhelming majority of ticks. Without the guard every one of
        them logs a respawn at the tank's current position -- the line a
        session post-mortem uses to find the moment of death would
        appear thousands of times, once per tick, and never mark a
        death.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _note_respawn

        bot = Bot("https://test.tankpit.com/", headless=True, world=WorldService())
        assert bot._respawn_deadline_ms == 0

        with caplog.at_level(logging.INFO):
            _note_respawn(bot, _self_at(10, 10, 1000))

        assert not any(_RESPAWNED in record.message for record in caplog.records)

    def test_a_pending_respawn_is_announced_once_and_cleared(
        self,
        fake_env: FakeEnv,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Control: a real respawn logs, clears the wait, and does not repeat."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_body import _note_respawn

        bot = Bot("https://test.tankpit.com/", headless=True, world=WorldService())
        bot._respawn_deadline_ms = 999_000

        with caplog.at_level(logging.INFO):
            _note_respawn(bot, _self_at(10, 10, 1000))
            first = [r for r in caplog.records if _RESPAWNED in r.message]
            _note_respawn(bot, _self_at(10, 10, 1000))
            second = [r for r in caplog.records if _RESPAWNED in r.message]

        assert len(first) == 1
        assert len(second) == 1
        assert bot._respawn_deadline_ms == 0
