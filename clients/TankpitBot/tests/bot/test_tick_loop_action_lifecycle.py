"""Tests for which in-flight actions are still live, and which get cleared.

Split from ``test_tick_loop_command_error.py`` (634 lines, over the
600-line ceiling): that file owns the 0x52 rejection paths, this one owns
the lifecycle questions asked before any error is consulted -- is this
action still in flight, and does a refused move belong to it.
"""

from __future__ import annotations

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, InFlightActionDict, make_in_flight_action
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tests.conftest import FakeEnv


def _pending(kind: ActionKind, target_x: int, target_y: int) -> InFlightActionDict:
    """Return a pending in-flight action of the requested kind.

    Args:
        kind: Action kind.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.

    Returns:
        A pending in-flight action record.
    """
    return make_in_flight_action(kind, target_x, target_y, get_current_time_ms())


class TestInFlightActionLifecycle:
    """The pending gate and the rejected-movement clear."""

    def test_a_confirmed_action_is_no_longer_in_flight(self, fake_env: FakeEnv) -> None:
        """Only a PENDING action is still in flight.

        The kind survives its resolution -- ``in_flight_action`` keeps
        ``kind="move"`` after the move confirms, and only ``outcome``
        moves to ``confirmed``. Gating on the kind alone therefore hands
        an already-resolved action back to the movement waiter, which
        re-runs its stall and rejection checks against a target the bot
        has finished with and can clear state the next plan depends on.
        """
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        bot = Bot("https://test.tankpit.com/", headless=True, world=WorldService())
        pending = _pending("move", 150, 150)
        bot._state_data = bot._state_data.copy()
        bot._state_data["in_flight_action"] = InFlightActionDict(
            kind=pending["kind"],
            target_x=pending["target_x"],
            target_y=pending["target_y"],
            started_ms=pending["started_ms"],
            outcome="confirmed",
        )

        assert has_in_flight_action(bot) is False

    def test_a_pending_action_is_in_flight(self, fake_env: FakeEnv) -> None:
        """Control: the same action while pending IS in flight."""
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        bot = Bot("https://test.tankpit.com/", headless=True, world=WorldService())
        bot._state_data = bot._state_data.copy()
        bot._state_data["in_flight_action"] = _pending("move", 150, 150)

        assert has_in_flight_action(bot) is True

    def test_a_teleport_is_not_cleared_by_a_stale_walk_rejection(self, fake_env: FakeEnv) -> None:
        """A rejected WALK to a tile does not cancel a teleport to it.

        ``is_move_target_failed`` records tiles the server refused to
        WALK to, and teleporting to a refused tile is the normal
        recovery -- the two commands are routed differently, which is
        why the fallback exists at all. Without the kind check the
        in-flight teleport is cleared on the strength of the very
        rejection it was planned to answer, and the bot replans in a
        loop.
        """
        from tankpit_bot.bot.tick_loop_actions import _clear_rejected_movement

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        ws.mark_move_target_failed(150, 150, get_current_time_ms())

        assert _clear_rejected_movement(bot, _pending("teleport", 150, 150)) is False

    def test_control_a_walk_to_that_tile_is_cleared(self, fake_env: FakeEnv) -> None:
        """Control: the same refused tile DOES clear a move action."""
        from tankpit_bot.bot.tick_loop_actions import _clear_rejected_movement

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        ws.mark_move_target_failed(150, 150, get_current_time_ms())

        assert _clear_rejected_movement(bot, _pending("move", 150, 150)) is True
