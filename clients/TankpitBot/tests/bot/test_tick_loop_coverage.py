"""Tests for the tick loop's timing and receipt paths.

Early-wake sleep, the wire-silence watchdog, friendly-fire disproof,
and the drain and teleport-precondition receipts.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    InFlightActionDict,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_self_state
from tests.bot._tick_loop_fakes import _FakePage
from tests.conftest import (
    FakeEnv,
)


class TestEarlyWakeSleep:
    """Tests for the early-wake between-tick sleep."""

    def test_idle_wait_sleeps_the_full_window(self) -> None:
        """With no in-flight action the sleep is one uninterrupted window."""
        from tankpit_bot.bot.tick_loop import _wait_between_ticks
        from tankpit_bot.protocol.commands import TICK_RATE_MS

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        waits: list[float] = []

        class _RecordingPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)

        waited = _wait_between_ticks(bot, _RecordingPage())

        assert waited == TICK_RATE_MS
        assert waits == [float(TICK_RATE_MS)]

    def test_in_flight_wait_wakes_on_fresh_wire_traffic(self) -> None:
        """Fresh CDP traffic during an in-flight action ends the sleep early.

        The Artax-era fixed sleep waited out the whole 2 s window after
        a completion message arrived, drifting one server tick behind a
        human who acts the moment the tank arrives (user observation
        2026-07-30: "you can click as soon as it reaches it's
        destination and it'll go instantly to the next action").
        """
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import (
            _WAKE_SLICE_MS,
            _wait_between_ticks,
        )
        from tankpit_bot.browser import get_current_time_ms

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        bot._state_data["in_flight_action"] = make_in_flight_action(
            "scan", 0, 0, get_current_time_ms()
        )
        waits: list[float] = []

        class _TrafficPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)
                bot._cdp_message_buffer.append("traffic")

        waited = _wait_between_ticks(bot, _TrafficPage())

        assert waited == _WAKE_SLICE_MS
        assert waits == [float(_WAKE_SLICE_MS)]

    def test_in_flight_wait_without_traffic_runs_the_window(self) -> None:
        """A quiet wire keeps the in-flight sleep at the full window."""
        from tankpit_bot.bot.states import make_in_flight_action
        from tankpit_bot.bot.tick_loop import (
            _WAKE_SLICE_MS,
            _wait_between_ticks,
        )
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.protocol.commands import TICK_RATE_MS

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._cdp_message_buffer = []
        bot._state_data["in_flight_action"] = make_in_flight_action(
            "scan", 0, 0, get_current_time_ms()
        )
        waits: list[float] = []

        class _QuietPage(_FakePage):
            def wait_for_timeout(self, timeout: float) -> None:
                waits.append(timeout)

        waited = _wait_between_ticks(bot, _QuietPage())

        assert waited == TICK_RATE_MS
        assert waits == [float(_WAKE_SLICE_MS)] * (TICK_RATE_MS // _WAKE_SLICE_MS)


class TestWireSilenceWatchdog:
    """Tests for the connection-lost wire-silence watchdog."""

    def test_disarmed_before_first_game_message(self) -> None:
        """A zero stamp (boot, lobby) never trips the watchdog."""
        from tankpit_bot.bot.tick_body import _check_wire_silence

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.world.last_game_message_ms == 0

        _check_wire_silence(bot)

    def test_fresh_traffic_passes(self) -> None:
        """A recent game message keeps the session alive."""
        from tankpit_bot.bot.tick_body import _check_wire_silence

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot.world.last_game_message_ms = get_current_time_ms() - 1_000

        _check_wire_silence(bot)

    def test_silence_past_limit_raises_connection_lost(self) -> None:
        """Wire silence past the limit ends the session with a receipt.

        Session 3 of run 20260730: the game socket died at 11:58:32,
        the page auto-reconnected to the lobby (socket OPEN, so the
        ws-ready gate passed), and the bot injected map_open into a
        dead session for 43 minutes -- 243 consecutive stalls, zero
        inbound world messages. The watchdog turns that zombie into a
        90-second clean exit the harness can relaunch from.
        """
        from tankpit_bot.bot.session_exit import SessionExitError
        from tankpit_bot.bot.tick_body import (
            _WIRE_SILENCE_LIMIT_MS,
            _check_wire_silence,
        )

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot.world.last_game_message_ms = get_current_time_ms() - _WIRE_SILENCE_LIMIT_MS - 1

        with pytest.raises(SessionExitError) as exc_info:
            _check_wire_silence(bot)

        assert exc_info.value.reason == "connection_lost"
        assert "no game wire message" in exc_info.value.detail


class TestFriendlyFireDisproof:
    """Tests for consuming err=3 friendly_fire as target disproof."""

    def test_friendly_fire_blocks_target_and_releases_matching_lock(self) -> None:
        """One err=3 blocklists the id and clears the matching combat lock.

        Session 4 of run 20260730 (20:36): Yuppler left the game, the
        0x58 grace kept his registry entry, every map open re-stamped
        the ghost's freshness, and the bot fired 43 consecutive
        rejected shots. The disproof turns the first rejection into a
        block + lock release.
        """
        from tankpit_bot.bot.tick_combat_feedback import _disprove_target_by_friendly_fire

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 1229,
                "combat_target_x": 245,
                "combat_target_y": 76,
            }
        )

        _disprove_target_by_friendly_fire(bot, 1229, "Yuppler")

        assert "1229" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == -1
        assert bot._ai_state["combat_target_x"] == 0

    def test_friendly_fire_keeps_unrelated_lock(self) -> None:
        """Disproving a non-locked target leaves the held lock alone."""
        from tankpit_bot.bot.tick_combat_feedback import _disprove_target_by_friendly_fire

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 514,
                "combat_target_x": 117,
                "combat_target_y": 139,
            }
        )

        _disprove_target_by_friendly_fire(bot, 1229, "Yuppler")

        assert "1229" in bot._ai_state["blocked_combat_targets"]
        assert bot._ai_state["combat_target_id"] == 514


class TestDrainReceipt:
    """A code=4 riding our own pickup is a drain receipt, not a desync."""

    def test_own_drain_code4_does_not_mark_desync(self, fake_env: FakeEnv) -> None:
        """Flag s9-4: a +241 pickup drained the container, the same
        click's code=4 marked memory desynced, and a paid rescan
        re-learned what the bot had just done itself.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.state.types import WorldStateDict, make_container_state

        started_ms = get_current_time_ms()
        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "self_state": make_self_state(
                    tank_id=1,
                    x=100,
                    y=100,
                    team=1,
                    rank=0,
                    fuel=1000,
                    leaderboard_position=0,
                ),
                "containers": {
                    "150,150": make_container_state(
                        x=150,
                        y=150,
                        is_fuel=True,
                        volume=400,
                        timestamp_ms=started_ms,
                        failed_pickups=0,
                    )
                },
            }
        )
        # The wire credited US a gain within the click -- the
        # corrected discriminator (2026-08-13): a tile-record test
        # also matched the removal broadcast riding an already-empty
        # click, so 23/23 stale empties classified as own drains and
        # the rescan gate never fired (run arterial). Wire order is
        # dispatch -> gain -> code=4, so the gain is stamped at/after
        # the action's ``started_ms``.
        ws.record_own_gain(get_current_time_ms())
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = InFlightActionDict(
            kind="collect",
            target_x=150,
            target_y=150,
            started_ms=started_ms,
            outcome="pending",
        )
        ws.last_command_error = 4

        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert ws.container_desync_ms == 0


class TestDetonationInterruptedWalk:
    """A cant_go riding a walk-over detonation never blames the container."""

    def _make_world_with_container(self) -> WorldService:
        """Build a session holding one full equipment-container belief."""
        from tankpit_bot.state.types import WorldStateDict, make_container_state

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "self_state": make_self_state(
                    tank_id=1,
                    x=100,
                    y=100,
                    team=1,
                    rank=0,
                    fuel=1000,
                    leaderboard_position=0,
                ),
                "containers": {
                    "103,147": make_container_state(
                        x=103,
                        y=147,
                        is_fuel=False,
                        volume=0,
                        timestamp_ms=get_current_time_ms(),
                        failed_pickups=0,
                    )
                },
            }
        )
        return ws

    def test_cant_go_with_detonation_in_window_keeps_container_clean(
        self, fake_env: FakeEnv
    ) -> None:
        """Flag 6 (run arterial 2026-08-13 22:25:45): the detonation on
        (103,143) halted the walk to equipment at (103,147), the server
        refused the remainder with code=1, and the failed_pickups
        increment demoted the innocent (and reachable) container -- the
        replan paid a 60-fuel teleport to different equipment four
        tiles further away.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        started_ms = get_current_time_ms()
        ws = self._make_world_with_container()
        # Wire order: dispatch -> walk -> detonation on own tile ->
        # cant_go for the refused remainder, all inside the window.
        ws.last_own_mine_hit_ms = get_current_time_ms()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = InFlightActionDict(
            kind="collect",
            target_x=103,
            target_y=147,
            started_ms=started_ms,
            outcome="pending",
        )
        ws.last_command_error = 1

        result = _wait_for_movement_action(bot, action)

        assert result is False
        container = ws.world_state["containers"]["103,147"]
        assert container["failed_pickups"] == 0

    def test_cant_go_without_detonation_still_blames_the_container(self, fake_env: FakeEnv) -> None:
        """A cant_go with no detonation in the window is a genuine
        terrain refusal and keeps the failed-pickup mark (the
        2026-07-06 semantic split's code-0/1 arm)."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = self._make_world_with_container()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = InFlightActionDict(
            kind="collect",
            target_x=103,
            target_y=147,
            started_ms=get_current_time_ms(),
            outcome="pending",
        )
        ws.last_command_error = 1

        result = _wait_for_movement_action(bot, action)

        assert result is False
        container = ws.world_state["containers"]["103,147"]
        assert container["failed_pickups"] == 1


class TestTeleportPreconditionReceipt:
    """Code 0 on a teleport blames the map state, never the tile."""

    def test_code0_teleport_rejection_leaves_tile_unmarked(self, fake_env: FakeEnv) -> None:
        """Flag s10-1: the map closed server-side while the snapshot
        still read open; the rejected larder landing was innocent and
        must stay teleportable for the deferred retry."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "TELEPORTING"
        action = InFlightActionDict(
            kind="teleport",
            target_x=221,
            target_y=209,
            started_ms=get_current_time_ms(),
            outcome="pending",
        )
        ws.last_command_error = 0

        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert ws.is_move_target_failed(221, 209, get_current_time_ms()) is False
