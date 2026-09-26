"""Tests for command-error clearing in the tick loop.

One class per rejection path the loop must clear rather than stall on.
The codes that do not apply to the in-flight kind -- orphans, dropped
and reported -- are :mod:`tests.bot.test_command_error_orphans`.
"""

from __future__ import annotations

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    ActionKind,
    BotState,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.ledger.events import ActionKind as LedgerActionKind
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_self_state
from tests._runtime_logging_support import capture_runtime_events, event_fields
from tests.bot._state_machine_fixtures import _pending_action
from tests.conftest import (
    FakeEnv,
)


class TestClearCommandError:
    """The Supervisor (0x52) error code clears every in-flight action kind."""

    def test_clear_command_error_is_silent_when_nothing_is_pending(self, fake_env: FakeEnv) -> None:
        """The movement path reports no orphan when there is no error at all.

        Same shape as the scan path above, but the return value is
        identical either way (``False``), so only the absence of the
        diagnostic distinguishes the two. Without the guard, ``-1`` is
        tested against the move whitelist, misses, and is announced as
        an orphan on every tick a move waits.
        """
        from tankpit_bot.bot.tick_loop_command_errors import _clear_command_error

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        action = _pending_action(ActionKind.MOVE)

        with capture_runtime_events() as records:
            result = _clear_command_error(bot, action)

        assert result is False
        kinds = [event_fields(record).get("diagnostic_kind") for record in records]
        assert "orphan_command_error" not in kinds

    def test_command_error_clears_collect_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You can't do this`` (code 0) aborts a pending collect in < 1 s.

        Without the hook the bot waited the full
        ``action_stall_timeout_ms`` (10 s) on every server denial; live
        run 20260620-184223 wasted 40 s of session time on four such
        rejections. Illegal geometry blacklists the container position
        via ``failed_pickups`` (unlike code 4, which removes the
        belief outright).
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 0  # "You can't do this"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE
        assert ws.last_command_error == -1

    def test_cant_go_on_collect_records_a_movement_rejection(self, fake_env: FakeEnv) -> None:
        """A cant_go rejecting a walk-pickup lands in the movement record.

        Run bot-20260730-110x ticks 95-107: twelve consecutive
        rejected walk-pickups under fire were invisible to the
        per-tile move marks because collect rejections only fed
        ``failed_pickups`` — the escape's movement-dead detector
        needs the shared "the server refused a move" fact regardless
        of the command kind.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 1  # "You can't go there!"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert ws.recent_movement_rejections(get_current_time_ms(), 10000) == 1

    def test_non_movement_rejection_is_not_recorded(self, fake_env: FakeEnv) -> None:
        """A code-0 collect rejection is not a movement refusal."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 0  # "You can't do this"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert ws.recent_movement_rejections(get_current_time_ms(), 10000) == 0

    def test_command_error_clears_collect_on_inventory_full(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Inventory full`` (code 7) aborts the pickup, keeps the container.

        Empirical guard: live capture 20260620-190728 / 20260620-190830
        delivered ``error_code=7`` over the wire after pickup dispatches
        at full inventory. Without code 7 in the blocking set the
        collect would idle the full ``action_stall_timeout_ms`` (10 s)
        before replanning. User mechanic (2026-07-18): containers fill
        whatever is empty and code 7 fires only at all-slots-full --
        the container is NOT blacklisted (it is fine; the tank is
        full) and every slot belief reconciles up to capacity, the
        rejection being an authoritative absolute inventory statement.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.state.types import WorldStateDict, make_container_state

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        ws.world_state = WorldStateDict(
            **{
                **ws.world_state,
                "containers": {
                    "150,150": make_container_state(
                        x=150,
                        y=150,
                        is_fuel=False,
                        volume=0,
                        timestamp_ms=get_current_time_ms(),
                        failed_pickups=0,
                    )
                },
            }
        )
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 7  # "Inventory full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE
        assert ws.last_command_error == -1
        container = ws.world_state["containers"]["150,150"]
        assert container["failed_pickups"] == 0
        # No self_state rank in this fixture-free world? position update
        # created one at rank 0 -> capacity applies; all slots snapped up.
        from tankpit_bot.physics.capacity import inventory_capacity

        rank = ws.world_state["self_state"]["rank"] if ws.world_state["self_state"] else 0
        cap = inventory_capacity(rank)
        inv = ws.inventory_state
        assert inv["armor_shields"]["count"] >= cap
        assert inv["dual_shots"]["count"] >= cap
        assert inv["missile_shots"]["count"] >= cap
        assert inv["homing_shots"]["count"] >= cap
        assert inv["extra_radars"]["count"] >= cap
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts(ws.ledger, LedgerActionKind.COLLECT) == {"inventory_full": 1}

    def test_command_error_tank_full_does_not_mark_failed_pickup(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Tank full`` (code 5) clears the action WITHOUT blacklisting.

        Bug 0.3 (2026-07-06): a code=5 rejection means the container
        was not empty -- the server refused the transfer because the
        tank could not accept it. Under Bug 0.2's pre-dispatch gate (now ``pickup_not_worth_walk``)
        pre-dispatch gate the overflow scenario cannot occur in the
        normal flow, so a surviving code=5 is a race between
        planner-time and dispatch-time fuel state. Blacklisting a
        still-full container is wrong -- next tick with headroom will
        successfully consume it. The in-flight action is still
        cleared (the planner replans this tick) but ``failed_pickups``
        stays at 0 so the container remains a candidate. Pre-fix
        behavior: the 22:37 fuel-loop's four consecutive
        partial-transfer + code=5 events blacklisted four still-full
        fuel containers.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.state.types import (
            WorldStateDict,
            make_container_state,
        )

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
                        timestamp_ms=get_current_time_ms(),
                        failed_pickups=0,
                    )
                },
            }
        )
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE
        assert ws.last_command_error == -1
        container = ws.world_state["containers"]["150,150"]
        assert container["failed_pickups"] == 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts(ws.ledger, LedgerActionKind.COLLECT) == {"clamped_transfer": 1}

    def test_command_error_empty_container_removes_belief(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``Empty container`` (code 4) deletes the container belief.

        The server says the container is drained, so the volume the
        planner acted on is contradicted -- the belief is removed
        outright rather than blacklisted. (Until 2026-07-19 this
        removal was done by the DOM game-log "Empty container"
        consumer one or two ticks later; the wire code is the same
        signal, earlier, and the DOM channel is now witness-only.)
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action
        from tankpit_bot.state.types import (
            WorldStateDict,
            make_container_state,
        )

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
                        timestamp_ms=get_current_time_ms(),
                        failed_pickups=0,
                    )
                },
            }
        )
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.COLLECT, target_x=150, target_y=150)

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE
        assert ws.last_command_error == -1
        assert ws.world_state["containers"] == {}
        # The disproof also marks the container memory desynced so the
        # collect cascade radars before pursuing further remembered
        # stock (user ruling 2026-07-30: one stale item = one radar).
        assert ws.container_desync_ms > 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts(ws.ledger, LedgerActionKind.COLLECT) == {"pickup_empty": 1}

    def test_command_error_clears_teleport_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You can't go there!`` aborts a pending teleport in < 1 s."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.TELEPORTING
        action = _pending_action(ActionKind.TELEPORT, target_x=200, target_y=200)

        ws.last_command_error = 1  # "You can't go there!"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE

    def test_already_there_on_move_clears_and_marks_the_target(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You are already there`` (code 6) aborts a move and tombstones its tile.

        A move whose target IS the dispatch position draws a 6 and no
        completing wire signal ever comes. Until 2026-09-05 the code
        was orphan-discarded, so the tile was never marked failed and
        the planner re-derived the identical zero-length move every
        tick: demo-1 17:48-19:00 dispatched ``move -> (239,48)`` from
        (239,48) 395 times over 72 minutes. The mark is the loop
        break -- the next replan skips the tile and the cascade moves
        on.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(239, 48)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.MOVE, target_x=239, target_y=48)

        ws.last_command_error = 6  # "You are already there"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() is BotState.IDLE
        assert ws.last_command_error == -1
        assert ws.is_move_target_failed(239, 48, get_current_time_ms()) is True

    def test_no_command_error_lets_wait_continue(self, fake_env: FakeEnv) -> None:
        """No 0x52 error pending -> normal wait machinery runs."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.MOVE, target_x=150, target_y=150)

        # No error code set; default -1 means no rejection pending.
        result = _wait_for_movement_action(bot, action)

        # The action is still in-flight (not rejected, not stalled, not
        # blocked) so wait returns True to continue waiting.
        assert result is True
