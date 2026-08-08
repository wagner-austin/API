"""Tests for command-error clearing in the tick loop.

One class per rejection path the loop must clear rather than stall on.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    ActionKind,
    InFlightActionDict,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_self_state
from tests.conftest import (
    FakeEnv,
    FakeFileSystem,
)


class TestClearCommandError:
    """The Supervisor (0x52) error code clears every in-flight action kind."""

    def _make_pending_action(
        self,
        kind: ActionKind,
        *,
        target_x: int = 100,
        target_y: int = 100,
    ) -> InFlightActionDict:
        """Build a pending in-flight action of the requested kind."""
        return InFlightActionDict(
            kind=kind,
            target_x=target_x,
            target_y=target_y,
            started_ms=get_current_time_ms(),
            outcome="pending",
        )

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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        ws.last_command_error = 0  # "You can't do this"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        ws.last_command_error = 7  # "Inventory full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
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

        assert outcome_counts(ws.ledger, "collect") == {"inventory_full": 1}

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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        ws.last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1
        container = ws.world_state["containers"]["150,150"]
        assert container["failed_pickups"] == 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts(ws.ledger, "collect") == {"clamped_transfer": 1}

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
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("collect", target_x=150, target_y=150)

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1
        assert ws.world_state["containers"] == {}
        # The disproof also marks the container memory desynced so the
        # collect cascade radars before pursuing further remembered
        # stock (user ruling 2026-07-30: one stale item = one radar).
        assert ws.container_desync_ms > 0
        from tankpit_bot.ledger.ring import outcome_counts

        assert outcome_counts(ws.ledger, "collect") == {"pickup_empty": 1}

    def test_command_error_clears_teleport_action(self, fake_env: FakeEnv) -> None:
        """A 0x52 ``You can't go there!`` aborts a pending teleport in < 1 s."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "TELEPORTING"
        action = self._make_pending_action("teleport", target_x=200, target_y=200)

        ws.last_command_error = 1  # "You can't go there!"
        result = _wait_for_movement_action(bot, action)

        assert result is False
        assert bot.get_state() == "IDLE"

    def test_scan_wait_drops_orphan_error_and_stays_pending(self, fake_env: FakeEnv) -> None:
        """A 0x52 code arriving during a scan wait is an orphan and is dropped.

        Radar dispatch (``CMD_RADAR`` 0x66, client ``Mb``) is not
        server-side rejectable: the server accepts every scan and
        replies with a ``0x4F`` result. Any 0x52 that lands during the
        scan wait belongs to a PRIOR action (typically one that already
        completed via a different wire signal like
        ``container_consumed``). The wait discards the orphan code and
        stays pending so the scan can complete normally.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_scan_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "SCANNING"
        action = self._make_pending_action("scan")

        ws.last_command_error = 8  # "Insufficient fuel"
        result = _wait_for_scan_action(bot, action)

        assert result is True
        assert bot.get_state() == "SCANNING"
        assert ws.last_command_error == -1

    def test_map_open_wait_drops_orphan_error_and_stays_pending(self, fake_env: FakeEnv) -> None:
        """A 0x52 code arriving during a map_open wait is an orphan and is dropped.

        Regression guard for live run 2026-07-06 20:20:59: a late-
        arriving ``code=4`` from a collect that already completed via
        ``container_consumed`` was misattributed to the following
        ``map_open``. HUNT could not acquire, session exited
        ``no_viable_targets`` at fuel 531 with a fully-stocked tank.
        Map_open dispatch (``CMD_MAP_OPEN`` 0x6C, client ``Nb``) is
        server-side unconditional, so no 0x52 code is ever a legitimate
        map_open rejection.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        action = self._make_pending_action("map_open")

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_map_open_action(bot, action)

        assert result is True
        assert bot.get_state() == "IDLE"
        assert ws.last_command_error == -1

    def test_teleport_wait_drops_orphan_empty_container(self, fake_env: FakeEnv) -> None:
        """A code=4 during a teleport wait is an orphan; teleport stays pending.

        Teleport (``CMD_MAP_TELEPORT`` 0x74) can draw codes 0/1/8; an
        ``Empty container`` (4) can only originate from a pickup and so
        must belong to a prior collect.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "TELEPORTING"
        action = self._make_pending_action("teleport", target_x=200, target_y=200)

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() == "TELEPORTING"
        assert ws.last_command_error == -1

    def test_move_wait_drops_orphan_tank_full(self, fake_env: FakeEnv) -> None:
        """A code=5 (tank full) during a move wait is orphaned.

        Move (``CMD_MOVE`` 0x70) can draw codes 0/1/8; ``Tank full`` (5)
        can only originate from a fuel pickup.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("move", target_x=150, target_y=150)

        ws.last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() == "MOVING"
        assert ws.last_command_error == -1

    def test_orphan_command_error_emits_diagnostic(
        self, fake_fs: FakeFileSystem, fake_env: FakeEnv
    ) -> None:
        """The orphan-drop path emits an ``orphan_command_error`` diagnostic.

        Observability guard: without the diagnostic, a wire race that
        drops an orphan code is invisible in the events stream. This
        test drives the map_open orphan path and asserts a single
        diagnostic with the action_kind and error_code fields.
        """
        from tankpit_bot.bot.tick_loop_actions import _wait_for_map_open_action
        from tankpit_bot.diagnostics.event_stream import load_event_records
        from tankpit_bot.runtime_logging import configure_bot_runtime_logging

        ws = WorldService()
        artifacts = configure_bot_runtime_logging("20260706-202100")
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        action = self._make_pending_action("map_open")

        ws.last_command_error = 4  # "Empty container"
        _wait_for_map_open_action(bot, action)

        records = [
            record
            for record in load_event_records(Path(artifacts["latest_events_path"]))
            if record["fields"].get("diagnostic_kind") == "orphan_command_error"
        ]
        assert len(records) == 1
        assert records[0]["fields"] == {
            "diagnostic_kind": "orphan_command_error",
            "action_kind": "map_open",
            "error_code": 4,
        }

    def test_no_command_error_lets_wait_continue(self, fake_env: FakeEnv) -> None:
        """No 0x52 error pending -> normal wait machinery runs."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_movement_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "MOVING"
        action = self._make_pending_action("move", target_x=150, target_y=150)

        # No error code set; default -1 means no rejection pending.
        result = _wait_for_movement_action(bot, action)

        # The action is still in-flight (not rejected, not stalled, not
        # blocked) so wait returns True to continue waiting.
        assert result is True

    def test_scan_wait_with_no_error_stays_pending(self, fake_env: FakeEnv) -> None:
        """The scan drain path is a no-op when no 0x52 code is pending."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_scan_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "SCANNING"
        action = self._make_pending_action("scan")

        assert ws.last_command_error == -1
        result = _wait_for_scan_action(bot, action)

        assert result is True
        assert bot.get_state() == "SCANNING"

    def test_scan_and_map_open_whitelists_are_empty(self) -> None:
        """Whitelist invariant: scan and map_open are never rejected by any 0x52 code.

        Radar (``CMD_RADAR`` 0x66) and map_open (``CMD_MAP_OPEN`` 0x6C)
        are server-side unconditional. If a future change adds a code
        to either whitelist,
        :func:`~tankpit_bot.bot.tick_loop_actions._wait_for_scan_action`
        and :func:`~tankpit_bot.bot.tick_loop_actions._wait_for_map_open_action`
        must be updated to check the applicable-rejection outcome and
        transition the action -- currently they only call
        :func:`~tankpit_bot.bot.tick_loop_actions._drain_orphan_command_error`
        which never transitions.
        """
        from tankpit_bot.bot.tick_loop_command_errors import _COMMAND_ERROR_APPLICABILITY

        assert _COMMAND_ERROR_APPLICABILITY["scan"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["map_open"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["none"] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY["shoot"] == frozenset()
