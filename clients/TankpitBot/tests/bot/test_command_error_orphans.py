"""Tests for orphan 0x52 codes: a code that cannot belong to the in-flight action.

Split from ``test_tick_loop_command_error.py`` (601 lines, over the
600-line ceiling): that file owns the rejections that DO apply and the
clears they cause, this one owns the codes that do not -- each is
dropped, reported as an ``orphan_command_error``, and never transitions
the action -- plus the per-kind whitelist that decides which is which.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import ActionKind, BotState
from tankpit_bot.sniffer.world_service import WorldService
from tests._runtime_logging_support import capture_runtime_events, event_fields
from tests.bot._state_machine_fixtures import _pending_action
from tests.conftest import FakeEnv, FakeFileSystem


class TestOrphanCommandErrors:
    """A 0x52 code outside the in-flight kind's whitelist is dropped and reported."""

    def test_no_pending_error_emits_no_orphan_diagnostic(self, fake_env: FakeEnv) -> None:
        """An empty error slot is silence, not an orphan worth reporting.

        ``check_and_clear_command_error`` answers ``-1`` when nothing is
        pending, and the scan/map_open wait paths call it on EVERY tick
        they wait. Reporting that as an orphan stamps an
        ``orphan_command_error`` with ``error_code=-1`` into the
        diagnostic stream once per waiting tick, which is where the
        scorecard reads its rejection counts from.

        The control below proves the emitter fires for a genuine code,
        so silence here is the guard rather than a dead emitter.
        """
        from tankpit_bot.bot.tick_loop_command_errors import _drain_orphan_command_error

        ws = WorldService()
        action = _pending_action(ActionKind.SCAN)

        with capture_runtime_events() as records:
            _drain_orphan_command_error(ws, action)

        kinds = [event_fields(record).get("diagnostic_kind") for record in records]
        assert "orphan_command_error" not in kinds

    def test_control_a_real_orphan_code_does_emit(self, fake_env: FakeEnv) -> None:
        """Control: a genuine 0x52 arriving during a scan wait is reported."""
        from tankpit_bot.bot.tick_loop_command_errors import _drain_orphan_command_error

        ws = WorldService()
        ws.last_command_error = 4
        action = _pending_action(ActionKind.SCAN)

        with capture_runtime_events() as records:
            _drain_orphan_command_error(ws, action)

        kinds = [event_fields(record).get("diagnostic_kind") for record in records]
        assert "orphan_command_error" in kinds

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
        bot._state_data["state"] = BotState.SCANNING
        action = _pending_action(ActionKind.SCAN)

        ws.last_command_error = 8  # "Insufficient fuel"
        result = _wait_for_scan_action(bot, action)

        assert result is True
        assert bot.get_state() is BotState.SCANNING
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
        bot._state_data["state"] = BotState.IDLE
        action = _pending_action(ActionKind.MAP_OPEN)

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_map_open_action(bot, action)

        assert result is True
        assert bot.get_state() is BotState.IDLE
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
        bot._state_data["state"] = BotState.TELEPORTING
        action = _pending_action(ActionKind.TELEPORT, target_x=200, target_y=200)

        ws.last_command_error = 4  # "Empty container"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() is BotState.TELEPORTING
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
        bot._state_data["state"] = BotState.MOVING
        action = _pending_action(ActionKind.MOVE, target_x=150, target_y=150)

        ws.last_command_error = 5  # "Tank full"
        result = _wait_for_movement_action(bot, action)

        assert result is True
        assert bot.get_state() is BotState.MOVING
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
        bot._state_data["state"] = BotState.IDLE
        action = _pending_action(ActionKind.MAP_OPEN)

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

    def test_scan_wait_with_no_error_stays_pending(self, fake_env: FakeEnv) -> None:
        """The scan drain path is a no-op when no 0x52 code is pending."""
        from tankpit_bot.bot.tick_loop_actions import _wait_for_scan_action

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = BotState.SCANNING
        action = _pending_action(ActionKind.SCAN)

        assert ws.last_command_error == -1
        result = _wait_for_scan_action(bot, action)

        assert result is True
        assert bot.get_state() is BotState.SCANNING

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

        assert _COMMAND_ERROR_APPLICABILITY[ActionKind.SCAN] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY[ActionKind.MAP_OPEN] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY[ActionKind.NONE] == frozenset()
        assert _COMMAND_ERROR_APPLICABILITY[ActionKind.SHOOT] == frozenset()
