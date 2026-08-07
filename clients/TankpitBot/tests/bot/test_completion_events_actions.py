"""Tests for per-action completion events."""

from __future__ import annotations

from tankpit_bot.bot.tick_loop_actions import (
    _clear_stalled_action,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tankpit_bot.runtime_records import (
    require_int_field,
    require_str_field,
)
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.state import make_self_state
from tests.bot._completion_fixtures import (
    _decode_action_outcome_lines,
    _make_bot_with_in_flight,
)
from tests.conftest import FakeEnv, FakeFileSystem


class TestCompletionEventActions:
    """Tests for per-action completion events."""

    def test_collection_completion_emits_event_with_position_reached_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_maybe_complete_collection`` emits when tank reaches the pickup tile."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="COLLECTING",
            action_kind="collect",
            target_x=80,
            target_y=90,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=80,
            y=90,
            team=2,
            rank=1,
            fuel=750,
            leaderboard_position=1,
        )

        completed = bot._maybe_complete_collection(bot.get_world_state(), landed_state)

        assert completed is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "collect"
        assert require_str_field(fields, "outcome") == "position_reached"
        assert require_int_field(fields, "target_x") == 80
        assert require_int_field(fields, "target_y") == 90

    def test_collection_completion_defers_while_a_command_error_is_pending(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """A pending 0x52 blocks position-completion until it is attributed.

        Regression (sim-found 2026-07-22): a same-tile pickup at a
        consumed container completes instantly by position, so the
        code=4 "empty container" that should delete the stale belief
        orphans and the bot re-clicks the ghost forever. The
        completion must yield to the in-flight error handler.
        """
        reset_world_state()
        update_world_state_from_position(50, 50)
        configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="COLLECTING",
            action_kind="collect",
            target_x=80,
            target_y=90,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=80,
            y=90,
            team=2,
            rank=1,
            fuel=750,
            leaderboard_position=1,
        )
        ws = get_world_service()
        ws.last_command_error = 4

        assert bot._maybe_complete_collection(bot.get_world_state(), landed_state) is False
        assert bot.get_state() == "COLLECTING"

        ws.last_command_error = -1
        assert bot._maybe_complete_collection(bot.get_world_state(), landed_state) is True

    def test_collection_completion_emits_event_with_container_consumed_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_maybe_complete_collection`` emits container_consumed when target vanished."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="COLLECTING",
            action_kind="collect",
            target_x=80,
            target_y=90,
            started_ms=get_current_time_ms() - 1,
        )
        non_target_self = make_self_state(
            tank_id=1,
            x=70,
            y=85,
            team=2,
            rank=1,
            fuel=750,
            leaderboard_position=1,
        )

        completed = bot._maybe_complete_collection(bot.get_world_state(), non_target_self)

        assert completed is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "collect"
        assert require_str_field(fields, "outcome") == "container_consumed"
        assert require_int_field(fields, "target_x") == 80
        assert require_int_field(fields, "target_y") == 90
        assert require_int_field(fields, "landed_x") == 70
        assert require_int_field(fields, "landed_y") == 85

    def test_scan_completion_emits_event_with_radar_scan_complete_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_maybe_complete_scan`` emits scan completion via radar_scan_complete."""
        from tankpit_bot.sniffer.world_state import mark_radar_scan_complete

        reset_world_state()
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="SCANNING",
            action_kind="scan",
            target_x=0,
            target_y=0,
            started_ms=get_current_time_ms() - 1,
        )
        mark_radar_scan_complete()

        completed = bot._maybe_complete_scan(bot.get_world_state())

        assert completed is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "scan"
        assert require_str_field(fields, "outcome") == "radar_complete"

    def test_stalled_action_emits_event_with_stall_timeout_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_clear_stalled_action`` emits with stall_timeout for forced clearance."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="MOVING",
            action_kind="move",
            target_x=180,
            target_y=200,
            started_ms=1,
        )

        cleared = _clear_stalled_action(bot, bot._state_data["in_flight_action"])

        assert cleared is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "move"
        assert require_str_field(fields, "outcome") == "stall_timeout"
        assert require_int_field(fields, "target_x") == 180
        assert require_int_field(fields, "target_y") == 200
        # timeout_ms carries the configured action stall timeout the gate
        # compared elapsed time against; any non-zero AI config value is
        # acceptable.
        assert require_int_field(fields, "timeout_ms") > 0
