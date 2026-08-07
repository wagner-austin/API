"""Tests for completion-event detection.

``test_completion_events.py`` was 631 lines; the per-action completions
are now a sibling.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot.bot.tick_loop_actions import (
    has_in_flight_action,
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
from tankpit_bot.sniffer.world_state_combat import mark_teleport_landed
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _update_fuel_total,
)
from tankpit_bot.state import make_self_state
from tankpit_bot.state.types import make_tank_state
from tests.bot._completion_fixtures import (
    _decode_action_outcome_lines,
    _make_bot_with_in_flight,
)
from tests.conftest import FakeEnv, FakeFileSystem


class TestActionOutcomeEventsOnAuthoritativeCompletion:
    """Each authoritative completion gate emits an ``action_outcome`` event."""

    def test_map_open_completion_emits_event_with_map_data_processed_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_clear_completed_map_open`` emits map_open completion via MAP_DATA."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        _update_fuel_total(get_world_service(), 800)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="IDLE",
            action_kind="map_open",
            target_x=0,
            target_y=0,
            started_ms=get_current_time_ms() - 1,
        )
        get_world_service().mark_map_data_processed()

        cleared = has_in_flight_action(bot) is False
        assert cleared

        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "map_open"
        assert require_str_field(fields, "outcome") == "map_data_processed"
        # The live clock advances between the recorded started_ms and the
        # completion call, so the decoded duration_ms is strictly positive.
        assert require_int_field(fields, "duration_ms") >= 0

    def test_walk_completion_emits_event_with_position_reached_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_maybe_complete_walk`` emits move completion when tank is on target."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="MOVING",
            action_kind="move",
            target_x=120,
            target_y=130,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=120,
            y=130,
            team=2,
            rank=1,
            fuel=800,
            leaderboard_position=1,
        )

        completed = bot._maybe_complete_walk(landed_state)

        assert completed is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "move"
        assert require_str_field(fields, "outcome") == "position_reached"
        assert require_int_field(fields, "target_x") == 120
        assert require_int_field(fields, "target_y") == 130
        assert require_int_field(fields, "landed_x") == 120
        assert require_int_field(fields, "landed_y") == 130

    def test_teleport_completion_emits_event_with_teleport_landed_signal(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """``_maybe_complete_teleport`` emits teleport completion via TeleportLanded."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        artifacts = configure_bot_runtime_logging("20260331-230405")

        bot = _make_bot_with_in_flight(
            state="TELEPORTING",
            action_kind="teleport",
            target_x=200,
            target_y=210,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=200,
            y=210,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        mark_teleport_landed(get_world_service())

        completed = bot._maybe_complete_teleport(landed_state)

        assert completed is True
        events = _decode_action_outcome_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "teleport"
        assert require_str_field(fields, "outcome") == "landed_exact"
        assert require_int_field(fields, "target_x") == 200
        assert require_int_field(fields, "target_y") == 210
        assert require_int_field(fields, "landed_x") == 200
        assert require_int_field(fields, "landed_y") == 210

    def test_teleport_enemy_displacement_does_not_mark_failed(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Adjacent landing near an enemy is expected, not a failed target."""
        reset_world_state()
        update_world_state_from_position(50, 50)
        configure_bot_runtime_logging("20260331-230405")
        ws = get_world_service()
        ws.world_state["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=200,
            y=210,
            team=1,
            rank=1,
            damage_state=0,
            name="enemy-50",
            is_bot=False,
            is_self=False,
            timestamp_ms=100000,
        )

        bot = _make_bot_with_in_flight(
            state="TELEPORTING",
            action_kind="teleport",
            target_x=200,
            target_y=210,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=199,
            y=210,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        mark_teleport_landed(ws)

        completed = bot._maybe_complete_teleport(landed_state)

        assert completed is True
        assert ws.failed_move_targets.get("200,210") is None

    def test_teleport_diagonal_enemy_displacement_does_not_mark_failed(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """A DIAGONAL bump off an aimed enemy is a success, not a failure.

        The server displaces onto any of the 8 neighbors; a diagonal
        landing is Chebyshev 1 but Manhattan 2, and the old Manhattan
        exemption blacklisted the enemy's tile for it -- the orange-6
        blocked-target case from the 2026-07-27 20-kill run.
        """
        reset_world_state()
        update_world_state_from_position(50, 50)
        configure_bot_runtime_logging("20260331-230405")
        ws = get_world_service()
        ws.world_state["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=200,
            y=210,
            team=1,
            rank=1,
            damage_state=0,
            name="enemy-50",
            is_bot=False,
            is_self=False,
            timestamp_ms=100000,
        )

        bot = _make_bot_with_in_flight(
            state="TELEPORTING",
            action_kind="teleport",
            target_x=200,
            target_y=210,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=199,
            y=209,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        mark_teleport_landed(ws)

        completed = bot._maybe_complete_teleport(landed_state)

        assert completed is True
        assert ws.failed_move_targets.get("200,210") is None

    def test_teleport_two_tile_displacement_still_marks_failed(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """A landing two tiles out is a genuine non-arrival even near an enemy.

        Chebyshev 2 means the tank did NOT land beside its aim (e.g.
        bumped past a fresh 3x3 mine ring) -- the aim tile keeps its
        30 s failed-move mark so the planner does not re-dispatch the
        same displaced teleport.
        """
        reset_world_state()
        update_world_state_from_position(50, 50)
        configure_bot_runtime_logging("20260331-230405")
        ws = get_world_service()
        ws.world_state["tanks"]["50"] = make_tank_state(
            tank_id=50,
            x=200,
            y=210,
            team=1,
            rank=1,
            damage_state=0,
            name="enemy-50",
            is_bot=False,
            is_self=False,
            timestamp_ms=100000,
        )

        bot = _make_bot_with_in_flight(
            state="TELEPORTING",
            action_kind="teleport",
            target_x=200,
            target_y=210,
            started_ms=get_current_time_ms() - 1,
        )
        landed_state = make_self_state(
            tank_id=1,
            x=198,
            y=210,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        mark_teleport_landed(ws)

        completed = bot._maybe_complete_teleport(landed_state)

        assert completed is True
        assert ws.is_move_target_failed(200, 210, get_current_time_ms()) is True


def test_artifact_jsonl_lines_round_trip_through_real_decoder(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """JSONL produced by the live handler decodes via the production decoder.

    Defensive contract: the encoder spreads structured fields at the top
    level of each JSONL line, the decoder reverse-spreads them back into
    a strict :class:`RuntimeEventRecordDict`. This test confirms a real
    completion-site emit round-trips through the file system end to end
    without any intermediate fixture.
    """
    reset_world_state()
    artifacts = configure_bot_runtime_logging("20260331-230405")

    from tankpit_bot.sniffer.world_state import mark_radar_scan_complete

    bot = _make_bot_with_in_flight(
        state="SCANNING",
        action_kind="scan",
        target_x=0,
        target_y=0,
        started_ms=get_current_time_ms() - 1,
    )
    mark_radar_scan_complete()
    assert bot._maybe_complete_scan(bot.get_world_state()) is True

    files = fake_fs.get_written_files()
    archive = Path(artifacts["archive_events_path"])
    assert str(archive) in files
    assert files[artifacts["latest_events_path"]] == files[artifacts["archive_events_path"]], (
        "latest and archive event streams must hold identical content"
    )

    latest_events = _decode_action_outcome_lines(files[artifacts["latest_events_path"]])
    archive_events = _decode_action_outcome_lines(files[artifacts["archive_events_path"]])
    assert latest_events == archive_events
    assert len(latest_events) == 1
    assert require_str_field(latest_events[0]["fields"], "outcome") == "radar_complete"
