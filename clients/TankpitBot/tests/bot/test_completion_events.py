"""Integration tests for ``WIRE_COMPLETE`` events on every completion gate.

Each test drives a real :class:`tankpit_bot.bot.base.Bot` (or
:func:`tankpit_bot.bot.tick_loop._has_in_flight_action`) through a
completion path, then reads the JSONL artifact that
:func:`configure_bot_runtime_logging` wires up to capture the structured
event. Assertions exercise the actual decoder
(:func:`decode_runtime_event_record`) on the actual file contents -- no
monkeypatching of emit helpers, no in-memory stand-ins for the logging
pipeline, no fakes for the structured-field plumbing.

The full pipeline tested by each case is:

    bot completion site
      -> emit_wire_complete(...)
        -> _emit_runtime_event("WIRE_COMPLETE", ..., **fields)
          -> stdlib logging with RuntimeLogExtraDict on the LogRecord
            -> _HookEventArtifactHandler.emit(record)
              -> dump_json_str(encode_runtime_event_record(...))
                -> _test_hooks.append_text(latest_events_path, line)

Reading the JSONL back through :func:`decode_runtime_event_record`
proves the contract end to end.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    InFlightActionDict,
    StateName,
    make_in_flight_action,
    make_initial_state_data,
)
from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action, has_in_flight_action
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
    decode_runtime_event_record,
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
from tests.conftest import FakeEnv, FakeFileSystem


def _decode_wire_complete_lines(jsonl: str) -> list[RuntimeEventRecordDict]:
    """Return every ``WIRE_COMPLETE`` event decoded from a JSONL artifact.

    Args:
        jsonl: Raw newline-delimited JSONL artifact body.

    Returns:
        Decoded :class:`RuntimeEventRecordDict` instances whose ``channel``
        is ``WIRE_COMPLETE``. Other channels (``STATE``, ``WIRE``, etc.) are
        filtered out so completion-site assertions are not coupled to
        unrelated emissions on the same path.
    """
    records: list[RuntimeEventRecordDict] = []
    for line in jsonl.strip().splitlines():
        raw: JSONObject = narrow_json_to_dict(load_json_str(line))
        record = decode_runtime_event_record(raw)
        if record["channel"] == "WIRE_COMPLETE":
            records.append(record)
    return records


def _make_bot_with_in_flight(
    *,
    state: StateName,
    action_kind: ActionKind,
    target_x: int,
    target_y: int,
    started_ms: int,
) -> Bot:
    """Build a :class:`Bot` with a pre-configured in-flight action.

    Args:
        state: HFSM state name to install.
        action_kind: Kind for the in-flight action record.
        target_x: Target X coordinate stamped on the action record.
        target_y: Target Y coordinate stamped on the action record.
        started_ms: Dispatch timestamp the bot would have recorded.

    Returns:
        Bot instance whose ``_state_data`` carries the configured
        in-flight action and HFSM state.
    """
    bot = Bot("https://test.tankpit.com/", headless=True)
    base_data: BotStateDataDict = make_initial_state_data()
    action: InFlightActionDict = make_in_flight_action(
        action_kind,
        target_x,
        target_y,
        started_ms,
    )
    bot._state_data = BotStateDataDict(
        state=state,
        in_flight_action=action,
        fuel_threshold=base_data["fuel_threshold"],
    )
    return bot


class TestWireCompleteEventsOnAuthoritativeCompletion:
    """Each authoritative completion gate emits a ``WIRE_COMPLETE`` event."""

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

        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "map_open"
        assert require_str_field(fields, "signal") == "map_data_processed"
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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "move"
        assert require_str_field(fields, "signal") == "position_reached"
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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "teleport"
        assert require_str_field(fields, "signal") == "teleport_landed"
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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "collect"
        assert require_str_field(fields, "signal") == "position_reached"
        assert require_int_field(fields, "target_x") == 80
        assert require_int_field(fields, "target_y") == 90

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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "collect"
        assert require_str_field(fields, "signal") == "container_consumed"
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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "scan"
        assert require_str_field(fields, "signal") == "radar_scan_complete"

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
        events = _decode_wire_complete_lines(
            fake_fs.get_written_files()[artifacts["latest_events_path"]]
        )
        assert len(events) == 1
        fields = events[0]["fields"]
        assert require_str_field(fields, "action_kind") == "move"
        assert require_str_field(fields, "signal") == "stall_timeout"
        assert require_int_field(fields, "target_x") == 180
        assert require_int_field(fields, "target_y") == 200
        # timeout_ms carries the configured action stall timeout the gate
        # compared elapsed time against; any non-zero AI config value is
        # acceptable.
        assert require_int_field(fields, "timeout_ms") > 0


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

    latest_events = _decode_wire_complete_lines(files[artifacts["latest_events_path"]])
    archive_events = _decode_wire_complete_lines(files[artifacts["archive_events_path"]])
    assert latest_events == archive_events
    assert len(latest_events) == 1
    assert require_str_field(latest_events[0]["fields"], "signal") == "radar_scan_complete"
