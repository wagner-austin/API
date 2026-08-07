"""Tests for the replay engine."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.tick_loop_types import make_tick_decision
from tankpit_bot.bot.types import (
    make_hold_command,
    make_map_open_command,
    make_move_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.replay.engine import (
    _build_trace,
    _extract_command_target,
    _process_tick_batch,
    replay_session,
)
from tankpit_bot.replay.types import ReplayTickTraceDict
from tankpit_bot.state.types import (
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, CaptureSession
from tankpit_bot.types.literals import MessageDirection


def _make_session(
    messages: list[CapturedMessage],
    magic: str | None = "testmagic",
) -> CaptureSession:
    """Create a capture session for replay tests."""
    return CaptureSession(
        session_id="replay-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


def _make_text_message(
    timestamp_ms: int,
    text: str,
    direction: MessageDirection = "received",
) -> CapturedMessage:
    """Create a base64-encoded text message for testing.

    Text messages use a 2-byte LE length prefix followed by the text body.

    Args:
        timestamp_ms: Message timestamp.
        text: Text content of the message.
        direction: Message direction.

    Returns:
        CapturedMessage with base64-encoded payload.
    """
    body = text.encode("utf-8")
    length = len(body)
    frame = bytes([length & 0xFF, (length >> 8) & 0xFF]) + body
    payload = base64.b64encode(frame).decode("ascii")
    return CapturedMessage(
        timestamp_ms=timestamp_ms,
        direction=direction,
        payload=payload,
        ws_url="wss://tankpit.com/ws",
    )


_REPLAY_TABLE = build_session_xor_table("testmagic")
"""The table these tests' fixture sessions (``magic="testmagic"``) use.

``_process_tick_batch`` takes the table as a parameter now, so tests
pass it explicitly instead of leaning on module state
([[session-state-deglobalisation]])."""


def _cleanup() -> None:
    """Reset global state between tests.

    The XOR table is no longer among it — each replay builds its own
    from the capture's magic ([[session-state-deglobalisation]]).
    """


def _require_trace(
    result: tuple[AIStateDict, ReplayTickTraceDict | None],
) -> ReplayTickTraceDict:
    """Extract the trace from a _process_tick_batch result, failing if None.

    Args:
        result: Tuple of (ai_state, trace_or_none) from _process_tick_batch.

    Returns:
        The non-None trace.

    Raises:
        ValueError: If trace is None.
    """
    trace = result[1]
    if trace is None:
        raise ValueError("Expected a trace but got None")
    return trace


class TestReplaySessionValidation:
    """Tests for replay_session input validation."""

    def test_no_magic_raises_value_error(self) -> None:
        """replay_session raises ValueError when magic is None."""
        session = _make_session([], magic=None)
        with pytest.raises(ValueError, match="Cannot replay session without magic key"):
            replay_session(session)

    def test_empty_messages_returns_zero_ticks(self) -> None:
        """replay_session returns zero ticks for session with no messages."""
        _cleanup()
        session = _make_session([])
        result = replay_session(session)
        assert result["session_id"] == "replay-test"
        assert result["total_ticks"] == 0
        assert result["total_messages"] == 0
        assert result["traces"] == []
        _cleanup()

    def test_sent_only_messages_returns_zero_ticks(self) -> None:
        """replay_session skips sent messages and returns zero ticks."""
        _cleanup()
        session = _make_session(
            [
                _make_text_message(1000, "+1|Room|5|1,1,1,0,1,0,0|3|n|field42.gif|2026", "sent"),
            ]
        )
        result = replay_session(session)
        assert result["total_messages"] == 0
        assert result["total_ticks"] == 0
        _cleanup()

    def test_text_messages_no_self_state_returns_zero_ticks(self) -> None:
        """replay_session processes text messages but returns no ticks."""
        _cleanup()
        session = _make_session(
            [
                _make_text_message(1000, "+1|Room|5|1,1,1,0,1,0,0|3|n|field42.gif|2026"),
                _make_text_message(1100, "+2|Other|3|1,1,1,0,1,0,0|3|n|field43.gif|2026"),
            ]
        )
        result = replay_session(session)
        assert result["total_messages"] == 2
        assert result["total_ticks"] == 0
        _cleanup()


class TestExtractCommandTarget:
    """Tests for _extract_command_target."""

    def test_move_command(self) -> None:
        """Extracts coordinates from move command."""
        cmd = make_move_command(50, 60)
        assert _extract_command_target(cmd) == (50, 60)

    def test_radar_command(self) -> None:
        """Returns (0, 0) for radar command."""
        cmd = make_radar_command()
        assert _extract_command_target(cmd) == (0, 0)

    def test_map_open_command(self) -> None:
        """Returns (0, 0) for map_open command."""
        cmd = make_map_open_command()
        assert _extract_command_target(cmd) == (0, 0)

    def test_hold_command(self) -> None:
        """Returns (0, 0) for the SPA-idle hold command."""
        cmd = make_hold_command()
        assert _extract_command_target(cmd) == (0, 0)

    def test_teleport_command(self) -> None:
        """Extracts coordinates from teleport command."""
        cmd = make_teleport_command(120, 130)
        assert _extract_command_target(cmd) == (120, 130)


class TestBuildTrace:
    """Tests for _build_trace."""

    def test_builds_complete_trace(self) -> None:
        """_build_trace produces a complete ReplayTickTraceDict."""
        self_state = make_self_state(
            x=100,
            y=120,
            fuel=500,
            team=0,
            tank_id=1,
            rank=3,
            leaderboard_position=0,
        )
        world = make_empty_world_state()
        world = WorldStateDict(**{**world, "self_state": self_state})
        ai_state = make_initial_ai_state()
        decision = make_tick_decision(
            command=make_map_open_command(),
            behavior=make_behavior_score("HUNT", 0, 0, 0, "find_enemies"),
            updated_ai_state=ai_state,
            desired_equipment=[1, 2, 4, 5],
        )

        trace = _build_trace(0, 1000, decision, world, self_state, ai_state)

        assert trace["tick_index"] == 0
        assert trace["timestamp_ms"] == 1000
        assert trace["self_x"] == 100
        assert trace["self_y"] == 120
        assert trace["fuel"] == 500
        assert trace["behavior_mode"] == "HUNT"
        assert trace["behavior_score"] == 0
        assert trace["behavior_reason"] == "find_enemies"
        assert trace["command_type"] == "map_open"
        assert trace["target_x"] == 0
        assert trace["target_y"] == 0
        assert trace["ai_mode"] == "UNSET"
        assert trace["ai_mode_state"] == ""
        assert trace["combat_target_id"] == -1
        assert trace["resource_target_kind"] == ""
        assert trace["visible_threats"] == []
        assert trace["container_count"] == 0

    def test_builds_trace_with_containers(self) -> None:
        """_build_trace counts containers in world state."""
        self_state = make_self_state(
            x=50,
            y=60,
            fuel=800,
            team=0,
            tank_id=1,
            rank=3,
            leaderboard_position=0,
        )
        world = make_empty_world_state()
        containers = {
            "50,60": make_container_state(50, 60, True, 500, timestamp_ms=900),
            "52,63": make_container_state(52, 63, False, 0, timestamp_ms=900),
        }
        world = WorldStateDict(
            **{**world, "self_state": self_state, "containers": containers},
        )
        ai_state = make_initial_ai_state()
        decision = make_tick_decision(
            command=make_move_command(52, 63),
            behavior=make_behavior_score("COLLECT", 900, 52, 63, "fuel_collect"),
            updated_ai_state=ai_state,
            desired_equipment=[],
        )

        trace = _build_trace(1, 2000, decision, world, self_state, ai_state)

        assert trace["container_count"] == 2
        assert trace["target_x"] == 52
        assert trace["target_y"] == 63

    def test_builds_trace_with_durable_hunt_state(self) -> None:
        """_build_trace captures durable hunt mode state and target."""
        self_state = make_self_state(
            x=100,
            y=100,
            fuel=150,
            team=0,
            tank_id=1,
            rank=3,
            leaderboard_position=0,
        )
        world = make_empty_world_state()
        world = WorldStateDict(**{**world, "self_state": self_state})
        ai_state = AIStateDict(
            **{
                **make_initial_ai_state(),
                "mode": "HUNT",
                "mode_state": "ENGAGE",
                "combat_target_id": 42,
                "resource_target_kind": "",
            },
        )
        decision = make_tick_decision(
            command=make_map_open_command(),
            behavior=make_behavior_score("HUNT", 950, 101, 100, "shoot_target"),
            updated_ai_state=ai_state,
            desired_equipment=[1, 2, 4],
        )

        trace = _build_trace(7, 8000, decision, world, self_state, ai_state)

        assert trace["ai_mode"] == "HUNT"
        assert trace["ai_mode_state"] == "ENGAGE"
        assert trace["combat_target_id"] == 42
        assert trace["resource_target_kind"] == ""


def _inject_self_state(x: int, y: int, fuel: int) -> None:
    """Inject a self_state into the module-level world state for testing.

    Args:
        x: X coordinate.
        y: Y coordinate.
        fuel: Fuel amount.
    """
    from tankpit_bot.sniffer.world_state import get_world_service

    svc = get_world_service()
    self_state = make_self_state(
        x=x,
        y=y,
        fuel=fuel,
        team=0,
        tank_id=1,
        rank=3,
        leaderboard_position=0,
    )
    svc.world_state = WorldStateDict(**{**svc.world_state, "self_state": self_state})


class TestProcessTickBatch:
    """Tests for _process_tick_batch with injected world state."""

    def test_returns_none_trace_when_no_self_state(self) -> None:
        """_process_tick_batch returns None trace when self_state is absent."""
        _cleanup()
        ai_state = make_initial_ai_state()
        result = _process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 1000)
        assert result[1] is None
        _cleanup()

    def test_returns_trace_when_self_state_present(self) -> None:
        """_process_tick_batch returns a trace when self_state is available.

        Default inventory has all weapons at zero so the durable owner
        enters ``COLLECT``. With no tile-coverage and radar
        affordable, the forager dispatches ``forage_radar`` -- the
        substate derivation routes that to ``SENSE`` (radar = sensing
        the viewport), independent of whether the server-side extras
        count is empty (free 5x5) or stocked (full viewport).
        """
        _cleanup()
        _inject_self_state(100, 120, 500)
        ai_state = make_initial_ai_state()
        trace = _require_trace(_process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 1000))
        assert trace["self_x"] == 100
        assert trace["self_y"] == 120
        assert trace["fuel"] == 500
        assert trace["tick_index"] == 0
        assert trace["ai_mode"] == "COLLECT"
        assert trace["ai_mode_state"] == "SENSE"
        assert trace["combat_target_id"] == -1
        _cleanup()

    def test_carries_forward_updated_ai_state(self) -> None:
        """_process_tick_batch returns updated AI state."""
        _cleanup()
        _inject_self_state(100, 120, 500)
        ai_state = make_initial_ai_state()
        result = _process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 1000)
        updated_ai = result[0]
        assert updated_ai["config"] == ai_state["config"]
        _cleanup()

    def test_merges_kills_into_ai_state(self) -> None:
        """_process_tick_batch merges killed tank IDs from protocol."""
        _cleanup()
        from tankpit_bot.sniffer.world_state import get_world_service

        get_world_service().killed_tank_ids.add(42)
        _inject_self_state(100, 120, 500)
        ai_state = make_initial_ai_state()
        result = _process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 5000)
        updated_ai = result[0]
        assert "42" in updated_ai["killed_tank_ids"]
        assert updated_ai["killed_tank_ids"]["42"] == 5000
        _cleanup()


class TestReplaySessionMultiTick:
    """Tests for replay_session with multiple tick batches."""

    def test_multi_tick_with_traces(self) -> None:
        """replay_session produces traces when hook injects self_state.

        Uses the process_received_message_hook to inject self_state
        during message processing so the planner runs.
        """
        _cleanup()
        from tankpit_bot.sniffer.decoders import process_received_message as real_prm
        from tankpit_bot.sniffer.world_state import get_world_service

        call_count = 0

        def _injecting_hook(payload: str, xor_table: bytes) -> None:
            nonlocal call_count
            real_prm(get_world_service(), payload, xor_table)
            call_count += 1
            if call_count == 1:
                svc = get_world_service()
                self_state = make_self_state(
                    x=70,
                    y=80,
                    fuel=150,
                    team=0,
                    tank_id=1,
                    rank=3,
                    leaderboard_position=0,
                )
                svc.world_state = WorldStateDict(
                    **{**svc.world_state, "self_state": self_state},
                )

        from tankpit_bot import _test_hooks

        original = _test_hooks.process_received_message_hook
        _test_hooks.process_received_message_hook = _injecting_hook

        session = _make_session(
            [
                _make_text_message(1000, "+1|Room|5|1,1,1,0,1,0,0|3|n|field42.gif|2026"),
                _make_text_message(4000, "+2|Other|3|1,1,1,0,1,0,0|3|n|field43.gif|2026"),
                _make_text_message(7000, "+3|Third|2|1,1,1,0,1,0,0|3|n|field44.gif|2026"),
            ]
        )
        result = replay_session(session)

        _test_hooks.process_received_message_hook = original

        assert result["total_messages"] == 3
        assert result["total_ticks"] >= 2
        assert result["traces"][0]["self_x"] == 70
        assert result["traces"][0]["ai_mode"] == "COLLECT"
        assert result["traces"][0]["ai_mode_state"] in ("SENSE", "APPROACH", "")
        _cleanup()

    def test_multi_tick_batching(self) -> None:
        """replay_session batches messages into ticks by timestamp gap.

        TICK_RATE_MS is 2000ms, so messages must be >2s apart to form
        separate tick batches.
        """
        _cleanup()
        session = _make_session(
            [
                _make_text_message(1000, "+1|Room|5|1,1,1,0,1,0,0|3|n|field42.gif|2026"),
                _make_text_message(1500, "+2|Other|3|1,1,1,0,1,0,0|3|n|field43.gif|2026"),
                _make_text_message(5000, "+3|Third|2|1,1,1,0,1,0,0|3|n|field44.gif|2026"),
            ]
        )
        result = replay_session(session)
        assert result["total_messages"] == 3
        assert result["total_ticks"] == 0
        _cleanup()

    def test_produces_traces_with_injected_self_state(self) -> None:
        """Multi-tick planner execution produces correct traces."""
        _cleanup()
        _inject_self_state(80, 90, 600)
        ai_state = make_initial_ai_state()
        traces: list[ReplayTickTraceDict] = []

        result1 = _process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 2000)
        ai_state = result1[0]
        if result1[1] is not None:
            traces.append(result1[1])

        result2 = _process_tick_batch([], _REPLAY_TABLE, ai_state, 1, 4100)
        if result2[1] is not None:
            traces.append(result2[1])

        assert len(traces) == 2
        assert traces[0]["tick_index"] == 0
        assert traces[1]["tick_index"] == 1
        assert traces[0]["self_x"] == 80
        _cleanup()

    def test_final_batch_produces_trace(self) -> None:
        """The final batch in replay produces a trace when self_state exists."""
        _cleanup()
        _inject_self_state(70, 80, 400)
        ai_state = make_initial_ai_state()
        result1 = _process_tick_batch([], _REPLAY_TABLE, ai_state, 0, 2000)
        trace1 = _require_trace(result1)
        assert trace1["tick_index"] == 0

        result2 = _process_tick_batch([], _REPLAY_TABLE, result1[0], 1, 5000)
        trace2 = _require_trace(result2)
        assert trace2["tick_index"] == 1
        assert trace2["self_x"] == 70
        assert trace2["self_y"] == 80
        _cleanup()
