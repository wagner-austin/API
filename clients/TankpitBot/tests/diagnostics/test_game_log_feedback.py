"""End-to-end tests for game-log world-truth feedback consumption.

Every test drives the REAL pipeline:
:func:`tankpit_bot.runtime_logging.configure_bot_runtime_logging` ->
:func:`tankpit_bot.diagnostics.game_log_feedback.register_world_feedback_from_game_log`
-> real world-state mutations (``remove_container_at``,
``mark_move_target_failed``) + real JSONL artifact via
:class:`tests.conftest.FakeFileSystem`. Nothing is mocked. The feedback
lines mirror the live game log captured in run 20260610 where eight
``Empty container`` lines were discarded while the bot retried a
drained container every tick.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeFileSystem

from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.diagnostics.event_stream import load_event_records
from tankpit_bot.diagnostics.game_log_feedback import (
    record_move_dispatch,
    record_pickup_dispatch,
    register_world_feedback_from_game_log,
    reset_game_log_feedback,
)
from tankpit_bot.protocol import RadarContainerDict, RadarMineDict
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    configure_bot_runtime_logging,
)
from tankpit_bot.sniffer import world_state
from tankpit_bot.sniffer.world_state import get_world_service, is_move_target_failed
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar


def _seed_global_container(x: int, y: int, volume: int) -> None:
    """Track a fuel container in the global sniffer world state."""
    containers: list[RadarContainerDict] = [RadarContainerDict(x=x, y=y, volume=volume)]
    mines: list[RadarMineDict] = []
    update_world_state_from_radar(get_world_service(), containers, mines, [])


def _feedback_records(latest_events_path: str) -> list[RuntimeEventRecordDict]:
    """Return every ``game_log_feedback`` record from the artifact."""
    return [
        record
        for record in load_event_records(Path(latest_events_path))
        if record["fields"].get("diagnostic_kind") == "game_log_feedback"
    ]


def _entries(*texts: str) -> list[GameLogEntry]:
    """Return scraper-shaped entries for the given log line texts."""
    return [GameLogEntry(text=text, category="other") for text in texts]


def test_empty_container_removes_belief_at_pickup_target(fake_fs: FakeFileSystem) -> None:
    """An ``Empty container`` line deletes the contradicted container.

    Live run 20260610 end-state: the bot sat adjacent to (13,172)
    believing vol=81 and re-sent pickup_fuel every tick because the
    wire is silent on failed pickups.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")
    _seed_global_container(13, 172, 81)
    record_pickup_dispatch(13, 172)

    consumed = register_world_feedback_from_game_log(_entries("Empty container"))

    assert consumed == 1
    assert "13,172" not in world_state.get_world_state()["containers"]
    records = _feedback_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "game_log_feedback",
        "feedback": "empty_container",
        "target_x": 13,
        "target_y": 172,
    }


def test_empty_container_without_dispatch_is_a_no_op(fake_fs: FakeFileSystem) -> None:
    """No recorded pickup dispatch means no belief to correct."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    _seed_global_container(13, 172, 81)

    consumed = register_world_feedback_from_game_log(_entries("Empty container"))

    assert consumed == 1
    assert "13,172" in world_state.get_world_state()["containers"]
    assert _feedback_records(artifacts["latest_events_path"]) == []


def test_repeated_empty_container_lines_each_consume(fake_fs: FakeFileSystem) -> None:
    """Each repeated line consumes; removal of a gone container no-ops."""
    configure_bot_runtime_logging("20260610-120000")
    _seed_global_container(13, 172, 81)
    record_pickup_dispatch(13, 172)

    consumed = register_world_feedback_from_game_log(
        _entries("Empty container", "Empty container"),
    )

    assert consumed == 2
    assert "13,172" not in world_state.get_world_state()["containers"]


def test_tank_full_line_is_ignored(fake_fs: FakeFileSystem) -> None:
    """``Tank full`` is a no-op: capacity is rank-derived, not learned.

    The game-log ``Tank full`` line was previously consumed to learn a
    fuel-capacity watermark; that whole channel was retired 2026-07-06
    once the formula ``1000 + 100 * rank`` was cracked from
    ``tpclient.js Gc`` (see :mod:`tankpit_bot.state.rank_formulas`).
    The line now falls through as an unrelated log line.
    """
    artifacts = configure_bot_runtime_logging("20260610-120000")

    consumed = register_world_feedback_from_game_log(_entries("Tank full"))

    assert consumed == 0
    assert _feedback_records(artifacts["latest_events_path"]) == []


def test_blocked_move_marks_move_target_failed(fake_fs: FakeFileSystem) -> None:
    """A ``You can't go there!`` line fails the move target instantly."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    record_move_dispatch(40, 50)

    consumed = register_world_feedback_from_game_log(_entries("You can't go there!"))

    assert consumed == 1
    assert is_move_target_failed(40, 50, get_current_time_ms()) is True
    records = _feedback_records(artifacts["latest_events_path"])
    assert len(records) == 1
    assert records[0]["fields"] == {
        "diagnostic_kind": "game_log_feedback",
        "feedback": "blocked_move",
        "target_x": 40,
        "target_y": 50,
    }


def test_blocked_move_without_dispatch_is_a_no_op(fake_fs: FakeFileSystem) -> None:
    """No recorded move dispatch means no tile to fail."""
    artifacts = configure_bot_runtime_logging("20260610-120000")

    consumed = register_world_feedback_from_game_log(_entries("You can't go there!"))

    assert consumed == 1
    assert _feedback_records(artifacts["latest_events_path"]) == []


def test_unrelated_lines_consume_nothing(fake_fs: FakeFileSystem) -> None:
    """Ordinary log lines are not feedback."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    record_pickup_dispatch(13, 172)
    record_move_dispatch(40, 50)

    consumed = register_world_feedback_from_game_log(
        _entries("You hit red-1", "5 homing shots gained", "Zoom in"),
    )

    assert consumed == 0
    assert _feedback_records(artifacts["latest_events_path"]) == []


def test_reset_clears_dispatch_targets(fake_fs: FakeFileSystem) -> None:
    """``reset_game_log_feedback`` restores the fresh-process state."""
    artifacts = configure_bot_runtime_logging("20260610-120000")
    _seed_global_container(13, 172, 81)
    record_pickup_dispatch(13, 172)
    record_move_dispatch(40, 50)

    reset_game_log_feedback()

    consumed = register_world_feedback_from_game_log(
        _entries("Empty container", "You can't go there!"),
    )
    assert consumed == 2
    assert "13,172" in world_state.get_world_state()["containers"]
    assert is_move_target_failed(40, 50, get_current_time_ms()) is False
    assert _feedback_records(artifacts["latest_events_path"]) == []
