"""End-to-end replay test: drive the real MovementProbe through a capture.

The harness exercises the production
:class:`tankpit_bot.action_lab.movement_probe.MovementProbe` end-to-end.
Real decoders ingest the recorded bytes; the real
``_probe_single_movement_target`` body runs; the real
``_wait_for_move_outcome`` loop polls the real world-state singletons;
``capture_page_client_snapshot`` reads through the real CDP send path;
``_send_bytes`` is captured (rather than dispatched to a non-existent
live server).

No fakes of probe behavior. The only substitutes are at the OS-level
boundary -- the same boundary the live tank-pit page is on top of.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.action_lab._replay_core import ReplayResult
from tests.action_lab._replay_harness import replay_movement_attempt

from tankpit_bot.action_lab.movement_probe import MovementProbeError
from tankpit_bot.action_lab.movement_probe_types import MovementProbeAttemptResultDict
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import reset_xor_state

REPO_ROOT = Path(__file__).resolve().parents[2]
FUEL_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"


@pytest.fixture(autouse=True)
def _isolate_world_state() -> Generator[None, None, None]:
    """Reset world-state and XOR singletons around every test.

    The replay harness builds a global XOR table from the capture's
    magic key; without an explicit teardown reset, that table would
    leak into subsequent tests that decode bytes with a different key.
    """
    reset_world_state()
    reset_xor_state()
    yield
    reset_world_state()
    reset_xor_state()


@pytest.fixture()
def landed_position_replay() -> ReplayResult[MovementProbeAttemptResultDict]:
    """Replay a move-to-landed-position attempt against the fuel-probe capture.

    The recorded session ends with the tank at (146, 110); a move
    targeting that exact position lets the production
    ``_wait_for_move_outcome`` see ``arrived_exact`` once the recorded
    frames have driven world state to that final tile.
    """
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    return replay_movement_attempt(
        FUEL_CAPTURE,
        target,
        move_timeout_ms=600_000,
        frames_per_wait=5,
        initial_sync_batches=10,
    )


def test_replay_consumes_recorded_frames(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """Recorded frames feed the real probe through wait_for_timeout polls."""
    assert landed_position_replay.frames_fed > 0


def test_replay_dispatches_real_move_command(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """The real probe issues a ``move`` command through real ``move_to``."""
    assert "move" in landed_position_replay.dispatched_commands


def test_replay_attempt_records_target(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """The real attempt result preserves the supplied target verbatim."""
    target = landed_position_replay.attempt["target"]
    assert target["label"] == "recorded_final_tile"
    assert target["x"] == 146
    assert target["y"] == 110


def test_replay_attempt_carries_real_snapshots(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """Snapshots come from the real ``capture_page_client_snapshot`` path."""
    snapshot_before = landed_position_replay.attempt["snapshot_before"]
    snapshot_after = landed_position_replay.attempt["snapshot_after"]
    assert snapshot_before["client_present"] is True
    assert snapshot_after["client_present"] is True


def test_replay_attempt_reports_a_resolved_status(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """Production ``_wait_for_move_outcome`` reaches a terminal status."""
    assert landed_position_replay.attempt["status"] in (
        "arrived_exact",
        "move_timeout",
    )


def test_replay_attempt_records_self_position(
    landed_position_replay: ReplayResult[MovementProbeAttemptResultDict],
) -> None:
    """The real attempt records the bot's final decoded position."""
    settled_x = landed_position_replay.attempt["settled_x"]
    settled_y = landed_position_replay.attempt["settled_y"]
    if settled_x is None or settled_y is None:
        pytest.fail("expected settled position to be populated from real recorded frames")
    assert 0 <= settled_x <= 255
    assert 0 <= settled_y <= 255


def test_replay_unreachable_target_records_move_timeout() -> None:
    """A target the recording never reaches is a real ``move_timeout``.

    Drives the real ``_wait_for_move_outcome`` timeout branch end-to-end
    without any fake substitution -- production code observes the
    timeout because the recorded world state never visits the requested
    tile.
    """
    target = TeleportTargetDict(label="unreachable", x=200, y=200)
    result = replay_movement_attempt(
        FUEL_CAPTURE,
        target,
        move_timeout_ms=2_000,
        frames_per_wait=5,
        initial_sync_batches=10,
    )

    assert result.attempt["status"] == "move_timeout"
    assert "move" in result.dispatched_commands


def test_replay_with_map_open_queued_dispatches_both_commands() -> None:
    """``queue_map_open_during_move`` exercises the real map_open branch."""
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    result = replay_movement_attempt(
        FUEL_CAPTURE,
        target,
        move_timeout_ms=600_000,
        frames_per_wait=5,
        initial_sync_batches=10,
        queue_map_open_during_move=True,
        map_open_delay_ms=100,
    )

    assert "move" in result.dispatched_commands
    assert "map_open" in result.dispatched_commands


def test_replay_with_settle_delay_invokes_real_wait_for_timeout() -> None:
    """``settle_delay_ms > 0`` adds a final ``page.wait_for_timeout`` call.

    Exercises the real settle-delay branch through the actual probe
    attempt body. The harness's ``ReplayPage.wait_for_timeout`` records
    every duration, so the post-outcome settle wait shows up in the
    timeline.
    """
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    result = replay_movement_attempt(
        FUEL_CAPTURE,
        target,
        move_timeout_ms=600_000,
        frames_per_wait=5,
        initial_sync_batches=10,
        settle_delay_ms=200,
    )

    assert result.waits_ms[-1] == 200.0


def test_replay_raises_when_cdp_session_unavailable() -> None:
    """Real probe attempt fails fast when CDP is not attached.

    Without a CDP session, the page-client snapshot cannot be
    captured. The production attempt body raises immediately --
    exercised here through the real attempt without behavior fakes.
    """
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    with pytest.raises(MovementProbeError, match="cdp session is unavailable"):
        replay_movement_attempt(
            FUEL_CAPTURE,
            target,
            move_timeout_ms=600_000,
            frames_per_wait=5,
            initial_sync_batches=10,
            omit_cdp=True,
        )


def test_replay_raises_on_move_dispatch_failure() -> None:
    """When ``_send_bytes("move")`` returns False, the real probe raises.

    Simulates a live WebSocket-send failure by failing the real
    ``_send_bytes`` for the ``move`` label -- exactly what the live
    runtime does when the socket is down.
    """
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    with pytest.raises(MovementProbeError, match="move command dispatch failed"):
        replay_movement_attempt(
            FUEL_CAPTURE,
            target,
            move_timeout_ms=600_000,
            frames_per_wait=5,
            initial_sync_batches=10,
            fail_command=lambda cmd: cmd == "move",
        )


def test_replay_raises_on_map_open_dispatch_failure_during_move() -> None:
    """When queued ``map_open`` dispatch fails mid-move, the probe raises.

    Exercises the real ``queue_map_open_during_move=True`` branch by
    failing the real ``_send_bytes`` for the ``map_open`` label.
    """
    target = TeleportTargetDict(label="recorded_final_tile", x=146, y=110)
    with pytest.raises(
        MovementProbeError,
        match="map_open command dispatch failed during movement probe",
    ):
        replay_movement_attempt(
            FUEL_CAPTURE,
            target,
            move_timeout_ms=600_000,
            frames_per_wait=5,
            initial_sync_batches=10,
            queue_map_open_during_move=True,
            fail_command=lambda cmd: cmd == "map_open",
        )
