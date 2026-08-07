"""End-to-end replay test: drive the real TeleportProbe through a capture.

The harness exercises the production
:class:`tankpit_bot.action_lab.teleport.TeleportProbe` end-to-end.
Real decoders ingest the recorded bytes; the real
``_probe_single_target`` body sequences through map_open / teleport;
real ``capture_page_client_snapshot`` calls hit the
:class:`WorldStateDerivedCDP` substitute; ``_send_bytes`` is captured.

No fakes of probe behavior. The only substitutes are at the OS-level
boundary -- the same boundary the live tank-pit page is on top of.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.action_lab._replay_teleport import (
    TeleportReplayResult,
    replay_teleport_attempt,
)

from tankpit_bot.action_lab.teleport import TeleportProbeError
from tankpit_bot.action_lab.types import TeleportTargetDict

REPO_ROOT = Path(__file__).resolve().parents[2]
TELEPORT_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"


@pytest.fixture()
def teleport_attempt() -> TeleportReplayResult:
    """Replay one teleport attempt against the committed capture."""
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    return replay_teleport_attempt(
        TELEPORT_CAPTURE,
        target,
        map_sync_timeout_ms=600_000,
        teleport_timeout_ms=600_000,
    )


def test_replay_consumes_recorded_frames(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """Recorded frames feed the real probe through wait_for_timeout polls."""
    assert teleport_attempt.frames_fed > 0


def test_replay_dispatches_real_map_open_command(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """The real probe issues a ``map_open`` command through real ``open_map``."""
    assert "map_open" in teleport_attempt.dispatched_commands


def test_replay_attempt_records_target(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """The real attempt result preserves the supplied target verbatim."""
    target = teleport_attempt.attempt["target"]
    assert target["label"] == "recorded_pivot"
    assert target["x"] == 131
    assert target["y"] == 110


def test_replay_attempt_reports_a_resolved_status(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """The production attempt body reaches one of its declared terminal statuses."""
    assert teleport_attempt.attempt["status"] in (
        "landed_exact",
        "landed_offset",
        "map_sync_timeout",
        "teleport_timeout",
    )


def test_replay_attempt_records_fuel_before(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """The attempt records the real fuel snapshot from decoded frames."""
    fuel_before = teleport_attempt.attempt["fuel_before"]
    assert fuel_before >= 0


def test_replay_attempt_carries_real_snapshots(
    teleport_attempt: TeleportReplayResult,
) -> None:
    """Page-client snapshots come from the real capture path."""
    snapshots = teleport_attempt.attempt["page_snapshots"]
    if not snapshots:
        pytest.fail(
            "expected at least one page-client snapshot to be captured "
            "during the real teleport attempt body"
        )
    assert all(s["client_present"] is True for s in snapshots)
    first_phase = snapshots[0]["phase"]
    assert first_phase in (
        "before_map_open",
        "before_teleport",
        "after_map_data",
        "landed",
        "timeout",
    )


def test_replay_raises_when_cdp_session_unavailable() -> None:
    """Real probe attempt fails fast when CDP is not attached."""
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        replay_teleport_attempt(
            TELEPORT_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            omit_cdp=True,
        )


def test_replay_raises_on_map_open_dispatch_failure() -> None:
    """The real probe raises ``map_open command dispatch failed`` end-to-end.

    Simulates a live WebSocket-send failure by failing the real
    ``_send_bytes`` for the ``map_open`` label.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(TeleportProbeError, match="map_open command dispatch failed"):
        replay_teleport_attempt(
            TELEPORT_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            fail_command=lambda cmd: cmd == "map_open",
        )


def test_replay_raises_on_teleport_dispatch_failure() -> None:
    """The real probe raises ``teleport command dispatch failed`` end-to-end.

    Fails the real ``_send_bytes`` for any ``teleport(...)`` label.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(TeleportProbeError, match="teleport command dispatch failed"):
        replay_teleport_attempt(
            TELEPORT_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            fail_command=lambda cmd: cmd.startswith("teleport("),
        )
